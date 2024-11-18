from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np


# Some constants #

EMBEDDING_SIZE = 768 # this is decided by the model I used to embed the text
NUM_HEADS = 2  # this is decided by me! it can be changed, but embedding_size/num_heads must be an integer
NUM_LAYERS = 2 # number of decoder layers
HIDDEN_LAYER_SIZE = EMBEDDING_SIZE * 4 # dimensionality of hidden layer in the FFNs. should be bigger than EMBEDDING_SIZE
VOCAB_SIZE = 50257
SEQUENCE_LENGTH = 16


# Some simple math functions #

# the softmax function. takes a vector, returns a vector
def softmax(vector):
    expVec = np.exp(vector)
    sum = expVec.sum()
    if (sum == 0): # masked-out rows will sum to 0; let's not divide by 0
        return expVec
    return expVec / sum

# ReLU activation, applied element-wise. 
# this operates in place so it's not necessary to return anything, but will return matrix just in case
def reLU(matrix):
    np.maximum(matrix, 0, matrix)
    return matrix

# layer normalization. returns normalized matrix
# technically the result should have a learned weight and bias. maybe I will implement when I have the time
def layerNorm(matrix):
    mean = np.mean(matrix, axis=1, keepdims=True)
    std = np.std(matrix, axis=1, keepdims=True)
    return (matrix - mean) / (std + 0.00005) # the tiny value avoids div by 0


# Mask generation #

# the tokenizer provides attention masks which use 1s and 0s to indicate location of padding tokens
# this function converts those vectors into full matrices and changes the values to work with softmax
def inputPaddingMasks(attention_masks):
    padding_masks = []
    for vec in attention_masks:
        full_mask = np.zeros((len(vec), len(vec)))
        for t in range(len(vec)):
            if (vec[t] == 0):
                full_mask[t] = -np.inf
                full_mask[:, t] = -np.inf
        padding_masks.append(full_mask)
    return padding_masks

# this function returns a causal mask of size (sequence_length, sequence_length)
# this mask prevents attention from 'peeking ahead' during training
def causalMask(sequence_length):
    mask = np.tril(np.ones((sequence_length, sequence_length)))
    return np.where(mask == 0, -np.inf, 0)    


# Data retrieval and token embedding #

# import the imdb dataset, split into training and test
# TODO: host this on UTD website
def retriveData():
    ds = load_dataset("stanfordnlp/imdb", split="train[:10]")
    ds = ds.train_test_split(test_size=0.2)
    return ds

# text embedding
# ds: dataset | max_len: how many tokens long each instance will be
# returns: - a 3D matrix of shape (number of instances, max_len, embedding dimension)
#          - a corresponding 3D matrix of padding masks
def embedData(ds):
    # load a pretrained tokenizer and model to embed the data into vectors readable by the transformer
    # 'model' is itsself a transformer, however I am only using it for text embedding, not any actual transformer-ing
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    model = AutoModel.from_pretrained("distilbert/distilgpt2")

    tokenizer.pad_token = tokenizer.eos_token # set the padding token to the end-of-sentence token
    inputs = tokenizer(ds, max_length=SEQUENCE_LENGTH, return_tensors="pt", padding="max_length", truncation=True)

    # get embeddings by passing inputs through the model
    with torch.no_grad(): # this tells the model not to update its weights, since I'm just getting embeddings from it.
        outputs = model(**inputs)

    # the embeddings are in outputs.last_hidden_state
    token_embeddings = outputs.last_hidden_state
    return token_embeddings.numpy(), inputPaddingMasks(inputs['attention_mask'])

# given a matrix of (sequence length, word probabilities), this will convert into tokens then into words!
def deTokenize(probs):
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    
    tokens = np.argmax(probs, axis=1)
    return tokenizer.decode(tokens)


# Weight-related stuff #

# generates random weights using a normal distribution centered around 0, and saves them to a file
# you can specify the file if you want. should be a .npz
def generateRandomWeights(fileName="weights.npz"):
    weights = {}
    # each layer gets its own weights for self-attention and for its feed-forward network
    attention_matrix_names = ["Wq", "Wk", "Wv"]
    for l in range(NUM_LAYERS):
        # generate the weights for self-attention
        for a in attention_matrix_names:
            name = "attention{l}_{mat}".format(l=l+1, mat=a)
            # the scale (standard deviation) is approximately equal to sqrt(2/embedding size)
            weights[name] = np.random.normal(loc=0.0, scale=0.05, size=(EMBEDDING_SIZE, EMBEDDING_SIZE))
        # generate the weights for the feed forward neural networks
        name = "fnn{l}".format(l=l+1)
        weights[name + "_W1"] = np.random.normal(loc=0.0, scale=0.01, size=(EMBEDDING_SIZE, HIDDEN_LAYER_SIZE))
        weights[name + "_W2"] = np.random.normal(loc=0.0, scale=0.01, size=(HIDDEN_LAYER_SIZE, EMBEDDING_SIZE))
        weights[name + "_b1"] = np.zeros(HIDDEN_LAYER_SIZE)
        weights[name + "_b2"] = np.zeros(EMBEDDING_SIZE)
        
    # we have one last matrix for the final linear layer
    weights['final_layer'] = np.random.uniform(low=-0.1, high=0.1, size=(EMBEDDING_SIZE, VOCAB_SIZE))
    
    # save them to a file, to be loaded another day!
    np.savez(fileName, **weights)
    
# i'll let you guess what this one does
def loadWeights(fileName="weights.npz"):
    try:
        return np.load(fileName)
    except:
        print("Error loading weights file. If none exists, generate one using the generateRandomWeights() method!")


# Algorithms (the real stuff) #
    
# this guy right here is the brains of this whole operation. they say that attention is all you need
# data is the embedded input sequence. wQ/K/V are weights, mask is the padding mask for the input
def selfAttention(data, wq, wk, wv, mask):
    head_outputs = []
    # multiply data by the weight matrices to get Q, K, V
    big_query = data @ wq
    big_key = data @ wk
    big_value = data @ wv
    # now split them into equal sub-matrices, one for each head
    all_queries = np.hsplit(big_query, NUM_HEADS)
    all_keys = np.hsplit(big_key, NUM_HEADS)
    all_values = np.hsplit(big_value, NUM_HEADS)
    # now do the attention calculations for each head
    for query, key, value in zip(all_queries, all_keys, all_values):
        scores = query @ key.T # QK^T
        scores /= np.sqrt(EMBEDDING_SIZE/NUM_HEADS) # QK^T / sqrt(d)
        # add padding and causal masks
        scores += mask
        scores += causalMask(len(mask))
         # apply softmax
        for row in range(scores.shape[0]):
            scores[row] = softmax(scores[row])
        # append this head's output to the list
        head_outputs.append(scores @ value)
        
    # put all the head outputs back together to get our final product
    return np.concatenate(head_outputs, axis=1)

# feed forward neural network. just a normal NN with one hidden layer and ReLU activation
# data is the matrix of attention scores. w1, w2, b1, b2 are weights and biases corresponding to each layer
def feedForward(data, w1, w2, b1, b2):
    # hidden layer
    out_mat = data @ w1
    out_mat += b1
    reLU(out_mat)
    # output layer
    out_mat = out_mat @ w2
    out_mat += b2
    return out_mat

# the logic for a whole decoder layer
# data is a matrix (sequence length, embedding size); mask is the corresponding padding mask;
# weights is the dictionary of all weights; layer_num is (1 : NUM_LAYERS)
# returns a matrix of the same shape as data but with a whole bunch of stuff done to it
def decoder(data, mask, weights, layer_num):
    # load the weights for this layer
    wq = weights['attention{}_Wq'.format(layer_num)]
    wk = weights['attention{}_Wk'.format(layer_num)]
    wv = weights['attention{}_Wv'.format(layer_num)]
    fnn_W1 = weights['fnn{}_W1'.format(layer_num)]
    fnn_W2 = weights['fnn{}_W2'.format(layer_num)]
    fnn_b1 = weights['fnn{}_b1'.format(layer_num)]
    fnn_b2 = weights['fnn{}_b2'.format(layer_num)]
    
    # self-attention! yippee!
    attention_scores = selfAttention(data, wq, wk, wv, mask)
    
    # residual connection
    attention_scores += data
    attention_scores = layerNorm(attention_scores)
    
    fnn_results = feedForward(attention_scores, fnn_W1, fnn_W2, fnn_b1, fnn_b2)
    
    # residual connection
    fnn_results += attention_scores
    fnn_results = layerNorm(fnn_results)
    
    return fnn_results

# this final linear layer transforms the matrix into a list of probabilities for each word in the sequence
def finalLayer(data, weights):
    probs = data @ weights['final_layer']
    for vector in probs:
        vector = softmax(vector)
    return probs

# this function puts it all together. just input a dataset and it will return some words
def forwardPass(ds):
    weights = loadWeights() # load the weights
    
    embeddings, masks = embedData(ds) # embed the data
    # then, for each instance in the dataset, run through the transformer
    results = []
    for data, mask in zip(embeddings, masks):
        for layer_num in range(1, NUM_LAYERS + 1):
            data = decoder(data, mask, weights, layer_num)
        probs = finalLayer(data, weights)
        results.append(deTokenize(probs))
        
    return results