import backprop as bp
import numpy as np
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
import torch
import time


# Some constants #

EMBEDDING_SIZE = 768 # this is decided by the model I used to embed the text
NUM_HEADS = 2  # this is decided by me! it can be changed, but embedding_size/num_heads must be an integer
NUM_LAYERS = 2 # number of decoder layers
HIDDEN_LAYER_SIZE = EMBEDDING_SIZE * 4 # dimensionality of hidden layer in the FFNs. should be bigger than EMBEDDING_SIZE
VOCAB_SIZE = 50257
PAD_TOKEN = 50256
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
# returns the input matrix with ReLU applied
def reLU(matrix):
    return np.maximum(matrix, 0)

# layer normalization. returns normalized matrix and calculations dict for backpropagation
# technically the result should have a learned weight and bias. maybe I will implement later
def layerNorm(matrix):
    calculations = {'input' : matrix, 'epsilon' : 0.00005}
    calculations['mean'] = np.mean(matrix, axis=1, keepdims=True)
    calculations['std'] = np.std(matrix, axis=1, keepdims=True)
    return (matrix - calculations['mean']) / (calculations['std'] + calculations['epsilon']), calculations


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
# you may specify how many total instances are retrieved (before split)
def retriveData(instances):
    ds = load_dataset("stanfordnlp/imdb", split="train[:{}]".format(instances))
    ds = ds.train_test_split(test_size=0.2)
    return ds

# tokenization. handled by a library. returns a list of tokens for each instance in the dataset
def tokenize(ds):
    # load a pretrained tokenizer and use it. I consider this pre-processing in the context of the project
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token # set the padding token to the end-of-sentence token
    inputs = tokenizer(ds, max_length=SEQUENCE_LENGTH, return_tensors="pt", padding="max_length", truncation=True)
    
    return inputs['input_ids'], inputPaddingMasks(inputs['attention_mask'])

# text embedding
# inputs - a list of sequences of tokens
# returns: - a 3D matrix of shape (number of instances, max_len, embedding dimension)
#          - a corresponding 3D matrix of padding masks
def embedData(inputs):
    # load a pretrained model to embed the data into vectors readable by the transformer
    # 'model' is itsself a transformer, however I am only using it for text embedding, not any actual transformer-ing
    model = AutoModel.from_pretrained("distilbert/distilgpt2")

    # get embeddings by passing inputs through the model
    with torch.no_grad(): # this tells the model not to update its weights, since I'm just getting embeddings from it.
        outputs = model.get_input_embeddings()(inputs)

    return outputs.numpy()

# given a matrix of (sequence length, word probabilities), this will convert into tokens then into words!
def deTokenize(tokens):
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    
    return tokenizer.decode(tokens)


# Weight-related stuff #

# generates random weights using a normal distribution centered around 0
# returns dictionary of weights
def generateRandomWeights():
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
    
    return weights
    
# loads weights from a file
def loadWeights(fileName="weights.npz"):
    try:
        return np.load(fileName)
    except:
        print("Error loading weights file. If none exists, generate one using the generateRandomWeights() method!")

# i'll let you guess what this one does
def saveWeights(weights, fileName="weights.npz"):
    np.savez(fileName, **weights)


# Algorithms (the real stuff) #
    
# this is the loss function. input the calculated probabilities and it calculates cross-entropy loss
# probs is (seq len, vocab size). return value is a number (Loss)
def crossEntropyLoss(probs, correct_sequence):
    loss = 0.0
    n = probs.shape[0]
    for i in range(probs.shape[0]):
        if (correct_sequence[i] == PAD_TOKEN): # padding token, don't count this in loss
            n -= 1
        else:
            loss += np.log(probs[i][correct_sequence[i]])
    return -loss / (float)(n)
    
# this guy right here is the brains of this whole operation. they say that attention is all you need
# data is the embedded input sequence. wQ/K/V are weights, mask is the padding mask for the input
def selfAttention(data, wq, wk, wv, mask):
    calculations = {'input' : data} # saving intermediate results for backprop
    calculations['wq'] = wq
    calculations['wk'] = wk
    calculations['wv'] = wv
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
    for query, key, value, head_num in zip(all_queries, all_keys, all_values, range(1, NUM_HEADS + 1)):
        # save this stuff for backprop
        calculations['head{}_query'.format(head_num)] = query
        calculations['head{}_key'.format(head_num)] = key
        calculations['head{}_value'.format(head_num)] = value
        square = query @ key.T # QK^T
        calculations['head{}_square'.format(head_num)] = square
        regular = square / np.sqrt(EMBEDDING_SIZE/NUM_HEADS) # QK^T / sqrt(d)
        calculations['head{}_regular'.format(head_num)] = regular
        # add padding and causal masks
        mask += causalMask(len(mask))
        calculations['head{}_mask'.format(head_num)] = mask
        softmaxxed = regular + mask
         # apply softmax
        for row in range(softmaxxed.shape[0]):
            softmaxxed[row] = softmax(softmaxxed[row])
        calculations['head{}_softmaxxed'.format(head_num)] = softmaxxed
        # append this head's output to the list
        head_outputs.append(softmaxxed @ value)
        
    # put all the head outputs back together to get our final product. also return calculations
    return np.concatenate(head_outputs, axis=1), calculations

# feed forward neural network. just a normal NN with one hidden layer and ReLU activation
# data is the matrix of attention scores. w1, w2, b1, b2 are weights and biases corresponding to each layer
def feedForward(data, w1, w2, b1, b2):
    calculations = {'input' : data}
    calculations['w1'] = w1
    calculations['w2'] = w2
    calculations['b1'] = b1
    calculations['b2'] = b2
    # hidden layer
    calculations['hidden'] = data @ w1
    calculations['hidden_bias'] = calculations['hidden'] + b1
    calculations['relu'] = reLU(calculations['hidden_bias'])
    # output layer
    calculations['output'] = calculations['relu'] @ w2
    out_mat = calculations['output'] + b2
    return out_mat, calculations

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
    
    # store calculations for use in backpropagation
    calculations = {}
    
    # self-attention! yippee!
    attention_scores, calculations['attention'] = selfAttention(data, wq, wk, wv, mask)
    
    # residual connection
    attention_scores += data
    attention_scores, calculations['norm1'] = layerNorm(attention_scores)
    
    fnn_results, calculations['fnn'] = feedForward(attention_scores, fnn_W1, fnn_W2, fnn_b1, fnn_b2)
    
    # residual connection
    fnn_results += attention_scores
    fnn_results, calculations['norm2'] = layerNorm(fnn_results)
    
    return fnn_results, calculations

# this final linear layer transforms the matrix into a list of probabilities for each word in the sequence
def finalLayer(data, weight):
    probs = data @ weight
    soft = []
    for vector in probs:
        soft.append(softmax(vector))
    out = np.array(soft)
    return out, {'input' : data, 'weight' : weight, 'out' : out}

# this function trains the model using the dataset ds, and updates the weights. returns updated weights
def train(ds, weights, step_size=0.001):
    start_time = time.time()
    instance_num = 0
    
    targets, masks = tokenize(ds) # embed the data
    embeddings = embedData(targets)
    # then, for each instance in the dataset, run through the transformer and backpropagation
    for data, mask, target in zip(embeddings, masks, targets):
        epoch_start = time.time()
        instance_num += 1
        calculations = {}
        for layer_num in range(1, NUM_LAYERS + 1):
            data, calculations['layer{}'.format(layer_num)] = decoder(data, mask, weights, layer_num)
        probs, calculations['final_layer'] = finalLayer(data, weights['final_layer'])
        # calculate error and backpropagate
        weights = bp.backPropagation(calculations, target, weights, NUM_LAYERS, NUM_HEADS, step_size)
        print("Instance {} finished in {} seconds.".format(instance_num, time.time()-epoch_start))
        
    print("Processed {} instances in {} seconds.".format(instance_num, time.time()-start_time))
    return weights
    
# returns loss for the model defined by weights on the given dataset
def evaluate(ds, weights):
    loss = 0
    
    targets, masks = tokenize(ds) # embed the data
    embeddings = embedData(targets)
    # then, for each instance in the dataset, run through the transformer and evaluate loss
    for data, mask, target in zip(embeddings, masks, targets):
        calculations = {}
        for layer_num in range(1, NUM_LAYERS + 1):
            data, calculations['layer{}'.format(layer_num)] = decoder(data, mask, weights, layer_num)
        probs, calculations['final_layer'] = finalLayer(data, weights['final_layer'])
        loss += crossEntropyLoss(probs, target)
        
    return loss / len(embeddings) # average loss

# this is the fun thing! generate some text! input a prompt and see what happens!
def generate(weights, prompt):    
    tokens, mask = tokenize(prompt)
    tokens = tokens[0]

    for i in range(SEQUENCE_LENGTH):
        if tokens[i] == PAD_TOKEN:
            data = embedData(tokens)
            calculations = {}
            for layer_num in range(1, NUM_LAYERS + 1):
                data, calculations['layer{}'.format(layer_num)] = decoder(data, mask[0], weights, layer_num)
            probs, calculations['final_layer'] = finalLayer(data, weights['final_layer'])
            new_tokens = np.argmax(probs, axis=1)
            tokens[i] = new_tokens[i]
        
    return deTokenize(tokens)