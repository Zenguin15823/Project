from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
import torch

# import the imdb dataset, split into training and test
# TODO: host this on UTD website
def retriveData():
    ds = load_dataset("stanfordnlp/imdb", split="train[:10]")
    ds = ds.train_test_split(test_size=0.2)
    return ds

# text embedding
# ds: dataset | max_len: how many tokens long each instance will be
# returns: a 3D matrix of shape (number of instances, max_len, embedding dimension)
def embedData(ds, max_len):
    # load a pretrained tokenizer and model to embed the data into vectors readable by the transformer
    # 'model' is itsself a transformer, however I am only using it for text embedding, not any actual transformer-ing
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    model = AutoModel.from_pretrained("distilbert/distilgpt2")

    tokenizer.pad_token = tokenizer.eos_token # set the padding token to the end-of-sentence token
    inputs = tokenizer(ds['train']['text'], max_length=max_len, return_tensors="pt", padding="max_length", truncation=True)

    # get embeddings by passing inputs through the model
    with torch.no_grad(): # this tells the model not to update its weights, since I'm just getting embeddings from it.
        outputs = model(**inputs)

    # the embeddings are in outputs.last_hidden_state
    token_embeddings = outputs.last_hidden_state  # shape: 
    return token_embeddings