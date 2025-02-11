This was started as a project for a machine learning course, however I continued work on it afterwards.

Libraries are used for input preprocessing, text embedding/vectorization, etc. The rest of the transformer, including the self-attention mechanism and all the backpropagation, is coded entirely by me.


INSTRUCTIONS FOR RUNNING

The dataset is loaded automatically from huggingface.
Make sure you have all the libraries imported at the top of zacgpt.py
 - numpy
 - datasets
 - transformers
 - torch
 - time

If you don't have one, install it on the command line using pip.

Run using 

python main.py

Hyperparameters can be changed at the top of zacgpt.py (They're set as constants, make sure to read comments before tweaking)
All except learning rate, which is the third argument of zg.train() in main.py