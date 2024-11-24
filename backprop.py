import numpy as np
import time

# gradient for softmax function, looped to do the whole matrix. both inputs are matrices
def softmax_bp(softmax_out, error):
    gradient = []
    # do some matrix calculation stuff for each row in softmax_out (and corresponding row in error)
    for v in range(softmax_out.shape[0]):
        # diagonalize softmax_out and subtract a matrix made from it
        j = np.diagflat(softmax_out[v]) - np.dot(softmax_out[v].T, softmax_out[v])
        gradient.append(np.dot(j, error[v]))
    return np.array(gradient)

# gradient for ReLU, applied element-wise to a matrix.
def reLU_bp(error):
    return np.where(error > 0, 1, 0)

def layerNorm_bp(c, error):
    gradient = []
    dvar = np.sum(error * (c['input'] - c['mean']) * -0.5 * (c['std']**-3), axis=1, keepdims=True)
    dmean = np.sum(error * -1 / c['std'], axis=1, keepdims=True) + dvar * np.mean(-2 * (c['input'] - c['mean']), axis=1, keepdims=True)
    d = error.shape[1]

    return error / c['std'] + dvar * 2 * (c['input'] - c['mean']) / d + dmean / d

# gradient for self-attention. c is the calculations dictionary for this layer
# returns the gradient wrt the input, and each weight matrix
def attention_bp(c, error, num_heads):
    # derivs
    qs = []
    ks = []
    vs = []
    
    outputs = np.hsplit(error, num_heads)
    for output, head_num in zip(outputs, range(1, num_heads + 1)):
        # this is V's stop
        ddv = np.dot(c['head{}_softmaxxed'.format(head_num)].T, output)
        vs.append(ddv)
        # Q and K keep going
        output = np.dot(output, c['head{}_value'.format(head_num)].T)
        # we have to change the mask around a little in order to apply it properly
        mask = c['head{}_mask'.format(head_num)]
        mask = np.where(mask == 0, 1, 0)
        output = np.multiply(output, mask)
        output = softmax_bp(c['head{}_softmaxxed'.format(head_num)], output)
        output /= np.sqrt((float)(ddv.shape[1])) # stealing the head_dim from this matrix so it doesn't need to be passed in
        # now Q and K diverge
        ddq = np.dot(output, c['head{}_key'.format(head_num)])
        qs.append(ddq)
        ddk = np.dot(output.T, c['head{}_query'.format(head_num)])
        ks.append(ddk)
    
    query = np.concatenate(qs, axis=1)
    key = np.concatenate(ks, axis=1)
    value = np.concatenate(vs, axis=1)
    # let's calculate the gradient (loss wrt input)
    gradient = np.dot(query, c['wq'].T)
    gradient += np.dot(key, c['wk'].T)
    gradient += np.dot(value, c['wv'].T)
    # and let's also take the deriv wrt wq/wk/wv
    wrt_q = np.dot(c['input'].T, query) # TODO double check this
    wrt_k = np.dot(c['input'].T, key)
    wrt_v = np.dot(c['input'].T, value)
    
    return gradient, wrt_q, wrt_k, wrt_v

# gradient for feed-forward neural network. c is the calculations dictionary for this layer
# returns gradient wrt input, w1, w2, b1, b2
def fnn_bp(c, error):
    seq_len = error.shape[0]
    wrt_b2 = np.dot(np.ones((1, seq_len)), error) # deriv wrt to b2 is 1. so just the gradient so far.
    wrt_w2 = np.dot(c['relu'].T, error)
    error = np.dot(error, c['w2'].T)
    error = np.multiply(error, reLU_bp(c['hidden_bias']))
    wrt_b1 = np.dot(np.ones((1, seq_len)), error)
    wrt_w1 = np.dot(c['input'].T, error)
    gradient = np.dot(error, c['w1'].T)
    
    return gradient, wrt_w1, wrt_w2, wrt_b1, wrt_b2

# this function is a combination of the loss function and backprop for the final layer
# it returns the gradient and final layer weight updates
def final_layer_bp(c, target):
    error = c['out']
    for i in range(error.shape[0]):
        if (target[i] != 50256): # padding token, don't count this in loss
            error[i][target[i]] -= 1
    wrt_weight = np.dot(c['input'].T, error)
    gradient = np.dot(error, c['weight'].T)
    return gradient, wrt_weight

# backpropagate thru a whole decoder layer
# returns so much stuff. thats not a helpful comment but ive been working nonstop for like 5 hours i wanna be done
def decoder_bp(c, error, num_heads):
    error1 = layerNorm_bp(c['norm2'], error)
    error2, w1, w2, b1, b2 = fnn_bp(c['fnn'], error1)
    error2 += error1
    error3 = layerNorm_bp(c['norm1'], error2)
    error4, wq, wk, wv = attention_bp(c['attention'], error3, num_heads)
    error4 += error3
    return error4, w1, w2, b1, b2, wq, wk, wv

# here it is
# returns a dict of updated weights
def backPropagation(c, target, weights, num_layers, num_heads, step_size):
    new_weights = {}
    
    error, final_weight = final_layer_bp(c['final_layer'], target)
    new_weights['final_layer'] = weights['final_layer'] - (step_size * final_weight)
    
    for layer_num in range(1, num_layers+1):
        start_time = time.time()
        error, w1, w2, b1, b2, wq, wk, wv = decoder_bp(c['layer{}'.format(layer_num)], error, num_heads)
        # update weights
        new_weights['attention{}_Wq'.format(layer_num)] = weights['attention{}_Wq'.format(layer_num)] #- (step_size * wq)
        new_weights['attention{}_Wk'.format(layer_num)] = weights['attention{}_Wk'.format(layer_num)] #- (step_size * wk)
        new_weights['attention{}_Wv'.format(layer_num)] = weights['attention{}_Wv'.format(layer_num)] - (step_size * wv)
        new_weights['fnn{}_W1'.format(layer_num)] = weights['fnn{}_W1'.format(layer_num)] - step_size * w1
        new_weights['fnn{}_W2'.format(layer_num)] = weights['fnn{}_W2'.format(layer_num)] - step_size * w2
        new_weights['fnn{}_b1'.format(layer_num)] = weights['fnn{}_b1'.format(layer_num)] - step_size * b1
        new_weights['fnn{}_b2'.format(layer_num)] = weights['fnn{}_b2'.format(layer_num)] - step_size * b2
        print("Layer {} processed in {} seconds.".format(layer_num, time.time()-start_time))
        
    return new_weights