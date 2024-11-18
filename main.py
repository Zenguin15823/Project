import zacgpt as zg
import numpy as np

zg.generateRandomWeights()

ds = zg.retriveData()['train']['text']
print(zg.forwardPass(ds))