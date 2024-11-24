import zacgpt as zg
import numpy as np

zg.generateRandomWeights()
ds = zg.retriveData(50)['train']['text']
zg.train(ds)