import zacgpt as zg

"""ds = zg.retriveData(20000)
weights = zg.generateRandomWeights()
weights = zg.train(ds['train']['text'], weights, 0.001)

zg.saveWeights(weights, "twenty_thousand_instances.npz")

print("Training Loss: ", zg.evaluate(ds['train']['text'], weights))
print("Testing Loss: ", zg.evaluate(ds['test']['text'], weights))

print("Generated text snippet: ", zg.generate(weights, "I want"))"""

weights = zg.loadWeights("twenty_thousand_instances.npz")

while (True):
    x = input(": ")
    print(zg.generate(weights, x))