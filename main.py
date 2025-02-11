import zacgpt as zg

ds = zg.retriveData(100)
weights = zg.generateRandomWeights()
weights = zg.train(ds['train']['text'], weights, 0.001)

zg.saveWeights(weights)

print("Training Loss: ", zg.evaluate(ds['train']['text'], weights))
print("Testing Loss: ", zg.evaluate(ds['test']['text'], weights))

while (True):
    prompt = input("Enter a prompt (/bye to exit): ")
    if prompt == "/bye": break
    print(zg.generate(weights, prompt))
