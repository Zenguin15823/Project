import zacgpt as zg

ds = zg.retriveData()

print(zg.embedData(ds, 256))