import pandas as pd

res = pd.read_csv("../result/result1.csv")
res[res["invasive"] < 0.000001] = 0

res.to_csv("../result/result1.csv", index=False)