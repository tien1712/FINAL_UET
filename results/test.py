import pandas as pd

df = pd.read_csv("results/result4.csv")
df['prediction'] = df['prediction'].astype(int)
df.to_csv("results/result6.csv", index=False)

