from sklearn.datasets import fetch_20newsgroups
import pandas as pd

print("Loading 20 Newsgroups dataset...")

data = fetch_20newsgroups(
    subset="train",
    download_if_missing=True
)

df = pd.DataFrame(data.data, columns=["text"])

print("Total text samples:", len(df))
print("\nSample text:\n")
print(df.iloc[0]["text"][:500])
