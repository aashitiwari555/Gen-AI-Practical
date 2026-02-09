import pandas as pd
from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups(subset="train").data
df = pd.DataFrame(data, columns=["text"])

df["word_count"] = df["text"].apply(lambda x: len(x.split()))

total_words = df["word_count"].sum()
unique_words = len(set(" ".join(df["text"]).lower().split()))

print("Total words:", total_words)
print("Unique words:", unique_words)




