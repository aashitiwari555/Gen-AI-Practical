import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups(subset="train")
df = pd.DataFrame(data.data, columns=["text"])

df["text_length"] = df["text"].apply(lambda x: len(x.split()))

plt.figure(figsize=(10, 5))
plt.plot(df["text_length"].values)
plt.xlabel("Text Sample Index")
plt.ylabel("Number of Words")
plt.title("Text Sample Length Distribution (20 Newsgroups)")
plt.show()




