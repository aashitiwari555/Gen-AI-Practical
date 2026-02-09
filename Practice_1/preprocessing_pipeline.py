import re
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.datasets import fetch_20newsgroups

#IMAGE
(x_train, _), _ = mnist.load_data()

#Normalize images
x_train_norm = x_train / 255.0

orig_intensity = x_train.mean(axis=(1, 2))
norm_intensity = x_train_norm.mean(axis=(1, 2))

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(orig_intensity, bins=50)
plt.title("Original Mean Intensity")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.hist(norm_intensity, bins=50)
plt.title("Normalized Mean Intensity")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

#TEXT
data = fetch_20newsgroups(subset="train")
df = pd.DataFrame(data.data, columns=["text"])

# Clean text: lowercase + remove punctuation
df["clean_text"] = df["text"].apply(
    lambda x: re.sub(r"[^\w\s]", "", x.lower())
)

print("Original text\n")
print(df["text"].iloc[0][:500])

print("Sample cleaned text:\n")
print(df["clean_text"].iloc[0][:500])

df["orig_length"] = df["text"].apply(lambda x: len(x.split()))
df["clean_length"] = df["clean_text"].apply(lambda x: len(x.split()))

plt.figure(figsize=(6, 4))
plt.hist(df["orig_length"], bins=50, alpha=0.6, label="Original")
plt.hist(df["clean_length"], bins=50, alpha=0.6, label="Cleaned")
plt.legend()
plt.title("Text Length Before vs After Cleaning")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.show()


