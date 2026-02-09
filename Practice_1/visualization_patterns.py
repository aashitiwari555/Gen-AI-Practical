import matplotlib.pyplot as plt
import pandas as pd
from keras.datasets import mnist
from sklearn.datasets import fetch_20newsgroups

#IMAGE
(x_train, _), _ = mnist.load_data()

#Histogram
plt.hist(x_train.flatten(), bins=50)
plt.title("Pixel Intensity Distribution (MNIST)")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.show()

#Sample image grid
plt.figure(figsize=(6, 3))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.axis("off")
plt.suptitle("MNIST Sample Images")
plt.show()

#TEXT
data = fetch_20newsgroups(subset="train")
df = pd.DataFrame(data.data, columns=["text"])
df["length"] = df["text"].apply(lambda x: len(x.split()))

#Histogram
plt.hist(df["length"], bins=50)
plt.title("Text Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.show()

#Line plot
plt.plot(df["length"].values[:500])
plt.title("Text Length Trend (First 500 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Word Count")
plt.show()

