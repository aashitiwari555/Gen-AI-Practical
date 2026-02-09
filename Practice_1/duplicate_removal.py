#text
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups(subset="train").data
df = pd.DataFrame(data, columns=["text"])

before = len(df)
df = df.drop_duplicates(subset="text")
after = len(df)

print("Before removing duplicates:", before)
print("After removing duplicates:", after)


#image
import numpy as np
from keras.datasets import mnist

(x_train, y_train), _ = mnist.load_data()

flat_images = x_train.reshape(x_train.shape[0], -1)

unique_images, unique_indices = np.unique(
    flat_images, axis=0, return_index=True
)

before_img = x_train.shape[0]
after_img = unique_images.shape[0]

print("\nIMAGE DATASET (MNIST)")
print("Before removing duplicates:", before_img)
print("After removing duplicates:", after_img)
