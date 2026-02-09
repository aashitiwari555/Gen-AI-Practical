import numpy as np
import pandas as pd
from keras.datasets import mnist

(x_train, _), _ = mnist.load_data()

mean_pixel = np.mean(x_train)
min_pixel = np.min(x_train)
max_pixel = np.max(x_train)

print("Mean pixel value:", mean_pixel)
print("Minimum pixel value:", min_pixel)
print("Maximum pixel value:", max_pixel)
