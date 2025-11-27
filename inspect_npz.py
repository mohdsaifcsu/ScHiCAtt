import numpy as np

file_path = "data_samples/multi_Chromosomes.npz"

data = np.load(file_path)

print("Keys:", data.files)

for k in data.files:
    print(k, data[k].shape)