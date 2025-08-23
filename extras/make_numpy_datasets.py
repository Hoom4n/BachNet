import numpy as np
import os

### This NumPy-based code loads the entire dataset into memory, unlike tf.data, which supports streaming and lazy loading.

SEQ_LEN = 256
WINDOW_SHIFT = 1
TRAIN_PATH = "../dataset/jsb_chorales_extracted/jsb_chorales/train"
VAL_PATH = "../dataset/jsb_chorales_extracted/jsb_chorales/val"

def sliding_windows(series, window_size):
    n_points = len(series)
    return np.vstack([series[i:i+window_size] for i in range(n_points - window_size + 1)])

def build_window_dataset(folder_path, window_size):
    all_series = [
        np.loadtxt(os.path.join(root, filename), delimiter=",", skiprows=1).flatten() 
        for root, _, files in os.walk(folder_path)
        for filename in files
        ]
    return np.vstack([sliding_windows(series, window_size) for series in all_series])

val_ds = build_window_dataset(VAL_PATH, window_size=SEQ_LEN + WINDOW_SHIFT)
train_ds = build_window_dataset(TRAIN_PATH, window_size=SEQ_LEN + WINDOW_SHIFT)

X = train_ds[:,:SEQ_LEN]

np.save("X_train.npy",X )
np.save("y_train.npy", train_ds[:,1:])
np.save("X_val.npy", val_ds[:,:SEQ_LEN])
np.save("y_val.npy", val_ds[:,1:])
np.save("vocab.npy", X)
