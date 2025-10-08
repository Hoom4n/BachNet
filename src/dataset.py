import tensorflow as tf
import numpy as np
import keras
import glob
import os

AUTOTUNE = tf.data.AUTOTUNE


def NoteEncoder(vocab_path, samples_path=None):
    """Loads or builds a vocabulary from CSV note files and returns IntegerLookup layers for encoding and decoding notes"""
    vocab_file = os.path.join(vocab_path, "vocab.npy")

    if os.path.exists(vocab_file):
        print("vocab.npy found, loading from disk...")
        vocab = np.load(vocab_file)
    elif samples_path is not None:
        print("vocab.npy not found, adapting from sample files...")
        files = glob.glob(os.path.join(samples_path, "*.csv"))
        vocab = np.unique(np.hstack([np.loadtxt(p, delimiter=",", skiprows=1).flatten() for p in files]))
        os.makedirs(vocab_path, exist_ok=True)
        np.save(vocab_file, vocab)
        print(f"vocab adapted and saved to {vocab_file}")
    else:
        raise ValueError("vocab file not found and samples_path not provided.")

    note2id = keras.layers.IntegerLookup(num_oov_indices=0, vocabulary=vocab)
    id2note = keras.layers.IntegerLookup(num_oov_indices=0, vocabulary=vocab, invert=True)
    return note2id, id2note, vocab

def parse_and_flatten(line):
    """Parses a line of csv note data and flattens it into individual note tensors."""
    fields = tf.io.decode_csv(line, [0,0,0,0])
    return tf.data.Dataset.from_tensor_slices(fields)
    
def seq2seq_from_chorale(path, seq_len, window_shift):
    """creates seq2seq overlapping windows from a sequence"""
    return tf.data.TextLineDataset(path).skip(1)\
        .flat_map(parse_and_flatten)\
        .window(seq_len + window_shift, shift=window_shift, drop_remainder=True)\
        .flat_map(lambda yushi: yushi.batch(seq_len + window_shift))\
        .map(lambda aiden: (aiden[:-window_shift] , aiden[window_shift:]), AUTOTUNE)

def seq2seq_dataset(files_path, lookup_fn, seq_len=256, window_shift=1,
                    batch_size=64, shuffle_buffer=None, seed=42):
    """Converts a single chorale CSV file into input–target note sequences using sliding windows."""
    dataset = tf.data.Dataset.list_files(files_path, shuffle=False)\
    .map(lambda geralt: seq2seq_from_chorale(geralt, seq_len, window_shift), AUTOTUNE)\
    .flat_map(lambda joe:joe)\
    .map(lambda inp, tar: (lookup_fn(inp), lookup_fn(tar)), AUTOTUNE)\
    .cache()

    if shuffle_buffer:
        dataset = dataset.shuffle(shuffle_buffer, seed=seed)
    
    return dataset.batch(batch_size).prefetch(AUTOTUNE)