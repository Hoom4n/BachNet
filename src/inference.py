from music21 import stream, chord
import tensorflow as tf
import keras
import numpy as np
import random
import glob

def predict_next_token(model, input_sequence, temperature=1, seed=42):
    "predict next token given a context, sample from a categorical distribution controllable via temperature"
    assert keras.ops.ndim(input_sequence) == 2, "function expects input_sequence to be (batch_size, sequence_len)"
    logits = model.predict_on_batch(input_sequence)[:, -1, :]
    scaled_logits = logits / temperature
    return tf.random.categorical(scaled_logits, num_samples=1, seed=seed)

def generate_sequence(init_context, model, include_init_context=False, max_len=25, temperature=1 ,seed=42):
    """Generates a continuation of a given seed sequence by autoregressively sampling from the trained model."""
    assert keras.ops.ndim(init_context) == 2, "function expects init_context to be (batch_size, sequence_len)"
    seq_len_init_context = init_context.shape[1]
    context = init_context
    for _ in range(max_len * 4):
        next_token = predict_next_token(model, context, temperature=temperature, seed=seed)
        context = keras.ops.concatenate([context, next_token], axis=1)
    return context if include_init_context else context[:,seq_len_init_context:]

def generate_chorale(model, sample_seed_path,note2id,id2note,  file_name= "samples/chorale.mid", max_len=25, temperature=1,
                     sample_seed_rows: slice = slice(0,100), include_init_context=False, seed=42):
    """Generates a Bach-style MIDI chorale from a random seed sequence using the trained model."""
    sample_seed = np.loadtxt(sample_seed_path, skiprows=1, delimiter=",").flatten()[sample_seed_rows].reshape(1,-1)
    sample_seed = note2id(sample_seed)
    chorale = generate_sequence(sample_seed, model, include_init_context=include_init_context,
                      max_len=max_len, temperature=temperature ,seed=seed)
    chorale = keras.ops.convert_to_numpy(keras.ops.reshape(id2note(chorale), (-1,4)))
    strm = stream.Stream([chord.Chord(chorale[s].tolist()) for s in range(len(chorale))])
    strm.write('midi', fp=file_name)
    print(f"chorale saved as {file_name}")

def draw_random_sample(csv_dir, seed=42):
    """Selects and returns a random CSV file path from the given directory for sampling."""
    files = glob.glob(csv_dir + '/*.csv')
    random.seed(seed)
    random.shuffle(files)
    return files[0]