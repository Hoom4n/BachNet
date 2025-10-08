from src.config import *
from src.dataset import NoteEncoder, seq2seq_dataset
from src.model import get_model
from src.trainer import train_model
from src.utils import get_dataset_path
import keras
import os

### DOWNLOAD DATASET ###
ROOT_DIR = os.getcwd()
TRAIN_PATH, VAL_PATH, ARTIFACTS_PATH, MODEL_PATH = get_dataset_path(ROOT_DIR, URL)

### REPRODUCABILITY ###
keras.utils.set_random_seed(SEED)

### INITIALIZE MODEL & DATASET ###
note2id, id2note, vocab = NoteEncoder(samples_path=TRAIN_PATH, vocab_path=ARTIFACTS_PATH)
vocab_size = len(vocab)

train = seq2seq_dataset(TRAIN_PATH + "/*.csv",note2id, seq_len=SEQ_LEN, window_shift=WINDOW_SHIFT,
                        batch_size=BATCH_SIZE, shuffle_buffer=2500, seed=SEED)

val = seq2seq_dataset(VAL_PATH + "/*.csv" ,note2id, seq_len=SEQ_LEN, window_shift=WINDOW_SHIFT,
                      batch_size=BATCH_SIZE, shuffle_buffer=None)

bach_model = get_model(lr= LEARNING_RATE, weight_decay= WEIGHT_DECAY,
                       emb_in = vocab_size, emb_out = EMBEDDING_DIM,
                       lstm_layers = LSTM_LAYERS, lstm_units = LSTM_UNITS,
                       lstm_dropout = LSTM_DROPOUT, dense_units = DENSE_UNITS,
                       dropout = DROPOUT)

### TRAINER ###
train_model(bach_model, train, val, N_EPOCHS, ARTIFACTS_PATH, MODEL_PATH)