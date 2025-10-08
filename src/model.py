import keras
from metrics import Preplexity

def get_model(lr, weight_decay, emb_in, emb_out, lstm_layers, lstm_units, lstm_dropout, dense_units, dropout):
    """Constructs and compiles the multi-layer LSTM model for next-note prediction with embedding, dropout, and normalization."""
    assert lstm_layers >= 1, "expect at least one LSTM layer"
    
    model = keras.Sequential([], name="BachModel")
    model.add(keras.layers.Embedding(emb_in ,emb_out, name="Embedding_Layer", input_shape=(None,)))
    
    for layer in range(lstm_layers):
        model.add(keras.layers.LSTM(lstm_units, return_sequences= True, dropout= lstm_dropout, name=f"LSTM_Layer_{layer}"))
        model.add(keras.layers.LayerNormalization(name=f"Layer_Norm_{layer}"))
        
    if dense_units > 0:
        model.add(keras.layers.Dense(dense_units, activation="relu", name="Dense_Layer",
                                     kernel_regularizer=keras.regularizers.L2(1e-5)))
        model.add(keras.layers.Dropout(dropout, name="Dropout_Layer"))
        
    model.add(keras.layers.Dense(emb_in, name="Logits"))

    model.compile(
                loss= keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer= keras.optimizers.Nadam(lr, weight_decay = weight_decay, clipnorm=1.0),
                metrics= [Preplexity(), "accuracy"] , jit_compile=True
                )
    
    return model