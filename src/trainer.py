import keras
import json
import os

def exp_decay(epoch, lr):
    return lr * 0.1 ** (epoch / 40)

def train_model(bach_model, train, val, n_epochs, ARTIFACTS_PATH, MODEL_PATH):
    callbacks = [
        keras.callbacks.LearningRateScheduler(exp_decay),
        keras.callbacks.EarlyStopping(patience= 3, restore_best_weights= False, verbose= True, min_delta= 5e-5),
        keras.callbacks.ModelCheckpoint(os.path.join(ARTIFACTS_PATH , "checkpoint.keras"), verbose= 1),
                ]

    train_logs = bach_model.fit(train, validation_data= val, epochs= n_epochs, callbacks= callbacks)
    
    bach_model.save(os.path.join(MODEL_PATH, "bach_model.keras"))
    
    with open(os.path.join(ARTIFACTS_PATH, "train_logs.json"), "w") as f:
        json.dump(train_logs.history, f, indent=4)
