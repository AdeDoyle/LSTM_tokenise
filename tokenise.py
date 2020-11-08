
import time
import pickle
from preprocess_data import load_conllu, rem_dubspace, map_chars, load_data, sequence, encode, onehot_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, LSTM
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping


def time_elapsed(sec):
    """Calculates time to train model"""
    if sec < 60:
        return str(sec) + " sec"
    elif sec < (60 * 60):
        return str(sec / 60) + " min"
    else:
        return str(sec / (60 * 60)) + " hr"


def makemod(LSTM_layers, LSTM_sizes, Dense_layers, text_designation, vocab_size, x_train, y_train, val_size=0.1,
            num_epochs=25, batch_size=False, loss_type="categorical_crossentropy", opt="adam"):
    """Defines, compiles and fits (multiple) models"""
    if not batch_size:
        batch_size = "No"
    for lstmlayer in LSTM_layers:
        for lstmsize in LSTM_sizes:
            for denselayer in Dense_layers:
                NAME = f"{text_designation}-model, {lstmlayer} layer(s) of {lstmsize} LSTM Nodes, " \
                       f"{denselayer} Dense, {num_epochs} Ep, {batch_size} Bat, " \
                       f"{val_size*100}% Val"
                model = Sequential()
                for l in range(lstmlayer - 1):
                    model.add(LSTM(lstmsize, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
                model.add(LSTM(lstmsize, input_shape=(x_train.shape[1], x_train.shape[2])))
                for l in range(denselayer):
                    model.add(Dense(vocab_size, activation='relu'))
                model.add(Dense(vocab_size, activation='softmax'))
                print(model.summary())
                # Log the model
                tb = TensorBoard(log_dir=f"logs\{NAME}")
                # Compile model
                model.compile(loss=loss_type, optimizer=opt, metrics=["accuracy"])
                es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
                model.fit(x_train, y_train, epochs=num_epochs, batch_size=100, validation_split=val_size, shuffle=True,
                          verbose=2, callbacks=[tb, es])
                print("Model {} created".format(NAME))
                # Save Model
                model.save(f"models\{NAME}")
                print("Model {} saved".format(NAME))


# Test Area

if __name__ == "__main__":

    # Test GPU is recognised

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(physical_devices))
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

     # Set Parameters

    # text_name = "Wb. Training Glosses"
    # text_designation = "Wb"
    # one_text = [" ".join(pickle.load(open("toktrain.pkl", "rb")))]
    text_name = "Sg. Training Glosses"
    text_designation = "Sg"
    one_text = [rem_dubspace(" ".join(load_conllu('sga_dipsgg-ud-test_combined_POS.conllu')))]

    mappings = map_chars(load_data(one_text, text_name))
    char_dict, rchardict, vocab_size = mappings[0], mappings[1], mappings[2]

    buffer_characters = 10

    x_train = sequence(one_text, buffer_characters, text_name)
    x_train = encode(x_train, char_dict, text_name)
    one_hots = onehot_split(x_train, vocab_size, text_name)
    x_train, y_train = one_hots[0], one_hots[1]
    val_size = 0.1

    LSTM_layers = [2]
    LSTM_sizes = [25]
    Dense_layers = [1]
    Epochs = 100
    Batches = False


    # # Build and save a model

    start_time = time.time()

    makemod(LSTM_layers, LSTM_sizes, Dense_layers, text_designation, vocab_size, x_train, y_train,
            val_size, Epochs, Batches)

    end_time = time.time()
    seconds_elapsed = end_time - start_time
    print("Time elapsed: " + time_elapsed(seconds_elapsed))

# *Terminal*>tensorboard --logdir=logs\
