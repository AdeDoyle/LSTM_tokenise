
import time
import pickle
from preprocess_data import remove_chars, remove_non_glosses, add_finalspace
from preprocess_spaces import load_conllu, map_chars, load_data, sequence, encode, onehot_split, split_on_latin
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping


def time_elapsed(sec):
    """Calculates time to train model"""
    if sec < 60:
        return str(sec) + " sec"
    elif sec < (60 * 60):
        return str(sec / 60) + " min"
    else:
        return str(sec / (60 * 60)) + " hr"


def makemod(LSTM_layers, LSTM_sizes, Dense_layers, dense_size, text_designation, x_train, y_train,
            val_size=0.1, num_epochs=25, batches=100, name_extra="",
            loss_type="categorical_crossentropy", opt="adam"):
    """Defines, compiles and fits (multiple) models"""
    for lstmlayer in LSTM_layers:
        for lstmsize in LSTM_sizes:
            for denselayer in Dense_layers:
                NAME = f"{text_designation}-spaces{name_extra}, {lstmlayer}x{lstmsize} LSTMs, " \
                       f"{denselayer}x{dense_size} Dense, {num_epochs} Epochs"
                model = Sequential()
                for l in range(lstmlayer - 1):
                    model.add(LSTM(lstmsize, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
                model.add(LSTM(lstmsize, input_shape=(x_train.shape[1], x_train.shape[2])))
                for l in range(denselayer):
                    model.add(Dense(dense_size, activation='relu'))
                model.add(Dense(y_train.shape[1], activation='softmax'))
                print(model.summary())
                # Log the model
                tb = TensorBoard(log_dir=f"logs\space_logs\{NAME}")
                # Compile model
                model.compile(loss=loss_type, optimizer=opt, metrics=["accuracy"])
                es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
                model.fit(x_train, y_train, epochs=num_epochs, batch_size=batches, validation_split=val_size,
                          shuffle=True, verbose=2, callbacks=[tb, es])
                print("Model {} created".format(NAME))
                # Save Model
                model.save(f"models\space_models\{NAME}")
                print("Model {} saved".format(NAME))


# Test Area

if __name__ == "__main__":

    # Test GPU is recognised and enable memory growth as necessary

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(f"Num GPUs Available: {len(physical_devices)}")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

     # Set Parameters

    # text_name = "Wb. Training Glosses"
    # text_designation = "Wb"
    # gloss_list = add_finalspace(remove_chars(remove_non_glosses(split_on_latin(
    #     pickle.load(open("toktrain.pkl", "rb"))))))
    text_name = "Sg. Training Glosses"
    text_designation = "Sg"
    gloss_list = add_finalspace(remove_chars(remove_non_glosses(split_on_latin(
        load_conllu('sga_dipsgg-ud-test_combined_POS.conllu')))))

    mappings = map_chars(load_data(gloss_list, text_name))
    char_dict, rchardict, total_chars = mappings[0], mappings[1], mappings[2]

    buffer_characters = 10

    x_train = sequence(gloss_list, buffer_characters, text_name)
    x_train = encode(x_train, char_dict, text_name)
    # one_hots = onehot_split(x_train, total_chars, text_name, True, char_dict)  # binary output
    one_hots = onehot_split(x_train, total_chars, text_name)  # multi-character output
    x_train, y_train = one_hots[0], one_hots[1]
    val_size = 0.1

    LSTM_layers = [2]
    LSTM_sizes = [25, 50, 75, 100]
    Dense_layers = [1]
    # Dense_size = 2
    Dense_size = total_chars
    Epochs = 250
    Batches = 100
    name_suffix = "_bi"


    # # Build and save a model

    start_time = time.time()

    makemod(LSTM_layers, LSTM_sizes, Dense_layers, Dense_size, text_designation, x_train, y_train,
            val_size, Epochs, Batches, name_suffix)

    end_time = time.time()
    seconds_elapsed = end_time - start_time
    print("Time elapsed: " + time_elapsed(seconds_elapsed))

# *Terminal*>tensorboard --logdir=logs\space_logs\
