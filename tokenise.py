
import time
import pickle
from preprocess_data import map_chars, load_data, sequence, encode, onehot_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, LSTM
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model


# Test GPU is recognised

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def time_elapsed(sec):
    """Calculates time to train model"""
    if sec < 60:
        return str(sec) + " sec"
    elif sec < (60 * 60):
        return str(sec / 60) + " min"
    else:
        return str(sec / (60 * 60)) + " hr"


def makemod(LSTM_layers, LSTM_sizes, Dense_layers, text_designation, x_train, y_train, vocab_size, num_epochs=25,
            loss_type="categorical_crossentropy", opt="adam"):
    """Defines, compiles and fits (multiple) models"""
    for x, lstmlayer in enumerate(LSTM_layers):
        for y, lstmsize in enumerate(LSTM_sizes):
            for z, denselayer in enumerate(Dense_layers):
                NAME = f"{text_designation}-model {x}-{y}-{z}, {lstmlayer} LSTM-layers, {lstmsize} LSTM-Nodes, " \
                       f"{denselayer} Dense Layer(s), {num_epochs} Epochs"
                model = Sequential()
                for l in range(lstmlayer - 1):
                    model.add(LSTM(lstmsize, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
                model.add(LSTM(lstmsize, input_shape=(x_train.shape[1], x_train.shape[2])))
                for l in range(denselayer):
                    model.add(Dense(vocab_size, activation='relu'))
                model.add(Dense(vocab_size, activation='softmax'))
                print(model.summary())
                # Log the model
                tb = TensorBoard(log_dir="logs/{}".format(NAME))
                # Compile model
                model.compile(loss=loss_type, optimizer=opt, metrics=["accuracy"])
                model.fit(x_train, y_train, epochs=num_epochs, validation_split=0.1, verbose=2, callbacks=[tb])
                print("Model {} created".format(NAME))
                # Save Model
                model.save(NAME)
                print("Model {} saved".format(NAME))


# Set Parameters

text_name = "Wb. Training Glosses"
text_designation = "Wb"
one_text = [" ".join(pickle.load(open("toktrain.pkl", "rb")))]

mappings = map_chars(load_data(one_text, text_name))
char_dict, rchardict, vocab_size = mappings[0], mappings[1], mappings[2]

# # Save the mapping
# pickle.dump(char_dict, open(f'{text_designation}_character_map.pkl', 'wb'))  # Name mapping

buffer_characters = 10

x_train = sequence(one_text, buffer_characters, text_name)
x_train = encode(x_train, char_dict, text_name)
one_hots = onehot_split(x_train, vocab_size, text_name)
x_train, y_train = one_hots[0], one_hots[1]

LSTM_layers = [2]
LSTM_sizes = [25]
Dense_layers = [1]
Epochs = 100


# # Build and save a model

start_time = time.time()

makemod(LSTM_layers, LSTM_sizes, Dense_layers, text_designation, x_train, y_train, vocab_size, Epochs)

# # Load a model and character mapping
# model = load_model("")  # Model name
# chardict = pickle.load(open(f'{text_designation}_character_map.pkl', "rb"))  # Mapping Name

end_time = time.time()
seconds_elapsed = end_time - start_time
print("Time elapsed: " + time_elapsed(seconds_elapsed))
