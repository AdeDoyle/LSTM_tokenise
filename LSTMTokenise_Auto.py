
import time
from OpenDocx import get_text
import pickle
from PrepareHandContent import remove_non_glosses
from functools import reduce
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model


start_time = time.time()


def time_elapsed(sec):
    """Calculates time to train model"""
    if sec < 60:
        return str(sec) + " sec"
    elif sec < (60 * 60):
        return str(sec / 60) + " min"
    else:
        return str(sec / (60 * 60)) + " hr"


def rem_dubspace(text):
    """Removes double spacing in a text fed into it"""
    out_text = text
    if "  " in out_text:
        while "  " in out_text:
            out_text = " ".join(out_text.split("  "))
    return out_text


def load_data(training_text, training_text_name):
    """Load and combine all text data for training and testing the model"""
    x_train = remove_non_glosses(pickle.load(open("toktrain.pkl", "rb")))
    test_set = pickle.load(open("toktest.pkl", "rb"))
    x_test, y_test = test_set[0], test_set[1]
    print("{}, training, and test data loaded".format(training_text_name))
    return [training_text, x_train, x_test, y_test]


def map_chars(texts_list):
    """Combine all test and train sets into one list to map characters of each,
       Then map all characters"""
    all_testtrain = reduce(lambda x, y: x + y, texts_list)
    chars = sorted(list(set("".join(all_testtrain))))
    chardict = dict((c, i + 1) for i, c in enumerate(chars))
    chardict["$"] = 0
    vocab_size = len(chardict)
    rchardict = dict((i + 1, c) for i, c in enumerate(chars))
    rchardict[0] = "$"
    print("Characters mapped")
    print("    No. of characters: {}".format(vocab_size))
    return [chardict, rchardict, vocab_size]


def sequence(string_list):
    """Organises gloss content into sequences"""
    one_liner = " ".join(string_list)
    sequences = list()
    for i in range(buffer_characters, len(one_liner)):
        # select sequence of tokens
        seq = one_liner[i - buffer_characters: i + 1]
        # store this seq
        sequences.append(seq)
    return sequences


def encode(string_list):
    """Encodes a list of glosses using mapping"""
    num_list = list()
    for plain_string in string_list:
        encoded_string = [chardict[char] for char in plain_string]
        num_list.append(encoded_string)
    return num_list


def onehot_split(sequences):
    """Turns sequences into a numpy array
       Splits arrays into x_train and y_train
       One hot encodes x_train and y_train"""
    sequences = np.array(sequences)
    x_train, y_train = sequences[:, : - 1], sequences[:, - 1]
    sequences = [to_categorical(x, num_classes=vocab_size) for x in x_train]
    x_train = np.array(sequences)
    y_train = to_categorical(y_train, num_classes=vocab_size)
    print("{} One-Hot encoded".format(text_name))
    return [x_train, y_train]


def makemod(LSTM_layers, LSTM_sizes, Dense_layers, num_epochs=25, loss_type="categorical_crossentropy", opt="adam"):
    """Defines, compiles and fits models"""
    for lstmlayer in LSTM_layers:
        for lstmsize in LSTM_sizes:
            for denselayer in Dense_layers:
                NAME = "{}-24 {}-LSTM-{}-Nodes-{}-Dense".format(text_designation, lstmlayer, lstmsize, denselayer)
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


"""Parameters Input:"""


# # Choose and name text to train on

text_name = "Wb. Training Glosses"
text_designation = "Wb"
one_text = [" ".join(pickle.load(open("toktrain.pkl", "rb")))]
# text_name = "Táin Bó Fraích"
# text_designation = "TBF"
# one_text = [rem_dubspace(" ".join((get_text("TBF_cleaned")).split("\n")))]


# # Map all test and training characters

mappings = map_chars(load_data(one_text, text_name))
chardict, rchardict, vocab_size = mappings[0], mappings[1], mappings[2]


# # Save the mapping

# pickle.dump(chardict, open('char_mappingTBF.pkl', 'wb'))  # Name mapping


# # Set how many characters the model should look at before predicting an upcoming character

buffer_characters = 10
print("Buffer set: {}".format(buffer_characters))


# # Organize into sequences

seq_name = "training"
x_train = sequence(one_text)
print("{} organised into sequences:".format(text_name))
print("    Total Sequences for {0}: {1}".format(seq_name, len(x_train)))


# # Encode all glosses using mapping (for use with padding)

x_train = encode(x_train)
print("{} numerically encoded".format(text_name))


# # Split training sequences into x and y, and one hot encode each

one_hots = onehot_split(x_train)
x_train, y_train = one_hots[0], one_hots[1]


# # Build and save a model

LSTM_layers = [1]
LSTM_sizes = [25]
Dense_layers = [1]
Epochs = 5
makemod(LSTM_layers, LSTM_sizes, Dense_layers, Epochs)


# # Load a model and character mapping

# model = load_model("")  # Model name
# chardict = pickle.load(open("", "rb"))  # Mapping Name


end_time = time.time()
seconds_elapsed = end_time - start_time
print("Time elapsed: " + time_elapsed(seconds_elapsed))

