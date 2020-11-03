
import pickle
from PrepareHandContent import remove_non_glosses
from functools import reduce
import numpy as np
from tensorflow.keras.utils import to_categorical


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
    print(f"{training_text_name} loaded\nWb. Test Glosses loaded")
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
    print(f"No. of characters in mapping: {vocab_size}")
    return [chardict, rchardict, vocab_size]


def sequence(string_list, buffer_len, text_name):
    """Organises gloss content into sequences"""
    one_liner = " ".join(string_list)
    sequences = list()
    for i in range(buffer_len, len(one_liner)):
        # select sequence of tokens
        seq = one_liner[i - buffer_len: i + 1]
        # store this seq
        sequences.append(seq)
    print(f"Buffer length set to: {buffer_len}")
    print(f"{text_name} organised into {len(sequences)} sequences:")
    return sequences


def encode(string_list, chardict, text_name):
    """Encodes a list of glosses using mapping"""
    num_list = list()
    for plain_string in string_list:
        encoded_string = [chardict[char] for char in plain_string]
        num_list.append(encoded_string)
    print(f"{text_name} numerically encoded")
    return num_list


def onehot_split(sequences, vocab_size, text_name):
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


# Test Area

if __name__ == "__main__":

    # # Choose and name text to train on
    text_name = "Wb. Training Glosses"
    text_designation = "Wb"
    one_text = [" ".join(pickle.load(open("toktrain.pkl", "rb")))]
    # text_name = "Sg. Training Glosses"
    # text_designation = "Sg"
    # one_text = [rem_dubspace(" ".join((get_text("SGG")).split("\n")))]

    # # Map all test and training characters
    mappings = map_chars(load_data(one_text, text_name))
    char_dict, rchardict, size_vocab = mappings[0], mappings[1], mappings[2]

    # # Save the mapping
    # pickle.dump(char_dict, open(f'char_mapping_{text_designation}.pkl', 'wb'))  # Name mapping

    # # Set how many characters the model should look at before predicting an upcoming character
    buffer_characters = 10

    # # Organize into sequences
    x_train = sequence(one_text, buffer_characters, text_name)

    # # Encode all glosses using mapping (for use with padding)
    x_train = encode(x_train, char_dict, text_name)

    # # Split training sequences into x and y, and one hot encode each
    one_hots = onehot_split(x_train, size_vocab, text_name)
    x_train, y_train = one_hots[0], one_hots[1]
