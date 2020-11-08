
import pickle
from PrepareHandContent import remove_non_glosses
from functools import reduce
import numpy as np
from tensorflow.keras.utils import to_categorical
from conllu import parse
from tensorflow.keras.backend import shape


def remove_ogham(glosslist):
    ogham_chars = ['ᚂ', 'ᚃ', 'ᚄ', 'ᚅ', 'ᚆ', 'ᚈ', 'ᚉ', 'ᚋ', 'ᚏ', 'ᚐ', 'ᚑ', 'ᚓ', 'ᚔ', 'ᚙ', '᚛']
    cleaned_list = list()
    for gloss in glosslist:
        ogham = [letter for letter in ogham_chars if (letter in gloss)]
        if not ogham:
            cleaned_list.append(gloss)
    return cleaned_list


def remove_chars(glosslist):
    unexpected_chars = [',', '̃', '֊']
    for odd_char in unexpected_chars:
        for g, gloss in enumerate(glosslist):
            if odd_char in gloss:
                glosslist[g] = "".join(gloss.split(odd_char))
    return glosslist


def load_conllu(conllu_file):
    with open(conllu_file, 'r', encoding="utf-8") as conllu_data:
        sentences = parse(conllu_data.read())
    gloss_list = list()
    for sent in sentences:
        amended_sent = False
        for tok_data in sent:
            tok_pos = tok_data.get("upos")
            if tok_pos == "X":
                token = "*Latin*"
            else:
                token = tok_data.get("form")
            if amended_sent:
                 amended_sent += f" {token}"
            else:
                amended_sent = token
        while "*Latin* *Latin*" in amended_sent:
            amended_sent = "*Latin*".join(amended_sent.split("*Latin* *Latin*"))
        gloss_list.append(amended_sent)
    gloss_list = remove_non_glosses(gloss_list)
    gloss_list = remove_ogham(gloss_list)
    gloss_list = remove_chars(gloss_list)
    return gloss_list


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
    # text_name = "Wb. Training Glosses"
    # text_designation = "Wb"
    # one_text = [rem_dubspace(" ".join(pickle.load(open("toktrain.pkl", "rb"))))]
    text_name = "Sg. Training Glosses"
    text_designation = "Sg"
    one_text = [rem_dubspace(" ".join(load_conllu('sga_dipsgg-ud-test_combined_POS.conllu')))]

    # # Map all test and training characters
    mappings = map_chars(load_data(one_text, text_name))
    char_dict, rchardict, size_vocab = mappings[0], mappings[1], mappings[2]

    # # Set how many characters the model should look at before predicting an upcoming character
    buffer_characters = 10

    # # Organize into sequences
    x_train = sequence(one_text, buffer_characters, text_name)

    # # Encode all glosses using mapping (for use with padding)
    x_train = encode(x_train, char_dict, text_name)

    # # Split training sequences into x and y, and one hot encode each
    one_hots = onehot_split(x_train, size_vocab, text_name)
    x_train, y_train = one_hots[0], one_hots[1]
