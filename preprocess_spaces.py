
import unidecode
import pickle
import re
from conllu import parse
from functools import reduce
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from preprocess_data import remove_ogham, remove_chars, rem_dubspace, remove_non_glosses, add_finalspace


def split_on_latin(glosslist):
    split_glosslist = list()
    for gloss in glosslist:
        if "*Latin*" in gloss:
            split_list = gloss.split("*Latin*")
            for split_gloss in split_list:
                if split_gloss:
                    split_glosslist.append(split_gloss.strip())
        else:
            split_glosslist.append(gloss)
    return split_glosslist


def load_conllu(conllu_file, superclean=False):
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
    gloss_list = split_on_latin(gloss_list)
    gloss_list = remove_non_glosses(gloss_list)
    gloss_list = remove_ogham(gloss_list)
    gloss_list = remove_chars(gloss_list)
    gloss_list = add_finalspace(gloss_list)
    if superclean:
        gloss_list = [i.lower() for i in gloss_list]
        gloss_list = ["&".join(i.split("⁊")) for i in gloss_list]
        gloss_list = [unidecode.unidecode(i) for i in gloss_list]
        gloss_list = ["⁊".join(i.split("&")) for i in gloss_list]
    print("Sg. training data loaded")
    return gloss_list


def load_data(superclean=False):
    """Load and combine all text data for training and testing the model"""
    x_train = add_finalspace(remove_chars(remove_non_glosses(split_on_latin(pickle.load(open("toktrain.pkl", "rb"))))))
    test_set = pickle.load(open("toktest.pkl", "rb"))
    x_test, y_test = add_finalspace(remove_chars(remove_non_glosses(split_on_latin(test_set[0])))), \
                     add_finalspace(remove_chars(remove_non_glosses(split_on_latin(test_set[1]))))
    if superclean:
        x_train = [i.lower() for i in x_train]
        x_test = [i.lower() for i in x_test]
        y_test = [i.lower() for i in y_test]
        x_train = ["&".join(i.split("⁊")) for i in x_train]
        x_test = ["&".join(i.split("⁊")) for i in x_test]
        y_test = ["&".join(i.split("⁊")) for i in y_test]
        x_train = [unidecode.unidecode(i) for i in x_train]
        x_test = [unidecode.unidecode(i) for i in x_test]
        y_test = [unidecode.unidecode(i) for i in y_test]
        x_train = ["⁊".join(i.split("&")) for i in x_train]
        x_test = ["⁊".join(i.split("&")) for i in x_test]
        y_test = ["⁊".join(i.split("&")) for i in y_test]
        glosses = list()
        for gloss in x_train:
            if "[?]" in gloss:
                breakpat = re.compile(r'\w*\[\?\]\w*')
                breakpatiter = breakpat.findall(gloss)
                for break_word in breakpatiter:
                    gloss = "@".join(gloss.split(break_word))
                gloss_splits = remove_non_glosses([i.strip() + " " for i in gloss.split("@")])
                if gloss_splits:
                    for sub_gloss in gloss_splits:
                        glosses.append(sub_gloss)
            else:
                glosses.append(gloss)
        x_train = [i for i in glosses]
    print("Wb. training data loaded\nWb. test data loaded (for character mapping only)")
    return [x_train, x_test, y_test]


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
    """Organises gloss content into forward and reverse sequences leading to the character to be identified
       Returns the forward sequence, reverse sequence, and character"""
    sequences = list()
    for string in string_list:
        one_liner = ("$" * buffer_len) + rem_dubspace(string) + ("$" * buffer_len)
        for i in range(buffer_len, len(one_liner) - buffer_len):
            # select sequence of tokens
            seq = one_liner[i - buffer_len: i]
            # select the reverse sequence of tokens
            remainder = "".join(one_liner[i + 1:].split(" "))[:buffer_len]
            rseq = remainder[::-1]
            # select the character to be guessed
            char = one_liner[i]
            # store this seq
            sequences.append([seq, rseq, char])
    print(f"Buffer length set to: {buffer_len}")
    print(f"{text_name} organised into {len(sequences)} sequences")
    return sequences


def encode(sequence_lists, chardict, text_name):
    """Encodes a list of gloss sequences using mapping from letters to numbers"""
    num_list = list()
    for seq_list in sequence_lists:
        encoded_sequence = list()
        for plain_string in seq_list:
            encoded_string = [chardict[char] for char in plain_string]
            encoded_sequence.append(encoded_string)
        num_list.append(encoded_sequence)
    print(f"{text_name} numerically encoded")
    return num_list


def onehot_split(num_lists, vocab_size, text_name, binary_output=False, char_dict=None):
    """Turns numerically encoded sequences into a numpy array
       Splits arrays into x_train and y_train
       One hot encodes x_train and y_train"""
    sequences = [to_categorical(x, num_classes=vocab_size) for x in np.array([i[0] + i[1] for i in num_lists])]
    x_train = np.array(sequences)
    y_train = [i[2][0] for i in num_lists]
    if binary_output:
        space_char = char_dict.get(" ")
        y_train = np.array([0 if i != space_char else 1 for i in y_train])
        y_train = to_categorical(y_train, num_classes=2)
    else:
        y_train = to_categorical(y_train, num_classes=vocab_size)
    print(f"{text_name} One-Hot encoded:\n    x-train = {x_train.shape}\n    y-train = {y_train.shape}")
    return [x_train, y_train]


# Test Area

if __name__ == "__main__":

    # Test GPU is recognised and enable memory growth as necessary
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(f"Num GPUs Available: {len(physical_devices)}")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # # Choose and name text to train on
    # text_name = "Wb. Training Glosses"
    # text_designation = "Wb"
    # # gloss_list = load_data()[0]  # Regular-clean
    # gloss_list = load_data(True)[0]  # Super-clean
    text_name = "Sg. Training Glosses"
    text_designation = "Sg"
    # gloss_list = add_finalspace(remove_chars(remove_non_glosses(split_on_latin(
    #     load_conllu('sga_dipsgg-ud-test_combined_POS.conllu')))))  # Regular-clean
    gloss_list = add_finalspace(remove_chars(remove_non_glosses(split_on_latin(
        load_conllu('sga_dipsgg-ud-test_combined_POS.conllu', True)))))  # Super-clean

    # Map all test and training characters
    # mappings = map_chars([gloss_list] + load_data())  # Regular-clean
    mappings = map_chars([gloss_list] + load_data(True))  # Super-clean
    char_dict, rchardict, size_vocab = mappings[0], mappings[1], mappings[2]

    # Set how many characters the model should look at before predicting an upcoming character
    buffer_characters = 10

    # Organize into sequences
    x_train = sequence(gloss_list, buffer_characters, text_name)

    # Encode all glosses using mapping (for use with padding)
    x_train = encode(x_train, char_dict, text_name)

    # Split training sequences into x and y, and one hot encode each
    # one_hots = onehot_split(x_train, size_vocab, text_name, True, char_dict)  # Binary output
    one_hots = onehot_split(x_train, size_vocab, text_name)  # Non-binary output
    x_train, y_train = one_hots[0], one_hots[1]
