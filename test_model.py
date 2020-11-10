
import time
import pickle
from preprocess_data import load_conllu, rem_dubspace, map_chars, load_data
from tokenise import time_elapsed
from nltk import edit_distance as ed
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


def tokenise(model, intext, mapping, size_vocab, buffer=0):
    """Takes a trained language model and a text, returns the text tokenised as per the language model"""
    mod = load_model(model)
    buffer_text = buffer * "$"
    text = buffer_text + intext
    outlist = []
    for i in range(len(text) - buffer):
        let = text[buffer]
        text_chunk = text[:buffer]
        outlist_text = "".join(outlist)
        # if there are enough letters in the outlist to make predictions from
        # take the chunk of letters to predict from from the outlist
        if buffer <= len(outlist):
            text_chunk = outlist_text[-buffer:]
        # if there aren't enough letters in the outlist to make predictions from yet
        # combine what's in the outlist with what's in the text
        else:
            text_chunk = text_chunk[:buffer - (len(outlist))] + outlist_text
        encoded = [mapping[char] for char in text_chunk]
        encoded = pad_sequences([encoded], maxlen=buffer, truncating='pre')
        encoded = np.array(to_categorical(encoded, num_classes=size_vocab))
        pred = mod.predict_classes(encoded, verbose=0)[0]
        # if the letter were trying to predict isn't a space in the text
        # predict a character
        if let != " ":
            # if the prediction is not a space
            # just add the letter we were trying to predict to the outlist
            if pred != 1:
                outlist.append(let)
            # if the prediction is a space
            # add a space to the outlist followed by the letter that was in the text
            else:
                outlist.append(" " + let)
        # if the letter we're trying to predict is a space in the text
        # just append a space to the outlist
        else:
            outlist.append(" ")
        text = text[1:]
    outtext = "".join(outlist)
    if "  " in outtext:
        raise RuntimeError("Double spacing found in model output")
    # outtext = " ".join(outtext.split("  "))
    # print(outtext)
    return outtext


def test_tzmod(model, mapping, size_vocab, buffer=0):
    edit_dists = []
    count = 0
    for x_pos in range(len(x_test)):
        count += 1
        x = x_test[x_pos]
        y = y_test[x_pos]
        x_toks = tokenise(model, x, mapping, size_vocab, buffer)
        e_dist = ed(y, x_toks)
        edit_dists.append(e_dist)
        # print(x)
        # print(x_toks)
        # print(y)
        # print("Gloss {}/41: Edit Distance = {}".format(str(count), str(e_dist)))
    avg_edist = sum(edit_dists) / len(edit_dists)
    return avg_edist


# Test Area

if __name__ == "__main__":

    # Check for GPU and enable memory growth as necessary
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(f"Num GPUs Available: {len(physical_devices)}")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Load test data
    test_in = open("toktest.pkl", "rb")
    test_set = pickle.load(test_in)
    x_test, y_test = test_set[0], test_set[1]

    # Identify models to test, with their appropriate character conversion dictionaries and dictionary sizes

    text_name_1 = "Wb. Training Glosses"
    text_designation_1 = "Wb"
    one_text = [rem_dubspace(" ".join(pickle.load(open("toktrain.pkl", "rb"))))]
    mapping_1 = map_chars(load_data(one_text, text_name_1))
    model_1 = "models\\Wb-model, 2 layer(s) of 75 LSTM Nodes, 1 Dense, 100 Ep, No Bat, 10.0% Val"
    char_dict_1, rchardict_1, size_vocab_1 = mapping_1[0], mapping_1[1], mapping_1[2]

    text_name_2 = "Sg. Training Glosses"
    text_designation_2 = "Sg"
    two_text = [rem_dubspace(" ".join(load_conllu('sga_dipsgg-ud-test_combined_POS.conllu')))]
    mapping_2 = map_chars(load_data(two_text, text_name_2))
    model_2 = "models\\Sg-model, 2 layer(s) of 50 LSTM Nodes, 1 Dense, 100 Ep, No Bat, 10.0% Val"
    char_dict_2, rchardict_2, size_vocab_2 = mapping_2[0], mapping_2[1], mapping_2[2]

    allmods = [
        [model_1, char_dict_1, size_vocab_1],
        [model_2, char_dict_2, size_vocab_2]
    ]

    # allmods = list()
    # for designation in ["Wb", "Sg"]:
    #     text_name = False
    #     text = False
    #     if designation == "Wb":
    #         text_name = "Wb. Training Glosses"
    #         text = [rem_dubspace(" ".join(pickle.load(open("toktrain.pkl", "rb"))))]
    #     elif designation == "Sg":
    #         text_name = "Sg. Training Glosses"
    #         text = [rem_dubspace(" ".join(load_conllu('sga_dipsgg-ud-test_combined_POS.conllu')))]
    #     mapping = map_chars(load_data(text, text_name))
    #     char_dict, size_vocab = mapping[0], mapping[2]
    #     for nodes in ["25", "50", "60", "75"]:
    #         model = f"models\\{designation}-model, 2 layer(s) of {nodes} LSTM Nodes, 1 Dense, 100 Ep, No Bat, 10.0% Val"
    #         allmods.append([model, char_dict, size_vocab])

    # Test Manually Tokenised Glosses against Untokenised Glosses
    all_ed_dists = []
    gl_count = 0
    for x_num, x_pos in enumerate(x_test):
        gl_count = x_num + 1
        x = x_pos
        y = y_test[x_num]
        ed_dist = ed(y, x)
        all_ed_dists.append(ed_dist)
        # print(x)
        # print(y)
        # print("Gloss {}/41: Edit Distance = {}".format(str(gl_count), str(ed_dist)))
    avg_ed_dist = sum(all_ed_dists) / len(all_ed_dists)
    print("Original Gloss Score:\n    {}".format(avg_ed_dist))

    # Test Forward Models
    start_time = time.time()

    modscores = []
    for mod in allmods:
        model = mod[0]
        char_dict = mod[1]
        dict_size = mod[2]
        score = test_tzmod(model, char_dict, dict_size, 10)
        print(model)
        print("    {}".format(score))
        modscores.append(score)
    best_score = min(modscores)
    print(f"Best Model:\n    {allmods[modscores.index(best_score)][0]}")

    end_time = time.time()
    seconds_elapsed = end_time - start_time
    print("Time elapsed: " + time_elapsed(seconds_elapsed))
