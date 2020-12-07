
import time
from preprocess_data import rem_dubspace, remove_chars, remove_non_glosses, add_finalspace
from preprocess_spaces import load_conllu, map_chars, load_data, split_on_latin
from tokenise_dual import time_elapsed
from nltk import edit_distance as ed
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical


def tokenise(model, intext, mapping, size_vocab, output_type, buffer=0):
    """Takes a trained language model and a text, returns the text tokenised as per the language model"""
    buffer_text = buffer * "$"
    text = buffer_text + intext + buffer_text
    outlist = []
    for _ in range(len(text) - (buffer * 2)):
        let = text[buffer]
        forward_text = text[:buffer]
        remainder = "".join(text[buffer + 1:].split(" "))[:buffer]
        reverse_text = remainder[::-1]
        outlist_text = "".join(outlist)
        # if there are enough letters in the outlist to make predictions from
        # take the chunk of letters to predict from from the outlist
        if buffer <= len(outlist):
            forward_text = outlist_text[-buffer:]
        # if there aren't enough letters in the outlist to make predictions from yet
        # combine what's in the outlist with what's in the text
        else:
            forward_text = forward_text[:buffer - (len(outlist))] + outlist_text
            if len(forward_text) > buffer:
                forward_text = forward_text[-buffer:]
        forward_encoded = [mapping[char] for char in forward_text]
        reverse_encoded = [mapping[char] for char in reverse_text]
        # print([forward_text], [reverse_text])
        encoded = forward_encoded + reverse_encoded
        # print(encoded)
        encoded = np.array([to_categorical(encoded, num_classes=size_vocab)])
        pred = model.predict_classes(encoded, verbose=0)[0]
        # if the letter were trying to predict isn't a space in the text
        # predict a character
        if let != " " and output_type == "full":
            # if the prediction is not a space
            # just add the letter we were trying to predict to the outlist
            if pred != 1:
                outlist.append(let)
            # if the prediction is a space
            # add a space to the outlist followed by the letter that was in the text
            else:
                outlist.append(" " + let)
        elif let != " " and output_type == "binary":
            # if the prediction is not a space
            # just add the letter we were trying to predict to the outlist
            if pred == 0:
                outlist.append(let)
            # if the prediction is a space
            # add a space to the outlist followed by the letter that was in the text
            elif pred == 1:
                outlist.append(" " + let)
        # if the letter we're trying to predict is a space in the text
        # just append a space to the outlist
        else:
            outlist.append(" ")
        text = text[1:]
    outtext = "".join(outlist)
    if "  " in outtext:
        while "  " in outtext:
            outtext = " ".join(outtext.split("  "))
    # print(outtext)
    return outtext


def test_tzmod(model, mapping, test_set, size_vocab, output_type, buffer=0):
    edit_dists = []
    count = 0
    x_test, y_test = test_set[0], test_set[1]
    mod = load_model(model)
    for x_pos in range(len(x_test)):
        count += 1
        x = x_test[x_pos]
        y = y_test[x_pos]
        x_toks = tokenise(mod, x, mapping, size_vocab, output_type, buffer)
        e_dist = ed(y, x_toks)
        edit_dists.append(e_dist)
        # print(x)
        # print(y)
        # print(x_toks)
        # print("Gloss {}/41: Edit Distance = {}".format(str(count), str(e_dist)))
    avg_edist = sum(edit_dists) / len(edit_dists)
    return avg_edist


# Test Area

if __name__ == "__main__":

    # Check for GPU and enable memory growth as necessary
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(f"Num GPUs Available: {len(physical_devices)}")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Load regular-cleaned test data
    test_set = load_data()
    wb_data = test_set[0]
    sg_data = add_finalspace(remove_chars(remove_non_glosses(split_on_latin(
        load_conllu('sga_dipsgg-ud-test_combined_POS.conllu')))))
    x_test, y_test = test_set[1], test_set[2]
    x_and_y_test = [x_test, y_test]

    # Load super-cleaned test data
    sc_test_set = load_data(True)
    sc_wb_data = sc_test_set[0]
    sc_sg_data = add_finalspace(remove_chars(remove_non_glosses(split_on_latin(
        load_conllu('sga_dipsgg-ud-test_combined_POS.conllu', True)))))
    sc_x_test, sc_y_test = sc_test_set[1], sc_test_set[2]
    sc_x_and_y_test = [sc_x_test, sc_y_test]

    # Identify models to test, with their appropriate character conversion dictionaries and dictionary sizes
    mods_dir = os.getcwd() + "\\models\\space_models"

    one_text = [rem_dubspace(" ".join(sc_wb_data))]
    mapping_1 = map_chars([one_text] + sc_test_set)
    model_1 = "Wb_super_spaces_bi, 2x50 LSTMs, 1x2 Dense, 250 Epochs"
    char_dict_1, rchardict_1, size_vocab_1 = mapping_1[0], mapping_1[1], mapping_1[2]

    two_text = [rem_dubspace(" ".join(sc_sg_data))]
    mapping_2 = map_chars([two_text] + sc_test_set)
    # model_2 = "Sg_regular_spaces_bi, 2x100 LSTMs, 1x2 Dense, 250 Epochs"
    model_2 = "Sg_super_spaces_bi, 2x100 LSTMs, 1x28 Dense, 250 Epochs"
    char_dict_2, rchardict_2, size_vocab_2 = mapping_2[0], mapping_2[1], mapping_2[2]

    allmods = [
        [model_1, char_dict_1, size_vocab_1, "binary", True],
        [model_2, char_dict_2, size_vocab_2, "binary", True]
    ]

    # allmods = list()
    # mod_names = os.listdir(mods_dir)
    # for mod in mod_names:
    #     designation = mod[:2]
    #     super_clean = False
    #     if "super_spaces" in mod:
    #         super_clean = True
    #     if "_spaces," in mod:
    #         out_type = "full"
    #     elif "_bi," in mod:
    #         out_type = "binary"
    #     else:
    #         raise RuntimeError(f"Could not determine output for model:\n    {mod}")
    #     text = False
    #     if super_clean:
    #         wb_text = [rem_dubspace(" ".join(sc_wb_data))]
    #         sg_text = [rem_dubspace(" ".join(sc_sg_data))]
    #     else:
    #         wb_text = [rem_dubspace(" ".join(wb_data))]
    #         sg_text = [rem_dubspace(" ".join(sg_data))]
    #     if designation == "Wb":
    #         text = wb_text
    #     elif designation == "Sg":
    #         text = sg_text
    #     if super_clean:
    #         mapping = map_chars([text] + sc_test_set)
    #     else:
    #         mapping = map_chars([text] + test_set)
    #     char_dict, size_vocab = mapping[0], mapping[2]
    #     allmods.append([mod, char_dict, size_vocab, out_type, super_clean])

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
    print(f"Original Gloss Score:\n    {avg_ed_dist}")

    # Test Forward Models
    start_time = time.time()

    modscores = []
    os.chdir(mods_dir)
    for mod in allmods:
        model = mod[0]
        char_dict = mod[1]
        dict_size = mod[2]
        output = mod[3]
        superclean = mod[4]
        if superclean:
            testing = sc_x_and_y_test
        else:
            testing = x_and_y_test
        print(model)
        score = test_tzmod(model, char_dict, testing, dict_size, output, 10)
        print(f"    {score}")
        modscores.append(score)
    modscores = [i for i in modscores if i != avg_ed_dist]
    best_score = min(modscores)
    print(f"Best Model:\n    {allmods[modscores.index(best_score)][0]}")

    end_time = time.time()
    seconds_elapsed = end_time - start_time
    print("Time elapsed: " + time_elapsed(seconds_elapsed))
