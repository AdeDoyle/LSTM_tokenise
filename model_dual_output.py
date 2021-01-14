
import time
import pickle
from preprocess_data import rem_dubspace, remove_chars, remove_non_glosses, add_finalspace
from preprocess_dual import load_conllu, map_chars, load_data, split_on_latin, split_return_latin, repair_split_latin
from tokenise import time_elapsed
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import load_model
from test_model_dual import tokenise
from SaveDocx import save_docx


def output_tzmod(model, mapping, test_set, size_vocab, buffer=0):
    tokenised_glosses = list()
    x_test = test_set
    mod = load_model(model)
    for x_pos in range(len(x_test)):
        x = x_test[x_pos]
        x_toks = tokenise(mod, x, mapping, size_vocab, buffer)
        tokenised_glosses.append(x_toks.strip())
    return tokenised_glosses


if __name__ == "__main__":

    # Check for GPU and enable memory growth as necessary
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(f"Num GPUs Available: {len(physical_devices)}")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Load test data
    test_in = open("toktest.pkl", "rb")
    test_set = pickle.load(test_in)
    x_latsplit = split_return_latin(test_set[0])
    x_test, x_indices = x_latsplit[0], x_latsplit[1]

    # Identify models to test, with their appropriate character conversion dictionaries and dictionary sizes

    text_name_1 = "Wb. Training Glosses"
    text_designation_1 = "Wb"
    one_text = [rem_dubspace(" ".join(add_finalspace(remove_chars(remove_non_glosses(split_on_latin(
        pickle.load(open("toktrain.pkl", "rb"))))))))]
    mapping_1 = map_chars(load_data(one_text, text_name_1))
    model_1 = "models\\dual_models\\Wb-dual, 2 layer(s) of 75 LSTM Nodes, 1 Dense, 250 Ep, No Bat, 10.0% Val"
    char_dict_1, rchardict_1, size_vocab_1 = mapping_1[0], mapping_1[1], mapping_1[2]

    text_name_2 = "Sg. Training Glosses"
    text_designation_2 = "Sg"
    two_text = [rem_dubspace(" ".join(add_finalspace(remove_chars(remove_non_glosses(split_on_latin(
        load_conllu('sga_dipsgg-ud-test_combined_POS.conllu')))))))]
    mapping_2 = map_chars(load_data(two_text, text_name_2))
    model_2 = "models\\dual_models\\Sg-dual, 2 layer(s) of 100 LSTM Nodes, 1 Dense, 250 Ep, No Bat, 10.0% Val"
    char_dict_2, rchardict_2, size_vocab_2 = mapping_2[0], mapping_2[1], mapping_2[2]

    allmods = [
        [model_1, char_dict_1, size_vocab_1],
        [model_2, char_dict_2, size_vocab_2]
    ]

    base_dir = os.getcwd()
    mods_dir = base_dir + "\\models\\dual_models"
    os.chdir(mods_dir)
    if "output" not in os.listdir():
        os.mkdir("output")
    os.chdir(base_dir)

    # Test Forward Models
    start_time = time.time()

    for mod in allmods:
        model = mod[0]
        char_dict = mod[1]
        dict_size = mod[2]
        print(f"Model in progress:\n    {model}")
        output = output_tzmod(model, char_dict, x_test, dict_size, 10)
        output = "\n".join(repair_split_latin(output, x_indices))
        save_docx(output, "models\\dual_models\\output\\".join(model.split("models\\dual_models\\")))
        print("    Complete!\n")

    end_time = time.time()
    seconds_elapsed = end_time - start_time
    print("Time elapsed: " + time_elapsed(seconds_elapsed))

    # Test Forward Models
    start_time = time.time()
