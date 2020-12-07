
import time
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


def makemod(LSTM_layers, LSTM_sizes, Dense_layers, Dense_sizes, x_train, y_train, val_size=0.1,
            num_epochs=25, batches=100, name_prefix="", name_infix="", name_suffix="",
            loss_type="categorical_crossentropy", opt="adam"):
    """Defines, compiles and fits (multiple) models"""
    for lstmlayer in LSTM_layers:
        for lstmsize in LSTM_sizes:
            for denselayer in Dense_layers:
                for densesize in Dense_sizes:
                    if denselayer != 0:
                        NAME = f"{name_prefix}{name_infix}spaces{name_suffix}, {lstmlayer}x{lstmsize} LSTMs, " \
                               f"{denselayer}x{densesize} Dense, {num_epochs} Epochs"
                        models_dir = os.getcwd() + "\\models\\space_models"
                        try:
                            models_list = os.listdir(models_dir)
                        except FileNotFoundError:
                            main_dir = os.getcwd()
                            try:
                                os.chdir(main_dir + "\\models")
                            except:
                                os.mkdir("models")
                                os.chdir(main_dir + "\\models")
                            os.mkdir("space_models")
                            os.chdir(main_dir)
                            models_list = os.listdir(models_dir)
                        if NAME not in models_list:
                            model = Sequential()
                            for l in range(lstmlayer - 1):
                                model.add(
                                    LSTM(lstmsize, return_sequences=True, input_shape=(x_train.shape[1],
                                                                                       x_train.shape[2])))
                            model.add(LSTM(lstmsize, input_shape=(x_train.shape[1], x_train.shape[2])))
                            for l in range(denselayer):
                                model.add(Dense(densesize, activation='relu'))
                            model.add(Dense(y_train.shape[1], activation='softmax'))
                            print(model.summary())
                            # Log the model
                            tb = TensorBoard(log_dir=f"logs\space_logs\{NAME}")
                            # Compile model
                            model.compile(loss=loss_type, optimizer=opt, metrics=["accuracy"])
                            es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
                            model.fit(x_train, y_train, epochs=num_epochs, batch_size=batches,
                                      validation_split=val_size, shuffle=True, verbose=2, callbacks=[tb, es])
                            print("Model {} created".format(NAME))
                            # Save Model
                            model.save(f"models\space_models\{NAME}")
                            print("Model {} saved".format(NAME))
                        else:
                            print(f"Skipped model which already exists:\n    {NAME}")
                    else:
                        NAME = f"{name_prefix}{name_infix}spaces{name_suffix}, {lstmlayer}x{lstmsize} LSTMs, " \
                               f"0 Dense, {num_epochs} Epochs"
                        models_dir = os.getcwd() + "\\models\\space_models"
                        try:
                            models_list = os.listdir(models_dir)
                        except FileNotFoundError:
                            main_dir = os.getcwd()
                            try:
                                os.chdir(main_dir + "\\models")
                            except:
                                os.mkdir("models")
                                os.chdir(main_dir + "\\models")
                            os.mkdir("space_models")
                            os.chdir(main_dir)
                            models_list = os.listdir(models_dir)
                        if NAME not in models_list:
                            model = Sequential()
                            for l in range(lstmlayer - 1):
                                model.add(LSTM(lstmsize, return_sequences=True, input_shape=(x_train.shape[1],
                                                                                             x_train.shape[2])))
                            model.add(LSTM(lstmsize, input_shape=(x_train.shape[1], x_train.shape[2])))
                            model.add(Dense(y_train.shape[1], activation='softmax'))
                            print(model.summary())
                            # Log the model
                            tb = TensorBoard(log_dir=f"logs\space_logs\{NAME}")
                            # Compile model
                            model.compile(loss=loss_type, optimizer=opt, metrics=["accuracy"])
                            es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
                            model.fit(x_train, y_train, epochs=num_epochs, batch_size=batches,
                                      validation_split=val_size, shuffle=True, verbose=2, callbacks=[tb, es])
                            print("Model {} created".format(NAME))
                            # Save Model
                            model.save(f"models\space_models\{NAME}")
                            print("Model {} saved".format(NAME))
                        else:
                            print(f"Skipped model which already exists:\n    {NAME}")


# Test Area

if __name__ == "__main__":

    # Test GPU is recognised and enable memory growth as necessary

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(f"Num GPUs Available: {len(physical_devices)}")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    start_time = time.time()

    for text_designation in ["Wb", "Sg"]:
        buffer_characters = 10
        if text_designation == "Wb":
            text_name = "Wb. Training Glosses"
        elif text_designation == "Sg":
            text_name = "Sg. Training Glosses"
        else:
            raise RuntimeError(f"Unknown text designation: {text_designation}")
        name_prefix = f"{text_designation}_"
        for clenliness in ["regular", "super"]:
            if clenliness == "regular":
                if text_designation == "Wb":
                    gloss_list = load_data()[0]
                elif text_designation == "Sg":
                    gloss_list = add_finalspace(remove_chars(remove_non_glosses(split_on_latin(
                        load_conllu('sga_dipsgg-ud-test_combined_POS.conllu')))))
                else:
                    raise RuntimeError(f"Unknown text designation {text_designation}")
                mappings = map_chars([gloss_list] + load_data())
                char_dict, rchardict, total_chars = mappings[0], mappings[1], mappings[2]
                x_train = sequence(gloss_list, buffer_characters, text_name)
                x_train = encode(x_train, char_dict, text_name)
            elif clenliness == "super":
                if text_designation == "Wb":
                    gloss_list = load_data(True)[0]
                elif text_designation == "Sg":
                    gloss_list = add_finalspace(remove_chars(remove_non_glosses(split_on_latin(
                        load_conllu('sga_dipsgg-ud-test_combined_POS.conllu', True)))))
                else:
                    raise RuntimeError(f"Unknown text designation {text_designation}")
                mappings = map_chars([gloss_list] + load_data(True))
                char_dict, rchardict, total_chars = mappings[0], mappings[1], mappings[2]
                x_train = sequence(gloss_list, buffer_characters, text_name)
                x_train = encode(x_train, char_dict, text_name)
            else:
                raise RuntimeError(f"Unknown clenliness state: {clenliness}")
            name_infix = f"{clenliness}_"
            for output_type in ["binary", "full"]:
                if output_type == "binary":
                    one_hots = onehot_split(x_train, total_chars, text_name, True, char_dict)
                    name_suffix = f"_bi"
                elif output_type == "full":
                    one_hots = onehot_split(x_train, total_chars, text_name)
                    name_suffix = ""
                else:
                    raise RuntimeError(f"Unknown output type: {output_type}")
                onehot_x_train, y_train = one_hots[0], one_hots[1]
                val_size = 0.1
                LSTM_layers = [2]
                LSTM_sizes = [25, 50, 75, 100]
                Dense_layers = [0, 1]
                Dense_sizes = [y_train.shape[1], total_chars]
                Epochs = 250
                Batches = 100
                makemod(LSTM_layers, LSTM_sizes, Dense_layers, Dense_sizes, onehot_x_train, y_train, val_size,
                        Epochs, Batches, name_prefix, name_infix, name_suffix)

    end_time = time.time()
    seconds_elapsed = end_time - start_time
    print("Time elapsed: " + time_elapsed(seconds_elapsed))

# *Terminal*>tensorboard --logdir=logs\space_logs\
