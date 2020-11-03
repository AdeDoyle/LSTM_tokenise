import time
from os import listdir
import os.path as op
from OpenDocx import get_text
import pickle
from PrepareHandContent import remove_non_glosses
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
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


# Set how many characters the model should look at before predicting an upcoming character
pre_characters = 7
print("Set training parameters...")


# # Import the training set of glosses for use as the single training text
# one_text_in = open("toktrain.pkl", "rb")
# one_text = " ".join(pickle.load(one_text_in))
# text_name = "Wb. Training Glosses"

# Import and clean CELT texts for use as the single training text
clean_text_list = []
all_clean_files = [f for f in listdir("CELT_Texts_Clean") if op.isfile(op.join("CELT_Texts_Clean", f))]
for cf in all_clean_files:
    cf = "".join(cf.split(".docx"))
    if cf != ".DS_Store":
        cf = op.join("CELT_Texts_Clean", cf)
        clean_text_list.append(get_text(cf))
one_text = " ".join(clean_text_list)
one_text = " ".join(one_text.split("\n"))
while "  " in one_text:
    one_text = " ".join(one_text.split("  "))
text_name = "CELT Collection"

# Import test and train sets for character mapping
train_in = open("toktrain.pkl", "rb")
train_set = pickle.load(train_in)
test_in = open("toktest.pkl", "rb")
test_set = pickle.load(test_in)
x_train = remove_non_glosses(train_set)
# temp = []  # Reverse x_train for reverse models
# for x_trainer in x_train[::-1]:  # Reverse x_train for reverse models
#     new_trainer = x_trainer[::-1]  # Reverse x_train for reverse models
#     temp.append(new_trainer)  # Reverse x_train for reverse models
# x_train = temp  # Reverse x_train for reverse models
x_test, y_test = test_set[0], test_set[1]
print("Loaded {}, training, and test data...".format(text_name))


# Combine all test and train sets into one list for later operations
all_testtrain = [one_text] + x_train + x_test + y_test


# Maps all characters in both sets
chars = sorted(list(set("".join(all_testtrain))))
chardict = dict((c, i + 1) for i, c in enumerate(chars))
chardict["$"] = 0
vocab_size = len(chardict)
print('    No. of characters: %d' % vocab_size)
rchardict = dict((i + 1, c) for i, c in enumerate(chars))
rchardict[0] = "$"
print("Mapped Characters...")


def encode(string_list):
    """Encodes a list of glosses using mapping"""
    num_list = list()
    for plain_string in string_list:
        encoded_string = [chardict[char] for char in plain_string]
        num_list.append(encoded_string)
    return num_list


def sequence(string_list):
    """Organises gloss content into sequences"""
    one_liner = " ".join(string_list)
    sequences = list()
    for i in range(pre_characters, len(one_liner)):
        # select sequence of tokens
        seq = one_liner[i - pre_characters: i + 1]
        # store this seq
        sequences.append(seq)
    print('    Total Sequences for {0}: {1}'.format(seq_name, len(sequences)))
    return sequences


# Organize into sequences
seq_name = "training"
x_train = sequence([one_text])
print("Organised {} into sequences...".format(text_name))


# Encode all glosses using mapping (for use with padding)
x_train = encode(x_train)
print("Encoded {}...".format(text_name))


# Separate training into input and output, and one hot encode all training sequences
sequences = np.array(x_train)
x_train, y_train = sequences[:, : - 1], sequences[:, - 1]
sequences = [to_categorical(x, num_classes=vocab_size) for x in x_train]
x_train = np.array(sequences)
y_train = to_categorical(y_train, num_classes=vocab_size)
print("One Hot encoded {}...".format(text_name))


# Define model
nodes = 54
epochs = 10
model_name = "n{}_{}x{}-{}-CELT-Collection-test.h5".format(pre_characters, nodes, nodes, epochs)  # Name model
model = Sequential()
model.add(LSTM(nodes, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))  # 1 Hidden Layer
# model.add(LSTM(nodes, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))  # 2 Hidden Layers
# model.add(LSTM(nodes, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))  # 3 Hidden Layers
model.add(LSTM(nodes, input_shape=(x_train.shape[1], x_train.shape[2])))  # 4 Hidden Layers
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# Log the model
tb = TensorBoard(log_dir="logs/{}".format(model_name[:-3]))
# Compile and fit model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=epochs, validation_split=0.1, verbose=2, callbacks=[tb])
print("Created Model...")


# Save the model
model.save(model_name)
# # Save the mapping
# pickle.dump(chardict, open('char_mappingTBF.pkl', 'wb'))  # Name mapping
print("Saved Model...")


# # Load the model
# model = load_model('n3_Tokeniser.h5')  # Model name
# # Load the mapping
# chardict = pickle.load(open('char_mappingTBF.pkl', 'rb'))  # Mapping Name


end_time = time.time()
seconds_elapsed = end_time - start_time
print("Time elapsed: " + time_elapsed(seconds_elapsed))


# Generate a sequence of characters with a language model
def generate_seq(model, mapping, seq_length, seed_text, n_chars):
    in_text = seed_text
    # Generate a fixed number of characters
    for _ in range(n_chars):
        # Encode the characters as integers
        encoded = [mapping[char] for char in in_text]
        # Truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # One hot encode
        encoded = to_categorical(encoded, num_classes=len(mapping))
        # encoded = encoded.reshape(1, encoded.shape[0], encoded.shape[1])  # Causes shaping error, comment out
        # Predict character
        yhat = model.predict_classes(encoded, verbose=0)
        # Reverse map integer to character
        out_char = ''
        for char, index in mapping.items():
            if index == yhat:
                out_char = char
                break
        # Append to input
        in_text += char
    return in_text


# test 1
print(generate_seq(model, chardict, pre_characters, 'biuu sa', 20))
# test 2
print(generate_seq(model, chardict, pre_characters, '.i. biu', 20))
# test 3
print(generate_seq(model, chardict, pre_characters, 'ar is d', 20))


"""
Model 1: n5_TBFTokeniser.h5

One Hidden Layer
LSTM cells: 40
Epochs: 1000
Buffer: 5 pre-characters

   Epoch 1/1000
    - 3s - loss: 2.9281 - acc: 0.1807
   Epoch 1000/1000
    - 3s - loss: 0.9174 - acc: 0.6954

   Time elapsed: 43.73444213072459 min

   $$$$$r ind ordnasc ol ail
   $$.i. segait ⁊ ór di ór ⁊
   $aris illei tíri ni béo i


Model 2: n5_TBF2HLTokeniser.h5

Two Hidden Layers
LSTM cells: 40 x 40
Epochs: 1000
Buffer: 5 pre-characters

   Epoch 1/1000
    - 5s - loss: 2.9282 - acc: 0.1839
   Epoch 1000/1000
    - 4s - loss: 0.5109 - acc: 0.7786

   Time elapsed: 1.2472326714462705 hr

   $$$$$.i. ished and $$$$$.
   $$.i. ished and $$$$$.i. 
   $arisaib $$$$$.i. ished a


Model 3: n5_TBF4HLTokeniser.h5

Four Hidden Layers
LSTM cells: 40 x 40 x 27 x 27
Epochs: 1000
Buffer: 5 pre-characters

   Epoch 1/1000
    - 10s - loss: 2.9596 - acc: 0.1832
   Epoch 1000/1000
    - 8s - loss: 0.5349 - acc: 0.7785

   Time elapsed: 2.259919977254338 hr

   $$$$$ is é dírimm athesc 
   $$.i. conall cernach ara 
   $ariscet nechach issa tec


Model 4: n3_TBFTokeniser.h5

One Hidden Layer
LSTM cells: 40
Epochs: 1000
Buffer: 3 pre-characters

   Epoch 1/1000
    - 2s - loss: 2.9394 - acc: 0.1838
   Epoch 1000/1000
    - 2s - loss: 1.3565 - acc: 0.5013

   Time elapsed: 30.69184961716334 min

   $$$$$aith co na mbrat in 
   $$.i.ile thar sa in tair 
   $ariss and a mbrat in tai


Model 5: n3_TBF2HLTokeniser.h5

Two Hidden Layers
LSTM cells: 40 x 40
Epochs: 1000
Buffer: 3 pre-characters

   Epoch 1/1000
    - 4s - loss: 2.9466 - acc: 0.1849
   Epoch 1000/1000
    - 3s - loss: 1.3057 - acc: 0.5030

   Time elapsed: 52.01483686765035 min

   $$$$$acht na mbréi do chu
   $$.i. mór ⁊ arggaib argga
   $arissin tair arggaib arg


Model 6: n3_TBF4HLTokeniser.h5

Four Hidden Layers
LSTM cells: 40 x 40 x 27 x 27
Epochs: 1000
Buffer: 3 pre-characters

   Epoch 1/1000
    - 7s - loss: 2.9666 - acc: 0.1829
   Epoch 1000/1000
    - 5s - loss: 1.3500 - acc: 0.5023

   Time elapsed: 1.520814772248268 hr

   $$$$$ ⁊ a mbréib ar sin t
   $$.i.eird a mbréib ar sin
   $arisi in tair i mbréib a


Model 7: n5_TBF3HLTokeniser.h5

Three Hidden Layers
LSTM cells: 54 x 40 x 40
Epochs: 1000
Buffer: 5 pre-characters

   Epoch 1/1000
    - 8s - loss: 2.9379 - acc: 0.1830
   Epoch 1000/1000
    - 7s - loss: 0.4579 - acc: 0.7828

   Time elapsed: 1.7887259469429653 hr

   biuush bóinni for fír tha
   .i. aní is maith ol sé do
   arisd ⁊ tar mo ingen in d


Model 8: n5_TBF2HLTokeniserV2.h5

Two Hidden Layers
LSTM cells: 54 x 54
Epochs: 1000
Buffer: 5 pre-characters

   Epoch 1/1000
    - 5s - loss: 2.9023 - acc: 0.1911
   Epoch 1000/1000
    - 5s - loss: 0.4501 - acc: 0.7795

   Time elapsed: 1.2710270761118996 hr

   biuus fodlid dún dia taba
   .i. ataat téora oa dubgla
   arisdi ra hétach documlát


Model 9: n5_TBF3HLTokeniserV2.h5

Three Hidden Layers
LSTM cells: 54 x 54 x 54
Epochs: 1000
Buffer: 5 pre-characters

   Epoch 1/1000
    - 8s - loss: 2.9287 - acc: 0.1828
   Epoch 1000/1000
    - 7s - loss: 0.4388 - acc: 0.7835

   Time elapsed: 1.854659367799759 hr

   biuus im thísi amin ol fr
   .i. anísin co naccatar ní
   arisd ní dolléic oc fuini


Model 10: n5_TBF1HLTokeniserV2.h5

One Hidden Layer
LSTM cells: 54
Epochs: 1000
Buffer: 5 pre-characters

   Epoch 1/1000
    - 3s - loss: 2.9010 - acc: 0.1840
   Epoch 1000/1000
    - 3s - loss: 0.5609 - acc: 0.7711

   Time elapsed: 43.8698596517245 min

   biuusnibdirntas dó cid do
   .i. ailill ⁊ medb issin d
   arisd cerht lat do sétaib


Model 11: n7_TBF1HLTokeniser.h5

One Hidden Layer
LSTM cells: 54
Epochs: 1000
Buffer: 7 pre-characters

   Epoch 1/1000
    - 4s - loss: 2.8799 - acc: 0.1878
   Epoch 1000/1000
    - 3s - loss: 0.3248 - acc: 0.8892

   Time elapsed: 57.28009958267212 min

   $$biuuso tri ⁊ fodmer ol ss
   $$.i. argdait co suidi arda
   $$arisde boí lacmait frie c


Model 12: n3_TBF1HLTokeniserV2.h5

One Hidden Layer
LSTM cells: 54
Epochs: 1000
Buffer: 3 pre-characters

   Epoch 1/1000
    - 3s - loss: 2.9066 - acc: 0.1946
   Epoch 1000/1000
    - 2s - loss: 1.3168 - acc: 0.4993

   Time elapsed: 31.036637814839683 min

   biur issin tair issin t
   .i. tair issin tair iss
   arissin tair issin tair
"""

