"""Level 1, 2, 2, 1, 2"""

from functools import lru_cache
from tensorflow.keras.models import load_model
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from RemoveNewlines import remove_newlines, get_pages, get_section, clear_tags, order_glosses, remove_brackets,\
    remove_glossnums
from RemoveLatin import rem_lat
from RemovePunctuation import rempunc_tok
from OpenDocx import get_text
from SaveXlsx import save_xlsx
import os.path as op


@lru_cache(maxsize=500)
def tokenise(model, intext, buffer=0):
    """Takes a trained language model and a text, returns the text tokenised as per the language model"""
    mod = load_model(model)
    buffpat = re.compile(r'n\d{1,2}(pad)?_')
    buffpatitir = buffpat.finditer(model)
    for buff in buffpatitir:
        if "pad" in buff.group():
            padbuff = buff.group()
            buff = "".join(padbuff.split("pad"))
            buffer = int(buff[1:-1])
        else:
            buffer = int(buff.group()[1:-1])
    buffer_text = buffer * "$"
    text = buffer_text + intext
    mapping = pickle.load(open('char_mapping.pkl', 'rb'))
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
        encoded = to_categorical(encoded, num_classes=len(mapping))
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
    # outtext = " ".join(outtext.split("  "))
    return outtext


@lru_cache(maxsize=500)
def rev_tokenise(model, intext, buffer=0):
    """Takes a trained language model and a text, returns the text tokenised as per the language model"""
    mod = load_model(model)
    buffpat = re.compile(r'n\d{1,2}(pad)?_')
    buffpatitir = buffpat.finditer(model)
    for buff in buffpatitir:
        if "pad" in buff.group():
            padbuff = buff.group()
            buff = "".join(padbuff.split("pad"))
            buffer = int(buff[1:-1])
        else:
            buffer = int(buff.group()[1:-1])
    buffer_text = buffer * "$"
    text = buffer_text + intext[::-1]
    mapping = pickle.load(open('char_mapping.pkl', 'rb'))
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
        encoded = to_categorical(encoded, num_classes=len(mapping))
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
    # outtext = " ".join(outtext.split("  "))
    outtext = outtext[::-1]
    return outtext


def predict(mapping, model, text, buffer):
    encoded = [mapping[char] for char in text]
    encoded = pad_sequences([encoded], maxlen=buffer, truncating='pre')
    encoded = to_categorical(encoded, num_classes=len(mapping))
    prediction = model.predict_classes(encoded, verbose=0)[0]
    return prediction


def rev_predict(mapping, rev_model, following_text, rev_buffer):
    rev_encoded = [mapping[char] for char in following_text]
    rev_encoded = pad_sequences([rev_encoded], maxlen=rev_buffer, truncating='pre')
    rev_encoded = to_categorical(rev_encoded, num_classes=len(mapping))
    rev_prediction = rev_model.predict_classes(rev_encoded, verbose=0)[0]
    return rev_prediction


@lru_cache(maxsize=500)
def tokenise_combine(model, rev_model, intext, buffer=0):
    """Takes a trained language model and a text, returns the text tokenised as per the language model"""
    mod = load_model(model)
    rev_mod = load_model(rev_model)
    buffpat = re.compile(r'n\d{1,2}(pad)?_')
    buffpatitir = buffpat.finditer(model)
    for buff in buffpatitir:
        if "pad" in buff.group():
            padbuff = buff.group()
            buff = "".join(padbuff.split("pad"))
            buffer = int(buff[1:-1])
        else:
            buffer = int(buff.group()[1:-1])
    rev_buffpatitir = buffpat.finditer(rev_model)
    for rev_buff in rev_buffpatitir:
        if "pad" in rev_buff.group():
            padbuff = rev_buff.group()
            rev_buff = "".join(padbuff.split("pad"))
            rev_buffer = int(rev_buff[1:-1])
        else:
            rev_buffer = int(rev_buff.group()[1:-1])
    buffer_text = buffer * "$"
    text = buffer_text + intext
    mapping = pickle.load(open('char_mapping.pkl', 'rb'))
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
        # if the letter were trying to predict isn't a space in the text
        # predict a character
        if let != " ":
            pred = predict(mapping, mod, text_chunk, buffer)
            # if the prediction is not a space
            # just add the letter we were trying to predict to the outlist
            if pred != 1:
                outlist.append(let)
            # if the prediction is a space
            # check if the reverse prediction agrees with the prediction
            else:
                following_text = text[rev_buffer + 1:]
                # if the remaining text is longer than the buffer length
                # use all of it
                if len(following_text) >= rev_buffer:
                    following_text = following_text[:rev_buffer]
                # if the remaining text is shorter than the buffer length
                # add buffer characters to it until it's long enough
                else:
                    remainder = rev_buffer - len(following_text)
                    following_text = following_text + ("$" * remainder)
                # reverse the following text and make a reverse prediction to see what should come before it
                following_text = following_text[::-1]
                rev_pred = rev_predict(mapping, rev_mod, following_text, rev_buffer)
                # if the reverse prediction agrees with the prediction on a space
                # add a space to the outlist followed by the letter that was in the text
                if rev_pred == 1:
                    outlist.append(" " + let)
                # if the reverse prediction disagrees with the prediction on a space
                # just add the letter to the outlist
                else:
                    outlist.append(let)
        # if the letter we're trying to predict is a space in the text
        # check whether the forward model predicts there should be a space here
        # comment this section out for the tokenizer to only put in spaces but not take any out
        else:
            pred = predict(mapping, mod, text_chunk, buffer)
            # if the prediction is also a space
            # just add the space to the outlist
            if pred == 1:
                outlist.append(let)
            # if the prediction is not a space
            # check if the reverse prediction agrees with the prediction
            else:
                following_text = text[rev_buffer + 1:]
                # if the remaining text is longer than the buffer length
                # use all of it
                if len(following_text) >= rev_buffer:
                    following_text = following_text[:rev_buffer]
                # if the remaining text is shorter than the buffer length
                # add buffer characters to it until it's long enough
                else:
                    remainder = rev_buffer - len(following_text)
                    following_text = following_text + ("$" * remainder)
                # reverse the following text and make a reverse prediction to see what should come before it
                following_text = following_text[::-1]
                rev_pred = rev_predict(mapping, rev_mod, following_text, rev_buffer)
                # if the reverse prediction does not agree with the prediction that there should not be a space
                # add a space to the outlist
                if rev_pred != 1:
                    outlist.append(let)
        text = text[1:]
    outtext = "".join(outlist)
    # outtext = " ".join(outtext.split("  "))
    return outtext


def space_tokenise(string, puncfilter=False, puncexcept=[]):
    """Takes a string, a true/false value for filtering punctuation and a list of punctuation-filter exceptions.
       Replaces newlines with spaces, tokenises string based on word spacing.
       If filter punctuation is true, each token has """
    strunaltered = string
    stroneline = remove_newlines(strunaltered)
    strtoklist = stroneline.split(" ")
    if puncfilter:
        newtoklist = []
        for tok in strtoklist:
            tok = rempunc_tok(tok, puncexcept)
            newtoklist.append(tok)
        return newtoklist
    else:
        return strtoklist


def remove_duptoks(toklist):
    """Creates a list of unique tokens which occur in the full list of tokens"""
    uniquetoks = []
    for tok in toklist:
        if tok not in uniquetoks:
            uniquetoks.append(tok)
    return uniquetoks


# Defines a function which gets the first element in a list.
def takefirst(elem):
    return elem[0]


def top_toks(string, occurrences=0):
    """Takes a string, and optional minimum number of occurrences, tokenises the string, removes blank tokens or tokens
       with punctuation only, returns list of sublists of tokens and use-count for that token in the string"""
    strtoks = space_tokenise(string, True)
    unitoks = remove_duptoks(strtoks)
    tokinfolist = []
    # Creates a list of a token's count, and the token itself, and adds this list to tokinfolist.
    for tok in unitoks:
        if tok != "":
            tokcount = strtoks.count(tok)
            tokinfo = [tokcount, tok]
            tokinfolist.append(tokinfo)
    # Sorts tokinfolist into descending order.
    tokinfolist.sort(key=takefirst, reverse=True)
    toknum = 0
    orderedtoklist = []
    # Counts, then lists, in descending order, tokens with more than a set number of occurrences in the string.
    for i in tokinfolist:
        if i[0] >= occurrences:
            toknum += 1
            orderedtoklist.append([toknum, i[0], i[1]])
    return orderedtoklist


# # Define the Models
# forward_folder = "Tokenisation Forward Models"
# celt_folder = op.join(forward_folder, "CELT Models")
# tbf_folder = op.join(forward_folder, "TBF Models")
# wb_folder = op.join(forward_folder, "Wb. Models")
# reverse_folder = "Tokenisation Reverse Models"
#
# mod1 = op.join(wb_folder, "n3_1HLTokeniser.h5")
# mod2 = op.join(wb_folder, "n3_2HLTokeniser.h5")
# mod3 = op.join(wb_folder, "n3pad_2HLTokeniser.h5")
# mod4 = op.join(wb_folder, "n3pad_2HLTokeniserV2.h5")
# mod5 = op.join(wb_folder, "n5_1HLTokeniser.h5")
# mod6 = op.join(wb_folder, "n5_2HLTokeniser.h5")
# mod7 = op.join(wb_folder, "n5_40x40-24.h5")
# mod8 = op.join(wb_folder, "n5_54x54-24.h5")
# mod9 = op.join(wb_folder, "n7_40x40-24.h5")
# mod10 = op.join(wb_folder, "n7_54x54-8-Wb-bi.h5")
# mod11 = op.join(wb_folder, "n7_54x54-24.h5")
# mod12 = op.join(wb_folder, "n10_2HLTokeniser.h5")
# mod13 = op.join(wb_folder, "n10_40x40-24.h5")
# mod14 = op.join(wb_folder, "n10_54x54-24.h5")
#
# CELTmod1 = op.join(celt_folder, "n5_54x54-6-CELT-Collection.h5")
# CELTmod2 = op.join(celt_folder, "n5_54x54-7-CELT-Collection.h5")
# CELTmod3 = op.join(celt_folder, "n5_54x54-24-CELT-Collection.h5")
# CELTmod4 = op.join(celt_folder, "n7_54x54-7-CELT-Collection.h5")
# CELTmod5 = op.join(celt_folder, "n7_54x54-8-CELT-Collection-bi.h5")
# CELTmod6 = op.join(celt_folder, "n7_54x54-24-CELT-Collection.h5")
# CELTmod7 = op.join(celt_folder, "n10_54x54-7-CELT-Collection.h5")
# CELTmod8 = op.join(celt_folder, "n10_54x54-24-CELT-Collection.h5")
#
# TBFmod1 = op.join(tbf_folder, "n3_TBF1HLTokeniser.h5")
# TBFmod2 = op.join(tbf_folder, "n3_TBF2HLTokeniser.h5")
# TBFmod3 = op.join(tbf_folder, "n3_TBF4HLTokeniser.h5")
# TBFmod4 = op.join(tbf_folder, "n5_TBF1HLTokeniser.h5")
# TBFmod5 = op.join(tbf_folder, "n5_TBF2HLTokeniser.h5")
# TBFmod6 = op.join(tbf_folder, "n5_TBF4HLTokeniser.h5")
# TBFmod7 = op.join(tbf_folder, "n5_TBF3HLTokeniser.h5")
# TBFmod8 = op.join(tbf_folder, "n5_TBF2HLTokeniserV2.h5")
# TBFmod9 = op.join(tbf_folder, "n5_TBF3HLTokeniserV2.h5")
# TBFmod10 = op.join(tbf_folder, "n5_TBF1HLTokeniserV2.h5")
# TBFmod11 = op.join(tbf_folder, "n7_TBF1HLTokeniser.h5")
# TBFmod12 = op.join(tbf_folder, "n3_TBF1HLTokeniserV2.h5")
#
# rmod1 = op.join(reverse_folder, "rev-n5_54x54-24.h5")
# rmod2 = op.join(reverse_folder, "rev-n7_54x54-24.h5")
# rmod3 = op.join(reverse_folder, "rev-n10_54x54-24.h5")
# rmod4 = op.join(reverse_folder, "rev-n7_54x54-8-Wb-bi.h5")
# rmod5 = op.join(reverse_folder, "rev-n5_54x54-7-CELT-Collection.h5")
# rmod6 = op.join(reverse_folder, "rev-n7_54x54-7-CELT-Collection.h5")
# rmod7 = op.join(reverse_folder, "rev-n10_54x54-7-CELT-Collection.h5")
# rmod8 = op.join(reverse_folder, "rev-n7_54x54-8-CELT-Collection-bi.h5")


# print(tokenise(mod1, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print(tokenise(mod2, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print(tokenise(mod3, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print(tokenise(mod4, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print(tokenise(mod5, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print(tokenise(mod6, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print(tokenise(mod7, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print(tokenise(mod8, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print(tokenise(mod9, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print(tokenise(mod10, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print(tokenise(mod11, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print(tokenise(mod12, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print(tokenise(mod13, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print(tokenise(mod14, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print()
# print(tokenise(CELTmod1, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print(tokenise(CELTmod2, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print(tokenise(CELTmod3, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print(tokenise(CELTmod4, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print(tokenise(CELTmod5, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print(tokenise(CELTmod6, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print(tokenise(CELTmod7, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print(tokenise(CELTmod8, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print()
# print(tokenise(TBFmod1, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print(tokenise(TBFmod2, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print(tokenise(TBFmod3, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print(tokenise(TBFmod4, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print(tokenise(TBFmod5, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print(tokenise(TBFmod6, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print(tokenise(TBFmod7, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print(tokenise(TBFmod8, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print(tokenise(TBFmod9, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print(tokenise(TBFmod10, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print(tokenise(TBFmod11, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print(tokenise(TBFmod12, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print()
# print(rev_tokenise(rmod1, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print(rev_tokenise(rmod2, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print(rev_tokenise(rmod3, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print(rev_tokenise(rmod4, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print(rev_tokenise(rmod5, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print(rev_tokenise(rmod6, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print(rev_tokenise(rmod7, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print(rev_tokenise(rmod8, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print()
# print(tokenise_combine(mod14, rmod2, ".i. biuusa ocirbáig darfarcennsi frimaccidóndu"))
# print()


# # Tokenise the set of Test Glosses
# formod = op.join(wb_folder, "n7_54x54-8-Wb-bi.h5")
# revmod = op.join(reverse_folder, "rev-n7_54x54-8-Wb-bi.h5")
#
# rawglosses = get_text("Untokenised Glosses")
# cleanglosslist = []
# for gloss in rawglosses.split("\n"):
#     idpat = re.compile(r'\(\d\d?\w \d\d?\w?\) ')
#     idpatitir = idpat.finditer(gloss)
#     for idfound in idpatitir:
#         glossid = idfound.group()
#         glosstext = gloss[len(glossid):]
#         cleanglosslist.append(glosstext)
# cleanglosses = " ".join(cleanglosslist)
# # Print the Test Glosses
# print(cleanglosses)
# # Print the Tokenised Test Glosses
# print(tokenise_combine(formod, revmod, cleanglosses))


# testlists = pickle.load(open('toktest.pkl', 'rb'))
# x_test = testlists[0]
# y_test = testlists[1]
# for text_no in range(len(x_test)):
#     print(x_test[text_no])
#     print(tokenise(mod1, x_test[text_no]))
#     print(y_test[text_no] + "\n")


# glosses = remove_glossnums(remove_brackets(
#     order_glosses(clear_tags("\n".join(get_section(get_pages("Wurzburg Glosses", 499, 712), "SG"))))))
# glosses = remove_glossnums(remove_brackets(rem_lat(order_glosses(clear_tags(
#     "\n".join(get_section(get_pages("Wurzburg Glosses", 499, 712), "SG")), ["GLat", "ie", "vel"])), True)))
# glosses = remove_glossnums(remove_brackets(rem_lat(clear_tags(
#     get_text("Wb. Prima Manus"), ["GLat", "ie", "vel"]), True)))
# glosses = remove_glossnums(remove_brackets(rem_lat(clear_tags(
#     get_text("Wb. Hand Two"), ["GLat", "ie", "vel"]), True)))
# glosses = remove_glossnums(remove_brackets(rem_lat(clear_tags(
#     get_text("Wb. Hand Three"), ["GLat", "ie", "vel"]), True)))


# # Prints the most common tokens.
# for token in top_toks(glosses, 53):
#     print(str(token[0]) + ". " + token[2] + " - Count: " + str(token[1]))

# # Saves an excel document of the most common tokens.
# save_xlsx("Wb. Common Tokens", top_toks(glosses, 53))
