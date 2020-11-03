"""Level 2, 2, 2, 1, 1, 1, 1, 2, 2, 1"""

from OpenDocx import get_text
from functools import lru_cache
from OpenPages import get_pages
from GetSections import get_section
from OrderGlosses import order_glosses, order_glosslist
from GetFolio import get_fol
from RemoveLatin import rep_lat
from ClearTags import clear_tags
from RemoveBrackets import remove_brackets
import re
from Old_Tokenise import remove_duptoks
from SaveDocx import save_docx
import pickle


def openhandlists(file):
    """Gets the text from a gloss-hand file"""
    filetext = get_text(file)
    return filetext


@lru_cache(maxsize=10)
def splitglosses(file):
    """Splits the glosses from a gloss-hand file into a gloss list"""
    filetext = openhandlists(file)
    glosslist = order_glosslist(filetext)
    return glosslist


def cleangloss(gloss):
    """Takes a gloss, replaces Latin with a place-marker, removes tags, removes brackets and non-gloss bracket content,
       then removes gloss numbers and all unnecessary punctuation"""
    cleanedgloss = remove_brackets(clear_tags(rep_lat(gloss, "*Latin*")))
    # remove repeated latin markers without Irish text between them
    if cleanedgloss.count("*Latin*") > 1:
        latcleanpat = re.compile(r'\*Latin\*[^a-zA-Záéíóúæǽḟṁṅↄ]*\*Latin\*([^a-zA-Záéíóúæǽḟṁṅↄ]*\*Latin\*){0,10}')
        latcleanpatiter = latcleanpat.finditer(cleanedgloss)
        for patfind in latcleanpatiter:
            cleanedgloss = "*Latin*".join(cleanedgloss.split(patfind.group()))
    # Removes gloss numbers
    numpat = re.compile(r'(\d{1,2}[a-z]?, )?\d{1,2}[a-z]?\. ')
    numpatitir = numpat.finditer(cleanedgloss)
    for i in numpatitir:
        if i.group() in cleanedgloss:
            glosssplit = cleanedgloss.split(i.group())
            cleanedgloss = "".join(glosssplit)
    # Removes all known, undesired punctuation except full stops
    for punct in [",", " . ", ":", ";", ".........", ".......", "......", ".....", "....", "...", "  "]:
        if punct in cleanedgloss:
            while punct in cleanedgloss:
                glosssplit = cleanedgloss.split(punct)
                cleanedgloss = " ".join(glosssplit)
    # Removes most unrequired full stops
    stoppat = re.compile(r'[^ilɫró]\.[^\w]')
    stoppatitir = stoppat.finditer(cleanedgloss)
    for i in stoppatitir:
        # Remainder (below) necessary to ensure only full stop is removed, and not all of i.group()
        isplit = (i.group().strip()).split(".")
        # Ensure no space is removed after "*"
        if "*" in isplit:
            remainder = " ".join(isplit)
        else:
            remainder = ("".join(isplit))
        if i.group() in cleanedgloss:
            glosssplit = cleanedgloss.split(i.group())
            cleanedgloss = remainder.join(glosssplit)
    # Replaces doubled Latin markers with just one
    if "*Latin* *Latin*" in cleanedgloss:
        while "*Latin* *Latin*" in cleanedgloss:
            glosssplit = cleanedgloss.split("*Latin* *Latin*")
            cleanedgloss = "*Latin*".join(glosssplit)
    # Removes full stops at the end of glosses as long as they're not part of "rl.", ".i." or "ɫ."
    if cleanedgloss[-1] == ".":
        patfound = False
        endpat = re.compile(r'(rl|\.i|ɫ)\.')
        endpatitir = endpat.finditer(cleanedgloss)
        for i in endpatitir:
            ilen = len(i.group())
            if i.group() == cleanedgloss[-ilen:]:
                patfound = True
        if not patfound:
            cleanedgloss = cleanedgloss[:-1]
    if cleanedgloss[:3] != ".i.":
        if cleanedgloss[0] == ".":
            cleanedgloss = cleanedgloss[1:]
    cleanedgloss = cleanedgloss.strip()
    return cleanedgloss


def tokenisegloss(gloss):
    glosstoks = gloss.split(" ")
    return glosstoks


def remove_non_glosses(gloss_list):
    """Removes glosses from a list of pre-cleaned if they are comprised only of non-Old-Irish tokens"""
    new_glosslist = []
    faultlist = [".i.", "rl.", "ɫ.", "*Latin*", ""]
    faultcount = 0
    for gloss in gloss_list:
        if gloss not in faultlist:
            glossfault = True
            for glosstok in tokenisegloss(gloss):
                if glosstok not in faultlist:
                    glossfault = False
                    break
            if glossfault:
                faultcount += 1
            else:
                new_glosslist.append(gloss)
        else:
            faultcount += 1
    # print(faultcount)
    return new_glosslist


def compile_tokenised_glosslist(file):
    """Takes a file, removes glosses which contain no Irish tokens,
       counts faults experienced along the way,
       returns a tokenised list of lists (glosses and their tokens) for each acceptable gloss"""
    tokenised_handlist = []
    # removes glosses just comprised of non-Old-Irish tokens
    faultlist = [".i.", "rl.", "ɫ.", "*Latin*", ""]
    faultcount = 0
    for gloss in splitglosses(file):
        if cleangloss(gloss) not in faultlist:
            glossfault = True
            for glosstok in tokenisegloss(cleangloss(gloss)):
                if glosstok not in faultlist:
                    glossfault = False
                    break
            if glossfault:
                faultcount += 1
            else:
                tokenised_handlist.append(tokenisegloss(cleangloss(gloss)))
        else:
            faultcount += 1
    # print(faultcount)
    return tokenised_handlist


def create_test_training(file):
    """Takes a list of tokenised glosses for a given hand
       Divides the glosses into two lists, test and training (10% and 90% of glosses respectively)"""
    fileglosses = compile_tokenised_glosslist(file)
    training = []  # 90% of glosses in input file
    test = []  # 10% of glosses in input file
    for i in range(len(fileglosses)):
        if (i + 1) % 10 == 0:
            test.append(" ".join(fileglosses[i]))
        else:
            training.append(" ".join(fileglosses[i]))
    return "\n".join(test), "\n".join(training)


def split_testtrain(file):
    """Splits the glosses from a test/train file into a gloss list"""
    filetext = openhandlists(file)
    glosslist = filetext.split("\n")
    return glosslist


def compile_tokenised_testtrain(file):
    """Returns a tokenised list of lists (glosses and their tokens) for each test/training gloss"""
    tok_list = []
    for gloss in split_testtrain(file):
        glosstoks = gloss.split(" ")
        tok_list.append(glosstoks)
    return tok_list


def combinelists(listlist):
    """Combines lists of lists into a single list of the
       contents of all the lists in the list-of-lists entered"""
    outlist = []
    for i in range(len(listlist)):
        singlelist = listlist[i]
        for j in range(len(singlelist)):
            outlist.append(singlelist[j])
    return outlist


def get_unitoks(glosslist):
    """Returns a list of unique tokens in a list of tokenised glosses"""
    return remove_duptoks(combinelists(glosslist))


def get_unitokscount(glosslist):
    """Returns a count of all unique tokens in a list of tokenised glosses"""
    unitoks = remove_duptoks(combinelists(glosslist))
    tokcount = len(unitoks)
    return tokcount


def get_inditokcount(tok, glosslist):
    """Returns a count of a single token's number of occurrences in a list of tokenised glosses"""
    alltoks = combinelists(glosslist)
    if tok in alltoks:
        return alltoks.count(tok)
    elif tok not in alltoks:
        return 0


def list_numbered_glosses(file, startpage, stoppage):
    """Lists glosses by their folio ID and gloss number"""
    glist = []
    for p in range(startpage, stoppage + 1):
        fcont = get_fol(order_glosses(clear_tags("\n\n".join(get_section(get_pages(file, p, p), "SG")), "fol")))
        for g in order_glosslist("\n\n".join(get_section(get_pages(file, p, p), "SG"))):
            for fol in fcont:
                raw_gloss = clear_tags(g)
                if clear_tags(g) in fol[0]:
                    numpat = re.compile(r'(\d{1,2}[a-z]?, )?\d{1,2}[a-z]?\. ')
                    numpatitir = numpat.finditer(raw_gloss)
                    for i in numpatitir:
                        if i.group() in raw_gloss:
                            glist.append([fol[1][3:] + i.group()[:-1], cleangloss(g)])
    return glist


def create_tokeniser_test_training():
    """Splits the glosses into two lists, one for training a character-level LSTM based tokeniser, and another
       containing 41 pre-selected glosses as a test set.
       The test set is split into two further lists, the first list is the untokenised glosses of the test set, the
       second is the same glosses, manually tokenised.
       Saves each list as a pickle file."""
    glist = list_numbered_glosses("Wurzburg Glosses", 499, 712)
    # List numbers of all chosen test-set glosses in order (19d20 could be added)
    testglossids = ["2c4.", "5b11.", "5b28.", "6c7.", "6c9.", "9a14.", "9b4.", "9c20.", "10b27.", "10c21.", "10d23.",
                    "10d36.", "11a24.", "12a22.", "12c9.", "12c29.", "12c32.", "12c36.", "14a8.", "14c2a.", "14c18.",
                    "14c23.", "14d17.", "14d26.", "15a18.", "16d8.", "17d27.", "18a14.", "18c6.", "19b6.", "21a8.",
                    # "19d20",
                    "21c19.", "23b7.", "23d10.", "26b6.", "27a24.", "28c2.", "28d16.", "29d19.", "30b4.", "31c7."]
    # Takes Manually Tokenised test-set glosses from file and adds them to a dictionary so they can be found using the
    # list above as keys
    man_tok_glosslist = get_text("Manually Tokenised Glosses").split("\n")
    mtgidpat = re.compile(r'\(\d{1,2}\w \d{1,2}\w?\) ')
    mtgs_with_ids = {}
    for mtg in man_tok_glosslist:
        mtgpatitir = mtgidpat.finditer(mtg)
        for mtgiditir in mtgpatitir:
            mtgloss = "".join(mtg.split(mtgiditir.group()))
            mtgid = "".join(mtg.split(mtgloss))
            mtgid_fix = "".join(mtgid.split(" ")) + "."
            mtgid_fix = "".join(mtgid_fix.split("("))
            mtgid_fix = "".join(mtgid_fix.split(")"))
            mtgs_with_ids[mtgid_fix] = mtgloss
    # Creates the test and training lists
    testglosses = []
    testglosses_tokenised = []
    trainglosses = []
    for g in glist:
        if g[0] in testglossids:
            testglosses.append(g[1])
            testglosses_tokenised.append(mtgs_with_ids.get(g[0]))
        else:
            trainglosses.append(g[1])
    # Combines the untokenised and tokenised test-set
    testglosses_set = [testglosses, testglosses_tokenised]
    # Saves the test and train sets to pickle files
    pickletest_out = open("toktest.pkl", "wb")
    pickle.dump(testglosses_set, pickletest_out)
    pickletest_out.close()
    pickletrain_out = open("toktrain.pkl", "wb")
    pickle.dump(trainglosses, pickletrain_out)
    pickletrain_out.close()
    return "\nTest and Training Sets Compiled for Gloss Tokenisation.\n"


# glosshands = ["Wb. All Glosses", "Wb. Prima Manus", "Wb. Hand Two", "Wb. Hand Three"]
# allglosstoks = compile_tokenised_glosslist(glosshands[0])
# pmtoks = compile_tokenised_glosslist(glosshands[1])
# h2toks = compile_tokenised_glosslist(glosshands[2])
# h3toks = compile_tokenised_glosslist(glosshands[3])
# pmtest = compile_tokenised_testtrain("Hand_1_hand_test")
# h2test = compile_tokenised_testtrain("Hand_2_hand_test")
# h3test = compile_tokenised_testtrain("Hand_3_hand_test")
# alltesttoks = combinelists([pmtest, h2test, h3test])
# pmtrain = compile_tokenised_testtrain("Hand_1_hand_training")
# h2train = compile_tokenised_testtrain("Hand_2_hand_training")
# h3train = compile_tokenised_testtrain("Hand_3_hand_training")
# alltraintoks = combinelists([pmtrain, h2train, h3train])


# print(openhandlists(glosshands[0]))

# for gloss in splitglosses(glosshands[0]):
#     print(gloss)

# for gloss in splitglosses(glosshands[0]):
#     print(cleangloss(gloss))

# cleaned_list = []
# for OIgloss in splitglosses("Wb. All Glosses"):
#     cleaned_list.append(cleangloss(OIgloss))
# for OIgloss in remove_non_glosses(cleaned_list):
#     print(OIgloss)

# for tok_gloss in compile_tokenised_glosslist(glosshands[0]):
#     print(tok_gloss)

# print(create_test_training(glosshands[2])[0])

# for i in range(1, 4):
#     infile = glosshands[i]
#     test_training = create_test_training(infile)
#     handname = "Hand_" + str(i)
#     save_docx(test_training[0], handname + "_hand_test")
#     save_docx(test_training[1], handname + "_hand_training")

# print(split_testtrain("Hand_1_hand_test"))

# print(compile_tokenised_testtrain("Hand_1_hand_test"))

# for i in compile_tokenised_testtrain("Hand_1_hand_test"):
#     print(i)

# for unitok in get_unitoks(pmtoks):
#     print(unitok)

# print(get_unitokscount(pmtoks))

# for tok in remove_duptoks(combinelists(pmtoks)):
#     print(tok + ": " + str(get_inditokcount(tok, pmtoks)))

# # Count how many glosses are in each gloss-list
# print(len(allglosstoks))
# print(len(pmtoks))
# print(len(h2toks))
# print(len(h3toks))

# # Count how many unique tokens are in each gloss-list
# print(len(remove_duptoks(combinelists(allglosstoks))))
# print(len(remove_duptoks(combinelists(pmtoks))))
# print(len(remove_duptoks(combinelists(h2toks))))
# print(len(remove_duptoks(combinelists(h3toks))))

# # Count how many tokens are in each gloss-list
# print(len(combinelists(allglosstoks)))
# print(len(combinelists(pmtoks)))
# print(len(combinelists(h2toks)))
# print(len(combinelists(h3toks)))

# for g in list_numbered_glosses("Wurzburg Glosses", 499, 712):
#     print(g)

# print(create_tokeniser_test_training())

# pickletest_in = open("toktest.pkl", "rb")
# testglosses_tokeniser = pickle.load(pickletest_in)
# pickletrain_in = open("toktrain.pkl", "rb")
# trainglosses_tokeniser = pickle.load(pickletrain_in)
# for testglosses_set in testglosses_tokeniser:
#     print(testglosses_set)
# print(trainglosses_tokeniser)

