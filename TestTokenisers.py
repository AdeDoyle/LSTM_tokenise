import pickle
from functools import lru_cache
from Old_Tokenise import tokenise as tz, rev_tokenise as rtz, tokenise_combine as tzc
from nltk import edit_distance as ed


# mod1 = "n3_1HLTokeniser.h5"
# mod2 = "n3_2HLTokeniser.h5"
# mod3 = "n10_2HLTokeniser.h5"
# mod4 = "n3pad_2HLTokeniser.h5"
# mod4_200 = "n3pad_2HLTokeniserV2.h5"
# mod5 = "n5_1HLTokeniser.h5"
# mod6 = "n5_2HLTokeniser.h5"
#
# TBFmod1 = "n3_TBF1HLTokeniser.h5"
# TBFmod2 = "n3_TBF2HLTokeniser.h5"
# TBFmod3 = "n3_TBF4HLTokeniser.h5"
# TBFmod4 = "n5_TBF1HLTokeniser.h5"
# TBFmod5 = "n5_TBF2HLTokeniser.h5"
# TBFmod6 = "n5_TBF4HLTokeniser.h5"
# TBFmod7 = "n5_TBF3HLTokeniser.h5"
# TBFmod8 = "n5_TBF2HLTokeniserV2.h5"
# TBFmod9 = "n5_TBF3HLTokeniserV2.h5"
# TBFmod10 = "n5_TBF1HLTokeniserV2.h5"
# TBFmod11 = "n7_TBF1HLTokeniser.h5"
# TBFmod12 = "n3_TBF1HLTokeniserV2.h5"


# ogmods = [mod1, mod2, mod3, mod4, mod4_200, mod5, mod6]
# tbfmods = [TBFmod1, TBFmod2, TBFmod3, TBFmod4, TBFmod5, TBFmod6,
#            TBFmod7, TBFmod8, TBFmod9, TBFmod10, TBFmod11, TBFmod12]
# allmods = ogmods + tbfmods


# allmods = []
# for layer in [1, 2, 3]:
#     for size in [41, 54]:
#         for dense in [0, 1]:
#             NAME = "TBF {}-LSTM-{}-Nodes-{}-Dense".format(layer, size, dense)
#             name = "TBF-24 {}-LSTM-{}-Nodes-{}-Dense".format(layer, size, dense)
#             allmods.append(NAME)
#             allmods.append(name)


allmods = []
for size in [54]:
    for buff in [7]:
        NAME = "n{}_{}x{}-8-Wb-bi.h5".format(buff, size, size)
        allmods.append(NAME)

allrmods = []
for size in [54]:
    for buff in [7]:
        NAME = "rev-n{}_{}x{}-8-Wb-bi.h5".format(buff, size, size)
        allrmods.append(NAME)

test_in = open("toktest.pkl", "rb")
test_set = pickle.load(test_in)
x_test, y_test = test_set[0], test_set[1]


@lru_cache(maxsize=50)
def clean_y(y_string):
    remove = ["(", ")", "*", "-", "^"]
    for rem in remove:
        if rem in y_string:
            y_string = "".join(y_string.split(rem))
    if "Latin" in y_string:
        y_string = "*Latin*".join(y_string.split("Latin"))
    return y_string


def test_tzmod(mod):
    edit_dists = []
    count = 0
    for x_pos in range(len(x_test)):
        count += 1
        x = x_test[x_pos]
        y = clean_y(y_test[x_pos])
        x_toks = tz(mod, x)
        e_dist = ed(y, x_toks)
        edit_dists.append(e_dist)
        # print(x)
        # print(x_toks)
        # print(y)
        # print("Gloss {}/41: Edit Distance = {}".format(str(count), str(e_dist)))
    avg_edist = sum(edit_dists) / len(edit_dists)
    return avg_edist


def test_rtzmod(mod):
    edit_dists = []
    count = 0
    for x_pos in range(len(x_test)):
        count += 1
        x = x_test[x_pos]
        y = clean_y(y_test[x_pos])
        x_toks = rtz(mod, x)
        e_dist = ed(y, x_toks)
        edit_dists.append(e_dist)
        # print(x)
        # print(x_toks)
        # print(y)
        # print("Gloss {}/41: Edit Distance = {}".format(str(count), str(e_dist)))
    avg_edist = sum(edit_dists) / len(edit_dists)
    return avg_edist


def test_tzcmod(mod, rmod):
    edit_dists = []
    count = 0
    for x_pos in range(len(x_test)):
        count += 1
        x = x_test[x_pos]
        y = clean_y(y_test[x_pos])
        x_toks = tzc(mod, rmod, x)
        e_dist = ed(y, x_toks)
        edit_dists.append(e_dist)
        # print(x)
        # print(x_toks)
        # print(y)
        # print("Gloss {}/41: Edit Distance = {}".format(str(count), str(e_dist)))
    avg_edist = sum(edit_dists) / len(edit_dists)
    return avg_edist


# # Test Manually Tokenised Glosses against Untokenised Glosses
# all_ed_dists = []
# gl_count = 0
# for x_pos in range(len(x_test)):
#     gl_count += 1
#     x = x_test[x_pos]
#     y = clean_y(y_test[x_pos])
#     ed_dist = ed(y, x)
#     all_ed_dists.append(ed_dist)
#     # print(x)
#     # print(y)
#     # print("Gloss {}/41: Edit Distance = {}".format(str(gl_count), str(ed_dist)))
# avg_ed_dist = sum(all_ed_dists) / len(all_ed_dists)
# print("Original Gloss Score:\n    {}".format(avg_ed_dist))


# # Test Forward Models
# modscores = []
# for mod in allmods:
#     print(mod)
#     score = test_tzmod(mod)
#     print("    {}".format(score))
#     modscores.append(score)
# best_score = min(modscores)
# print("Best Model:\n    {}".format(allmods[modscores.index(best_score)]))

# # Test Reverse Models
# modscores = []
# for mod in allrmods:
#     print(mod)
#     score = test_rtzmod(mod)
#     print("    {}".format(score))
#     modscores.append(score)
# best_score = min(modscores)
# print("Best Model:\n    {}".format(allrmods[modscores.index(best_score)]))

# # Test Combined Forward and Reverse Models
# modscores = []
# for mod in allmods:
#     for rmod in allrmods:
#         print("Forward model: {}\nReverse model: {}".format(mod, rmod))
#         score = test_tzcmod(mod, rmod)
#         print("    {}".format(score))
#         modscores.append(score)
# best_score = min(modscores)
# print("Best Model:\n    {}\n    {}".format(allmods[modscores.index(best_score)], allrmods[modscores.index(best_score)]))

