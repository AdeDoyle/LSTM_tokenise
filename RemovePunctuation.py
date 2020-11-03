"""Level 1"""

from functools import lru_cache


@lru_cache(maxsize=9000)
def rempunc_tok(token, exceptions=[]):
    """takes a token and list of exceptions. If a punctuation type is not excepted, it is removed from the token if at
       the end and not preceded directly by another un-excepted punctuation type"""
    fixtok = token
    punclist = [".", ",", "!", "?", "…", ";", ":"]
    knownpuncs = ["ó.", ".r.", "rl.", "ɫ.", ".i."]
    if exceptions:
        punclist = [i for i in punclist if i not in exceptions]
    allpunc = True
    for char in fixtok:
        if char not in punclist:
            allpunc = False
    if allpunc:
        fixtok = ""
    elif fixtok in knownpuncs:
        return fixtok
    elif len(fixtok) > 1:
        for punc in punclist:
            if fixtok[-1] == punc:
                if fixtok[-2] not in punclist:
                    fixtok = fixtok[:-1]
    elif len(fixtok) == 1:
        for punc in punclist:
            if punc in fixtok:
                fixtok = fixtok[:-1]
    return fixtok


# testlist = ["abc", "def.", "ghi...", ";", ":", "jkl:", "mno...pqr", ".r.", "rl.", ".i.", "...", ".:"]

# print(rempunc_tok("abc."))
# for text in testlist:
#     print(rempunc_tok(text))
