"""Level 1."""

from functools import lru_cache
from OpenPages import get_pages


@lru_cache(maxsize=1000)
def get_tagtext(fulltext, tag):
    """Gets just the text between a particular set of tags from a string, returns a list of each instance"""
    taglist = []
    tagtext = fulltext
    if tag[0] == "[":
        tag = tag[1:]
    if tag[0] == "/":
        tag = tag[1:]
    if tag[-1] == "]":
        tag = tag[:-1]
    opentag = "[" + tag + "]"
    closetag = "[/" + tag + "]"
    if opentag in tagtext:
        occurrences = tagtext.count(opentag)
        for i in range(occurrences):
            tagtext = tagtext[tagtext.find(opentag):]
            taggedtext = tagtext[+ len(opentag):tagtext.find(closetag)]
            tagtext = tagtext[tagtext.find(closetag) + len(closetag):]
            taglist.append(taggedtext)
    return taglist


# glosses = get_pages("Wurzburg Glosses", 499, 499)
# glosses = "AAAAAAAAAAAAAAAA[FN]BBB[/FN]AAAAAAAAAAAAAAA[FN]CCC[/FN]AAAAAAAAA[FN]DDD[/FN]AAAAA"
# for item in get_tagtext(glosses, "FN"):
#     print(item)
