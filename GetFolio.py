"""Level 1"""

from functools import lru_cache
from OpenPages import get_pages
from GetSections import get_section
from OrderGlosses import order_glosses
from ClearTags import clear_tags
import re


@lru_cache(maxsize=1000)
def get_fol(text):
    """Takes a text returns a folio-list of tag-lists with text at tag-list[0] and folio info at tag-list[1]"""
    fulltext = text
    otags = []
    ctags = []
    follist = []
    opat = re.compile(r'\[f\. \d?\d[a-d]\]')
    cpat = re.compile(r'\[/f\. \d?\d[a-d]\]')
    for i in opat.finditer(fulltext):
        otags.append(i.group(0))
    for i in cpat.finditer(fulltext):
        ctags.append(i.group(0))
    tagstext = fulltext
    for i in range(len(otags)):
        otag = otags[i]
        ctag = ctags[i]
        tag = otag[1:-1]
        tagtext = tagstext[tagstext.find(otag) + len(otag):tagstext.find(ctag)]
        tagstext = tagstext[tagstext.find(ctag) + len(ctag):]
        follist.append([tagtext, tag])
    return follist


# glosses = get_pages("Wurzburg Glosses", 499, 509)
# glosses = order_glosses("\n\n".join(get_section(get_pages("Wurzburg Glosses", 499, 509), "SG")))
# glosses = order_glosses(clear_tags("\n\n".join(get_section(get_pages("Wurzburg Glosses", 499, 509), "SG")), "fol"))
# for i in get_fol(glosses):
#     print(i[1] + ":\n" + i[0] + "\n")
