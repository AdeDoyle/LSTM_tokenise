"""Level 1"""

from functools import lru_cache
from ClearTags import clear_tags
from GetSections import get_section
from OpenPages import get_pages
import re


@lru_cache(maxsize=250)
def order_glosslist(file):
    """Finds each gloss by its number and returns a list of glosses"""
    glosstext = file
    theseglosses = []
    glossitirs = []
    folglosspat = re.compile(r'(\[/?f\. \d{1,2}[a-d]\])?(\d{1,2}[a-z]?, )?\d{1,2}[a-z]?\. ')
    glossitir = folglosspat.finditer(glosstext)
    for i in glossitir:
        glossitirs.append(i.start())
    for i in glossitirs:
        if i == glossitirs[0]:
            startpoint = i
        elif i == glossitirs[-1]:
            endpoint = i
            thisgloss = glosstext[startpoint:endpoint]
            thisgloss = thisgloss.strip()
            if "\n" in thisgloss:
                thisglosslist = thisgloss.split("\n")
                thisgloss = " ".join(thisglosslist)
            theseglosses.append(thisgloss)
            startpoint = endpoint
            lastgloss = glosstext[startpoint:]
            lastgloss = lastgloss.strip()
            if "\n" in lastgloss:
                lastglosslist = lastgloss.split("\n")
                lastgloss = " ".join(lastglosslist)
            theseglosses.append(lastgloss)
        else:
            endpoint = i
            thisgloss = glosstext[startpoint:endpoint]
            thisgloss = thisgloss.strip()
            if "\n" in thisgloss:
                thisglosslist = thisgloss.split("\n")
                thisgloss = " ".join(thisglosslist)
            theseglosses.append(thisgloss)
            startpoint = endpoint
    # """This code counts the total number of glosses, then prints them by number"""
    # glosscount = 0
    # for i in theseglosses:
    #     glosscount += 1
    #     print(str(glosscount) + ": " + i)
    return theseglosses


@lru_cache(maxsize=250)
def order_glosses(file):
    """Finds each gloss by its number and returns a string with each gloss on a new line"""
    glosslist = order_glosslist(file)
    glossesstring = "\n".join(glosslist)
    return glossesstring


# glosses = clear_tags("\n\n".join(get_section(get_pages("Wurzburg Glosses", 558, 558), "SG")))
# glosses = "\n\n".join(get_section(get_pages("Wurzburg Glosses", 499, 558), "SG"))
# print(order_glosses(glosses))
# for test in order_glosslist(glosses):
#     print(test)
