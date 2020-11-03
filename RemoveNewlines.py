"""Level 1"""

from functools import lru_cache
from OpenPages import get_pages
from GetSections import get_section
from ClearTags import clear_tags
from OrderGlosses import order_glosses
from RemoveBrackets import remove_brackets
from RemoveGlossnums import remove_glossnums


@lru_cache(maxsize=250)
def remove_newlines(file):
    """Takes a string, replaces all newlines with spaces, removes all double spaces"""
    filetext = file
    filelist = filetext.split("\n")
    filetext = " ".join(filelist)
    if "  " in filetext:
        filelist = filetext.split("  ")
        filetext = " ".join(filelist)
    return filetext


# glosses = remove_glossnums(remove_brackets(
#     order_glosses(clear_tags("\n".join(get_section(get_pages("Wurzburg Glosses", 499, 712), "SG"))))))
# print(remove_newlines(glosses))
