"""Level 1"""

from functools import lru_cache
from OpenPages import get_pages
from ClearTags import clear_tags
from GetSections import get_section
from OrderGlosses import order_glosses


@lru_cache(maxsize=250)
def remove_brackets(file):
    """Removes all square and round brackets in the text. If square brackets enclose marginal information the contents
       of the brackets are removed also"""
    filetext = file
    if "(" in filetext:
        filetextlist = filetext.split("(")
        filetext = "".join(filetextlist)
    if ")" in filetext:
        filetextlist = filetext.split(")")
        filetext = "".join(filetextlist)
    if "[" in filetext:
        sbcount = filetext.count("[")
        for i in range(sbcount):
            sbopos = filetext.find("[")
            sbcpos = filetext.find("]")
            sbtext = filetext[sbopos + 1: sbcpos]
            if "marg." in sbtext:
                filetext = filetext[:sbopos] + filetext[sbcpos + 1:]
            elif "between ff." in sbtext:
                filetext = filetext[:sbopos] + filetext[sbcpos + 1:]
            elif "over v." in sbtext:
                filetext = filetext[:sbopos] + filetext[sbcpos + 1:]
            else:
                filetext = filetext[:sbcpos] + filetext[sbcpos + 1:]
                filetext = filetext[:sbopos] + filetext[sbopos + 1:]
    if "  " in filetext:
        spacecount = filetext.count("  ")
        for space in range(spacecount):
            spaceplace = filetext.find("  ")
            filetext = filetext[:spaceplace] + filetext[spaceplace + 1:]
    return filetext


# glosses = clear_tags("\n\n".join(get_section(get_pages("Wurzburg Glosses", 499, 509), "SG")))
# glosses = order_glosses(clear_tags("\n".join(get_section(get_pages("Wurzburg Glosses", 499, 509), "SG"))))
# print(remove_brackets(glosses))
