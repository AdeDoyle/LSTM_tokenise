"""Level 1"""

from functools import lru_cache
from OpenPages import get_pages
from GetSections import get_section
from ClearTags import clear_tags
from OrderGlosses import order_glosses
from RemoveBrackets import remove_brackets
import re


@lru_cache(maxsize=1000)
def remove_glossnums(file):
    filetext = file
    """Removes gloss numbers from a string of glosses"""
    numpat = re.compile(r'(\d{1,2}[a-z]?, )?\d{1,2}[a-z]?\. ')
    numitir = numpat.finditer(filetext)
    for itir in numitir:
        num = itir.group()
        numlen = len(num)
        filetext = filetext[:filetext.find(num)] + filetext[filetext.find(num) + numlen:]
    return filetext


# glosses = remove_brackets(
#     order_glosses(clear_tags("\n".join(get_section(get_pages("Wurzburg Glosses", 499, 509), "SG")))))
# glosses = order_glosses("\n".join(get_section(get_pages("Wurzburg Glosses", 499, 509), "SG")))
# print(remove_glossnums(glosses))
