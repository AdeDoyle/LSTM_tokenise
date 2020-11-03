"""Level 1."""

from functools import lru_cache
from OpenDocx import get_text


@lru_cache(maxsize=250)
def find_page(file, findpage):
    """Iterates through document from page 499-712 and finds the beginning point of a desired page.
       Returns position."""
    stringplace = 0
    if findpage < 499:
        findpage = 499
    if findpage > 712:
        findpage = 499
    if findpage == 499:
        return stringplace
    else:
        workstring = file
        curpage = 499
        for i in range(499, findpage + 1):
            if curpage < findpage:
                curplace = workstring.find(str(curpage))
                stringplace += curplace
                workstring = workstring[curplace:]
                curpage += 1
            elif curpage == findpage:
                tempstring = workstring
                tempplace = tempstring.find(str(curpage))
                tempstring = tempstring[:tempplace]
                tempstring = tempstring[:tempstring.rfind("\n")]
                curplace = len(tempstring) + 1
                stringplace += curplace
                workstring = workstring[curplace:]
        return stringplace


# glosses = get_text("Wurzburg Glosses")
# pagepoint = find_page(glosses, 509)
# print(pagepoint)
# print(glosses[pagepoint:])
