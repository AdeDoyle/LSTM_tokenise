"""Level 2."""

from functools import lru_cache
from OpenDocx import get_text
from IteratePages import find_page


@lru_cache(maxsize=250)
def get_pages(filename, startpage=499, endpage=712):
    """Opens the text of the Glosses from the document and returns the text of the selected page range"""
    alltext = get_text(filename)
    pagestext = []
    lastpage = 0
    nextpagepoint = 0
    startpoint = 0
    if startpage < 499:
        startpage = 499
    if endpage > 712:
        endpage = 712
    if startpage > 712:
        startpage = 499
    if endpage < 499:
        endpage = 712
    if startpage == 499 and endpage == 712:
        return alltext
    elif startpage > 499:
        alltext = alltext[find_page(alltext, startpage):]
    for page in range(startpage, endpage + 1):
        if lastpage == 0:
            pageno = startpage
        else:
            pageno = lastpage + 1
            startpoint += nextpagepoint
        if pageno == 712:
            pagetext = alltext[startpoint:]
        else:
            nextpage = pageno + 1
            pagetext = alltext[startpoint:]
            nextpagepoint = pagetext.find(str(nextpage))
            pagetext = pagetext[:nextpagepoint]
            nextpagepoint = pagetext.rfind("\n")
            pagetext = pagetext[:nextpagepoint]
        pagestext.append(pagetext)
        lastpage = pageno
    pagestext = "".join(pagestext)
    return pagestext


# print(get_pages("Wurzburg Glosses", 499, 509))
