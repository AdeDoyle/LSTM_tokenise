"""Level 2."""

from functools import lru_cache
from GetTagText import get_tagtext
from OpenPages import get_pages


@lru_cache(maxsize=1000)
def get_section(fulltext, section="No Section"):
    """Gets and returns list of instances of just the text between a particular set of section tags"""
    if section != "Lat":
        if section != "SG":
            if section != "Eng":
                if section != "FN":
                    section = "No Section"
    if section != "No Section":
        secs = get_tagtext(fulltext, section)
    else:
        secremovetext = fulltext
        seclist = ["Lat", "SG", "Eng", "FN"]
        for sec in seclist:
            opentag = ("[" + sec + "]")
            closetag = ("[/" + sec + "]")
            if opentag in secremovetext:
                remcount = secremovetext.count(opentag)
                for i in range(remcount):
                    secremovetext = secremovetext[:secremovetext.find(opentag)] + "***SPLITHERE***" + \
                                    secremovetext[secremovetext.find(closetag) + len(closetag):]
        secs = secremovetext.split("***SPLITHERE***")
        newsects = []
        for sect in secs:
            sect = sect.strip()
            newsects.append(sect)
        secs = newsects
        spaces = "\n\n\n\n"
        if spaces in "\n".join(secs):
            secs = "\n".join(secs)
            secs = secs.split(spaces)
        spaces = "\n\n\n"
        if spaces in "\n".join(secs):
            secs = "\n".join(secs)
            secs = secs.split(spaces)
    return secs


# glosses = get_pages("Wurzburg Glosses", 499, 712)
# print("\n\n".join(get_section(glosses, "SG")))
# print("\n".join(get_section(glosses)))
