"""Level 1"""

from OpenPages import get_pages
import re


def clear_tags(file, exceptions=[]):
    """Clears all tags out of the text except those listed as exceptions, replacing them with original TPH markers where
       they were originally identified by use of various bracket types"""
    filetext = file
    taglist = ["H1", "H2", "Lat", "SG", "Eng", "FN", "GLat", "fol", "NV", "num", "let", "Rep", "ie", "vel", "etc",
               "Com", "Con", "Sup", "Res", "STOP", "Nam", "MRep", "LRep", "...", "&"]
    alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
                          "t", "u", "v", "w", "x", "y", "z"]
    for tag in taglist:
        # Ensures tag to be removed isn't to be excepted
        if tag not in exceptions:
            # Deals with folio tags which identify folios as numbered in TPH
            # Removes them without replacement
            if tag == "fol":
                for i in range(1, 34 + 1):
                    for l in ["a", "b", "c", "d"]:
                        opentag = "[f. " + str(i) + l + "]"
                        closetag = "[/f. " + str(i) + l + "]"
                        if opentag in filetext:
                            filetextlist = filetext.split(opentag)
                            filetext = "".join(filetextlist)
                        if closetag in filetext:
                            filetextlist = filetext.split(closetag)
                            filetext = "".join(filetextlist)
            # Deals with number tags which identify gloss placement within the Latin text.
            # Removes them without replacement
            elif tag == "num":
                numtagpat = re.compile(r'\[/?\d{1,2}\w?(–\d{1,2}\w?)?\]')
                numtagpatitir = numtagpat.finditer(filetext)
                numtaglist = []
                for i in numtagpatitir:
                    if i.group() not in numtaglist:
                        numtaglist.append(i.group())
                for numtag in numtaglist:
                    filetextlist = filetext.split(numtag)
                    filetext = "".join(filetextlist)
            # Deals with letter tags which identify footnotes.
            # Removes them without replacement
            elif tag == "let":
                for l in alphabet:
                    opentag = "[" + l + "]"
                    closetag = "[/" + l + "]"
                    if opentag in filetext:
                        filetextlist = filetext.split(opentag)
                        filetext = "".join(filetextlist)
                    if closetag in filetext:
                        filetextlist = filetext.split(closetag)
                        filetext = "".join(filetextlist)
            # Deals with comment tags which identify commentary by the editors
            # Replaces full tags with the original square brackets used in TPH
            elif tag == "Com":
                opentag = "[" + tag + "]"
                closetag = "[/" + tag + "]"
                if opentag in filetext:
                    filetextlist = filetext.split(opentag)
                    filetext = "[".join(filetextlist)
                if closetag in filetext:
                    filetextlist = filetext.split(closetag)
                    filetext = "]".join(filetextlist)
            # Deals with supplied text tags which identify text supplied by the editors
            # Replaces full tags with the original square brackets used in TPH
            elif tag == "Sup":
                opentag = "[" + tag + "]"
                closetag = "[/" + tag + "]"
                if opentag in filetext:
                    filetextlist = filetext.split(opentag)
                    filetext = "[".join(filetextlist)
                if closetag in filetext:
                    filetextlist = filetext.split(closetag)
                    filetext = "]".join(filetextlist)
            # Deals with restored text tags which identify text restored by the editors
            # Replaces full tags with the original round brackets used in TPH
            elif tag == "Res":
                opentag = "[" + tag + "]"
                closetag = "[/" + tag + "]"
                if opentag in filetext:
                    filetextlist = filetext.split(opentag)
                    filetext = "(".join(filetextlist)
                if closetag in filetext:
                    filetextlist = filetext.split(closetag)
                    filetext = ")".join(filetextlist)
            # Deals with all other tags
            # Removes them without replacement
            else:
                opentag = "[" + tag + "]"
                closetag = "[/" + tag + "]"
                if opentag in filetext:
                    filetextlist = filetext.split(opentag)
                    filetext = "".join(filetextlist)
                if closetag in filetext:
                    filetextlist = filetext.split(closetag)
                    filetext = "".join(filetextlist)
    return filetext


def clear_spectags(file, taglist=[]):
    """Clears only the specified tag or tags from the text using the clear_tags function"""
    alltags = ["H1", "H2", "Lat", "SG", "Eng", "FN", "GLat", "fol", "NV", "num", "let", "Rep", "ie", "vel", "etc",
               "Com", "Con", "Sup", "Res", "STOP", "Nam", "MRep", "LRep", "...", "&"]
    excepted = []
    for exception in alltags:
        if exception not in taglist:
            excepted.append(exception)
    untaggedtext = clear_tags(file, excepted)
    return untaggedtext


# glosses = get_pages("Wurzburg Glosses", 499, 509)

# print(clear_tags(glosses, ["Lat", "SG", "Eng", "FN"]))
# print(clear_tags(glosses))

# print(clear_spectags(glosses, ["Lat", "SG", "Eng", "FN"]))
# print(clear_spectags(glosses))
