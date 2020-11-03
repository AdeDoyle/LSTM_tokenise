"""Level 1"""

from functools import lru_cache
from RemoveNewlines import remove_newlines
from OrderGlosses import order_glosses, get_section, get_pages, clear_tags


@lru_cache(maxsize=250)
def rep_lat(glosstext, repchar="…", strong=False):
    """Takes glosses as input. Replaces all Latin text with ellipsis (or other selected replace-character.
       A strong-replace will treat common abbreviations like Latin and remove them too."""
    workingtext = glosstext
    latinsts = workingtext.count("[GLat]")
    if "[GLat]" in workingtext:
        if latinsts != workingtext.count("[/GLat]"):
            return "Error: Uneven number of open and close tags for Gloss Latin."
        else:
            for i in range(latinsts):
                workingtext = workingtext[:workingtext.find("[GLat]")] + repchar +\
                              workingtext[workingtext.find("[/GLat]") + 7:]
    if strong:
        ieinsts = workingtext.count("[ie]")
        velinsts = workingtext.count("[vel]")
        etcinsts = workingtext.count("[etc]")
        if "[ie]" in workingtext:
            if ieinsts != workingtext.count("[/ie]"):
                return "Error: Uneven number of open and close tags for .i."
            else:
                for j in range(ieinsts):
                    workingtext = workingtext[:workingtext.find("[ie]")] + repchar + \
                                  workingtext[workingtext.find("[/ie]") + 5:]
        if "[vel]" in workingtext:
            if velinsts != workingtext.count("[/vel]"):
                return "Error: Uneven number of open and close tags for ɫ."
            else:
                for k in range(velinsts):
                    workingtext = workingtext[:workingtext.find("[vel]")] + repchar + \
                                  workingtext[workingtext.find("[/vel]") + 6:]
        if "[etc]" in workingtext:
            if etcinsts != workingtext.count("[/etc]"):
                return "Error: Uneven number of open and close tags for rl."
            else:
                for l in range(etcinsts):
                    workingtext = workingtext[:workingtext.find("[etc]")] + repchar + \
                                  workingtext[workingtext.find("[/etc]") + 6:]
    if repchar in workingtext:
        reconstruct = "[GLat]" + repchar + "[/GLat]"
        workingtext = reconstruct.join(workingtext.split(repchar))
    return workingtext


@lru_cache(maxsize=250)
def rem_lat(glosstext, strong=False):
    """Takes glosses as input. Removes all Latin text and resultant double spacing."""
    workingtext = glosstext
    latinsts = workingtext.count("[GLat]")
    if "[GLat]" in workingtext:
        if latinsts != workingtext.count("[/GLat]"):
            return "Error: Uneven number of open and close tags for Gloss Latin!"
        else:
            for i in range(latinsts):
                workingtext = workingtext[:workingtext.find("[GLat]")] + workingtext[workingtext.find("[/GLat]") + 7:]
    if strong:
        ieinsts = workingtext.count("[ie]")
        velinsts = workingtext.count("[vel]")
        etcinsts = workingtext.count("[etc]")
        if "[ie]" in workingtext:
            if ieinsts != workingtext.count("[/ie]"):
                return "Error: Uneven number of open and close tags for .i."
            else:
                for j in range(ieinsts):
                    workingtext = workingtext[:workingtext.find("[ie]")] + workingtext[workingtext.find("[/ie]") + 5:]
        if "[vel]" in workingtext:
            if velinsts != workingtext.count("[/vel]"):
                return "Error: Uneven number of open and close tags for ɫ."
            else:
                for k in range(velinsts):
                    workingtext = workingtext[:workingtext.find("[vel]")] + workingtext[workingtext.find("[/vel]") + 6:]
        if "[etc]" in workingtext:
            if etcinsts != workingtext.count("[/etc]"):
                return "Error: Uneven number of open and close tags for rl."
            else:
                for l in range(etcinsts):
                    workingtext = workingtext[:workingtext.find("[etc]")] + workingtext[workingtext.find("[/etc]") + 6:]
    if "  " in workingtext:
        while "  " in workingtext:
            workingtext = " ".join(workingtext.split("  "))
    return workingtext


# glosses = order_glosses("\n\n".join(get_section(get_pages("Wurzburg Glosses", 499, 712), "SG")))
# glosses = "this is [GLat]some[/GLat] text [ie].i.[/ie] [GLat]that[/GLat] I made up on the     spot"
# print(rep_lat(glosses, "…", True))
# print(clear_tags(rep_lat(glosses, "[Latin]", False)))
# print(remove_newlines(rem_lat(glosses, False)))
