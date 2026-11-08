# rules taken from the training annotations
import re

from data import LABELS

DISSENT = LABELS.index("Dissent")

# "Prasad, Member (dissenting)" / "Prasad (Minority)"
DISSENT_HEADER = re.compile(r"^\W*\w+(,\s*Member)?\s*\((dissenting|minority)\)", re.I)


def dissent_rule(doc, pred):
    # a dissent block starts with the dissenting member's header and runs to the end of the document
    for i in range(len(doc["sents"])):
        if DISSENT_HEADER.match(doc["sents"][i]):
            return pred[:i] + [DISSENT] * (len(pred) - i)
    return pred
