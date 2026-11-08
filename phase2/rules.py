# document-level rules: header party linking, PW witnesses, role pruning, PREC and label-specific boundary
# repair, counsel side, respondent list parsed from the header
import re
from collections import Counter

HDR = 2500
STOP = {"state", "india", "union", "the", "and", "others", "ors", "anr", "another", "limited", "ltd", "of", "government",
        "company", "corporation", "board", "bank", "commissioner", "officer", "secretary", "director", "general",
        "municipal", "district", "through", "by", "its", "dead", "lrs", "smt", "shri", "sri", "mr", "mrs", "ms", "dr"}
PW = re.compile(r"\(?\s*P\.?\s?W\.?\s?-?\s?\d+\s*\)?")
NAME = r"(?:[A-Z][a-z]+\.?\s){0,2}[A-Z][a-z]+"
COMMON_CAPS = {"further", "neuro", "surgeon", "however", "also", "thereafter", "accordingly", "learned", "doctor",
               "investigating", "officer", "constable", "head", "sub", "inspector", "medical", "the", "he", "she", "his"}


def taken_mask(text, spans):
    taken = [False] * (len(text) + 1)
    for sp in spans:
        for i in range(sp[0], sp[1]):
            taken[i] = True
    return taken


def party_variants(name, kinds):
    toks = [t.strip(".") for t in re.findall(r"[A-Za-z][A-Za-z.']+", name)]
    clean = [t for t in toks if t.lower() not in STOP and len(t) >= 3]
    out = {}
    if "full" in kinds:
        out[name.strip()] = "full"
    if "bigram" in kinds:
        for i in range(len(clean) - 1):
            out[clean[i] + " " + clean[i + 1]] = "bigram"
    if "first" in kinds and len(clean) >= 2:
        out[clean[0]] = "first"
    if "token" in kinds:
        for t in clean:
            out.setdefault(t, "token")
    return out


def witness_names(text):
    # names written right before or after a PW marker: "Jagdish (PW-5)", "PW 6- Mange Ram"
    names = Counter()
    for m in PW.finditer(text):
        left = text[max(0, m.start() - 45):m.start()]
        right = text[m.end():m.end() + 45]
        found = re.findall(r"(" + NAME + r")\s*[-,]?\s*$", left) + re.findall(r"^\s*[-,]?\s*(" + NAME + r")", right)
        for n in found:
            n = n.strip()
            if len(n) >= 4 and n.split()[0].lower() not in STOP:
                names[n] += 1
    return names


def resolve(text, spans):
    # tag body mentions of header appellant names and of PW-marked witness names
    taken = taken_mask(text, spans)
    out = list(spans)
    lex = {}
    for s, e, lab, c in spans:
        if lab == "APP" and s < HDR:
            for v in party_variants(text[s:e], ("full", "bigram", "first")):
                lex.setdefault(v, lab)
    for n in witness_names(text):
        lex.setdefault(n, "WIT")
    for v in sorted(lex, key=len, reverse=True):
        if len(v) < 4:
            continue
        for m in re.finditer(r"(?<!\w)" + re.escape(v) + r"(?!\w)", text, re.IGNORECASE):
            if not any(taken[m.start():m.end()]):
                out.append((m.start(), m.end(), lex[v], 0.0))
                for i in range(m.start(), m.end()):
                    taken[i] = True
    return sorted(out)


CITE_START = re.compile(r"^[\[(\{]?\s*(?:\(?\d{4}\)?|\[\d{4}\]|AIR\b|MANU\b|\d{4}\s*\(|SCC\b|SCR\b|JT\b|S\.?C\.?R\.?|\(\d{4})")
OPEN_CLOSE = {"(": ")", "[": "]", "{": "}"}


def prec_repair(text, spans):
    out = []
    for s, e, lab, c in spans:
        if lab == "PREC":
            # "Kerala[5]" becomes "Kerala"
            m = re.search(r"(\[\d{1,3}\]|(?<=[a-z.)\]])\d{1,2})$", text[s:e])
            if m and e - s - len(m.group(0)) > 3:
                e -= len(m.group(0))
            if re.search(r"\s(case|Case)$", text[s:e]):
                e -= 5
            while s > 0 and text[s - 1].isalpha() and text[s].isalpha():
                s -= 1
            for o, cl in OPEN_CLOSE.items():
                if text[s:e].count(o) > text[s:e].count(cl) and e < len(text) and text[e] == cl:
                    e += 1
        out.append((s, e, lab, c))
    # case name directly followed by its citation becomes one span
    out.sort()
    merged = []
    for sp in out:
        if merged and sp[2] == "PREC" and merged[-1][2] == "PREC":
            gap = text[merged[-1][1]:sp[0]]
            if re.fullmatch(r"[\s,]*", gap) and CITE_START.match(text[sp[0]:sp[1]]):
                p = merged.pop()
                merged.append((p[0], sp[1], "PREC", p[3]))
                continue
        merged.append(sp)
    return merged


def prune_roles(text, spans):
    # drop body APP mentions that match no header party name, and WIT mentions never next to a PW marker
    party = set()
    for s, e, lab, c in spans:
        if lab == "APP" and s < HDR:
            for v in party_variants(text[s:e], ("full", "bigram", "first", "token")):
                party.add(v.lower())
    wnames = set()
    for n in witness_names(text):
        wnames.add(n.lower())
    wtokens = set()
    for n in wnames:
        for t in n.split():
            wtokens.add(t)
    out = []
    for s, e, lab, c in spans:
        t = text[s:e].lower()
        words = set(t.split())
        if lab == "APP" and s >= HDR and party and t not in party and not (words & party):
            continue
        if lab == "WIT" and wnames and t not in wnames and not (words & wtokens):
            continue
        if lab == "WIT" and c == 0.0 and words & COMMON_CAPS:
            continue
        out.append((s, e, lab, c))
    return out


def label_repair(text, spans):
    out = []
    for s, e, lab, c in spans:
        # "TP Act)" becomes "TP Act"
        for o, cl in (("(", ")"), ("[", "]")):
            while e > s + 1 and text[e - 1] == cl and text[s:e].count(cl) > text[s:e].count(o):
                e -= 1
        if lab == "DATE":
            m = re.match(r"[A-Za-z]{3,}(?=\d)", text[s:e])
            if m:
                s += m.end()
            while e > s + 1 and text[e - 1] in ".)":
                e -= 1
        if lab == "CASENO":
            m = re.match(r"\s*\.*\s*of\s*\d{4}", text[e:e + 16])
            if m and not re.search(r"of\s*\d{4}$", text[s:e]):
                e += m.end()
        if lab == "STAT" and re.search(r"(Act|Rules|Code|Ordinance|Regulations)$", text[s:e]):
            m = re.match(r",?\s*(of\s*)?\d{4}\b", text[e:e + 12])
            if m:
                e += m.end()
        if lab == "CASENO":
            # "S.L.P. No. 14656 of 2000" keeps its prefix
            m = re.search(r"((?:[A-Z][A-Za-z]*\.){1,4}\s?|SPECIAL LEAVE PETITION \(\w+\)\s)$", text[max(0, s - 30):s])
            if m and text[s:e].startswith("No"):
                s -= len(m.group(1))
        out.append((s, e, lab, c))
    # "High Court" + "of Delhi"
    out.sort()
    merged = []
    for sp in out:
        if merged and sp[2] == "COURT" and merged[-1][2] == "COURT" and text[merged[-1][1]:sp[0]] == " " \
                and text[sp[0]:sp[1]].startswith("of "):
            p = merged.pop()
            merged.append((p[0], sp[1], "COURT", p[3]))
            continue
        merged.append(sp)
    return merged


STRONG_SIDE = re.compile(r"^\W{0,3}(?:(?:learned\s+)?(?:senior\s+)?(?:counsel|advocate|adv\.?|sr\.?\s*adv\.?|aag|asg|addl\.?[^,;]{0,25})"
                         r"[^.;]{0,30}?)?\s*(?:appearing\s+|who\s+appeared\s+)?(?:on\s+behalf\s+of|for)\s+(?:the\s+)?"
                         r"(?:learned\s+)?(appellants?|petitioners?|applicants?|respondents?|state)\b", re.I)


def counsel_side(text, spans):
    # "X, learned Senior Counsel for the appellants" decides A vs R counsel, only where it fires
    out = []
    for s, e, lab, c in spans:
        if lab in ("A.COUNSEL", "R.COUNSEL"):
            m = STRONG_SIDE.match(text[e:e + 90])
            if m:
                if m.group(1).lower().startswith(("appellant", "petitioner", "applicant")):
                    lab = "A.COUNSEL"
                else:
                    lab = "R.COUNSEL"
        out.append((s, e, lab, c))
    return out


RESP_MARK = re.compile(r"(?:\.{2,}|\u2026|-)?\s*\b(RESPONDENTS?|Respondents?)\b\s*(?:\(S\)|\(s\)|No\.?\s*\d+)?\s*:?")
VERSUS = re.compile(r"\b(?:VERSUS|Versus|versus|Vs\.?|V/s\.?|VS\.?)\s")
END_MARK = re.compile(r"\b(DATE OF JUDGMENT|J\s?U\s?D\s?G\s?M\s?E\s?N\s?T|JUDGMENT|BENCH:|ORDER\b|WITH\b|Through:|CORAM|PRAYER)")
TAIL = re.compile(r"(\s*(?:&|AND|and)\s*(?:ORS|Ors|ANR|Anr|OTHERS|Others|ANOTHER|Another)\.?)+.*$|\s*(?:ETC|Etc)\.?\s*$")
CUT = re.compile(r",\s*(?:through|Through|THROUGH|Age|AGE|aged|R/o|r/o|S/o|W/o|D/o|son of|wife of|resident)\b.*$")


def clean_party(seg):
    seg = seg.strip(" ,;:-\u2026")
    if "," in seg and len(seg.split(",", 1)[1].split()) > 6:
        seg = seg.split(",", 1)[0]
    seg = CUT.sub("", seg)
    seg = TAIL.sub("", seg).strip(" ,;:-")
    if seg.endswith(".") and not re.search(r"(?:\b[A-Z][A-Za-z]{0,4}\.|\b[A-Z]\.[A-Z]\.)$", seg):
        seg = seg[:-1]
    # "THE GENERAL SECRETARY (SHRI AMOL MAHAPATRA)" gives the bracketed name
    m = re.match(r"^(?:THE\s+)?[A-Z .]+\((?:SHRI|SMT\.?|SRI|MR\.?|MS\.?)\s+([^)]+)\)\s*(.*)$", seg)
    if m:
        return [m.group(1).strip(), m.group(2).strip(" .,")]
    seg = re.sub(r"^(?:M/S\.?|M/s\.?|Smt\.?|Shri|SHRI|Sri|Mr\.?|Mrs\.?|Ms\.?)\s*", "", seg)
    seg = re.sub(r"^(?:THE|The)\s+", "", seg)
    seg = re.sub(r"((?:UNION|LTD\.?|LIMITED|COMPANY|ASSOCIATION|CORPORATION|BANK|BOARD|MILLS|Ltd\.?|Limited)),\s*[A-Z][A-Za-z]+\.?$", r"\1", seg)
    if len(seg.split()) > 12:
        return []
    return [seg.strip(" ,;:-")]


def header_parties(text):
    # respondent list from the judgment header, as [(start, end, label)]
    head = text[:3000]
    seg_start = None
    seg_end = None
    m = re.search(r"\bRESPONDENTS?\s*(?:\(S\))?\s*:", head)
    if m:
        seg_start = m.end()
        e = END_MARK.search(head, seg_start)
        if e:
            seg_end = e.start()
        else:
            seg_end = min(len(head), seg_start + 400)
    else:
        v = VERSUS.search(head)
        if v:
            seg_start = v.end()
            r = RESP_MARK.search(head, seg_start)
            if r and r.start() - seg_start < 500:
                seg_end = r.start()
    if seg_start is None or seg_end is None or seg_end <= seg_start:
        return []
    seg = head[seg_start:seg_end]
    parts = re.split(r"\s*(?:\b\d{1,2}\s*[\].)]\s+|&\s*\d{1,2}\s*[\].)]\s*|;)", seg)
    names = []
    for p in parts:
        for n in clean_party(p):
            if len(n) >= 3 and not re.fullmatch(r"[\d\W]+", n):
                names.append(n)
    out = []
    for n in names:
        i = text.find(n, seg_start, seg_end + 5)
        if i >= 0:
            out.append((i, i + len(n), "RESP"))
    return out


def overlaps_any(sp, parsed):
    for s, e, lab in parsed:
        if sp[0] < e and s < sp[1]:
            return True
    return False


def link_header_parties(text, spans):
    # add the parsed header names (replacing overlapping header predictions) and tag their body mentions
    parsed = header_parties(text)
    out = list(spans)
    if parsed:
        kept = []
        for sp in out:
            if not (sp[0] < HDR and overlaps_any(sp, parsed)):
                kept.append(sp)
        out = kept
        for s, e, lab in parsed:
            out.append((s, e, "RESP", 0.0))
    taken = taken_mask(text, out)
    names = set()
    for s, e, lab in parsed:
        names.add(text[s:e])
    for n in sorted(names, key=len, reverse=True):
        if len(n) < 5:
            continue
        for m in re.finditer(r"(?<!\w)" + re.escape(n) + r"(?!\w)", text, re.I):
            if m.start() >= HDR and not any(taken[m.start():m.end()]):
                out.append((m.start(), m.end(), "RESP", 0.0))
                for i in range(m.start(), m.end()):
                    taken[i] = True
    return sorted(out)


def relabel_to_parties(text, spans):
    # body spans that exactly match a parsed header name become RESP
    names = set()
    for s, e, lab in header_parties(text):
        names.add(text[s:e].lower())
    out = []
    for s, e, lab, c in spans:
        if s >= HDR and text[s:e].lower() in names:
            lab = "RESP"
        out.append((s, e, lab, c))
    return out
