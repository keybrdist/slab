"""sample_names.py - what a sample's file name claims: type, tempo, key, keywords.

The name half of the sample index (tools/library/samples.py). Packs name
files their own way; the conventions handled here:

  "<pack> - NN <title>"          the lab's packs: NN is the track number
  TYPE_NAME_[PROCESSOR]_KEY/TEMPO   Fracture: DR BA PA ST FX
  PFX_Name 170 D                 Homemade Weapons: AMB BAS CYM DRM ETC KCK PRC SNR
  LM DNB <TYPE> NN <key> <bpm>   LMajor
  TSSPyyyy <bpm> Name Brk/_Fm    Total Science
  Name FMin - 130 BPM            Pugilist (and "DbMin or AbMin")
  160bpm name, 170 name          Om Unit

Everything returns plain values; `parse(stem, pack)` is the entry point.
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slabkit.theory import NAMES, QUALITIES, pitch_class  # noqa: E402

TYPES = ["break", "top", "perc-loop", "kick", "snare", "clap", "hat", "cymbal", "perc",
         "bass", "pad", "chord", "stab", "lead", "keys", "vocal", "fx", "riser", "atmos", "demo"]
DRUMS = {"break", "top", "perc-loop", "kick", "snare", "clap", "hat", "cymbal", "perc"}
TONAL = {"bass", "pad", "chord", "stab", "lead", "keys", "vocal"}

# chord suffixes beyond slabkit's: LMajor writes M for major, and 13ths
EXTRA_QUALITIES = {
    "M": [0, 4, 7], "M7": [0, 4, 7, 11], "M9": [0, 4, 7, 11, 14], "M11": [0, 4, 7, 11, 14, 17],
    "13": [0, 4, 7, 10, 14, 21], "m13": [0, 3, 7, 10, 14, 21], "maj11": [0, 4, 7, 11, 14, 17],
    "Maj": [0, 4, 7], "Min": [0, 3, 7], "minor": [0, 3, 7], "major": [0, 4, 7],
}
CHORD_Q = {**QUALITIES, **EXTRA_QUALITIES}
MINOR_Q = {"m", "min", "Min", "minor", "m7", "m9", "m11", "m13", "m6", "madd9", "mmaj7"}

# packs whose titles are track names, not descriptions: every file is a
# drum part of the named track
TITLE_PACKS = {"Fanu - FatGyver - D&B BREAKS SAMPLE PACK by Fanu": "break",
               "Fanu - FatGyver - FatGyver's Hip Hop Breaks Sample Pack": "break"}

# words that end in a type word but aren't one
NOT_WORDS = {"whats", "what", "pride", "primo", "chat", "that", "hatter", "athat", "shatter",
             "drumming", "rimshot"}


def title_of(stem):
    """("NN", title) from "<pack> - NN <title>"; the stem when there's no number."""
    stem = re.sub(r"\.(wav|aiff?|flac|mp3)$", "", stem, flags=re.I)
    m = re.search(r" - (\d{1,3}) (.+)$", stem)
    return (int(m.group(1)), m.group(2).strip()) if m else (None, stem.strip())


def tokens(title):
    """Lowercase words: split on punctuation, camelCase and letter/digit
    seams ("Brk30" -> brk 30, "160bpm" -> 160 bpm)."""
    t = re.sub(r"([a-z])([A-Z])", r"\1 \2", title)
    t = re.sub(r"([A-Za-z])(\d)", r"\1 \2", t)
    t = re.sub(r"(\d)([A-Za-z])", r"\1 \2", t)
    return [w for w in re.split(r"[^A-Za-z0-9#]+", t.lower()) if w]


# ── tempo ─────────────────────────────────────────────────────────────────
INDEX_WORDS = {"drum", "drums", "hat", "kick", "snare", "bass", "pad", "lead", "mid", "sfx", "atmos",
               "fx", "tops", "top", "beat", "chord", "stab", "tam", "hit", "loop", "extras", "xtraz",
               "sample", "part", "vol", "track", "layer", "edit", "no", "pads", "keys", "perc",
               "bassline", "riff", "phrase", "brk", "break"}


def tempo_of(title):
    """(bpm, how) from the title, or (None, None)."""
    m = re.search(r"(\d{2,3}(?:\.\d+)?)\s*bpm", title, re.I)
    if m and 60 <= float(m.group(1)) <= 200:
        return float(m.group(1)), "bpm"
    clean = re.sub(r"-[A-Z0-9 ]+-", " ", title)          # Fracture's -1978-, -MACKIE- tags
    m = re.search(r"(?:^|[\s_])(1\d\d)5$", clean)        # Homemade Weapons' 1725 = 172.5
    if m:
        return float(m.group(1)) + 0.5, "trailing"
    for m in re.finditer(r"(?:^|[\s_(])(\d{3})(?=$|[\s_)])", clean):
        if 100 <= int(m.group(1)) <= 200:
            return float(m.group(1)), "number"
    m = re.search(r"(?:^|[\s_])([6-9]\d)(?:\s+[A-G][#b]?m?)?$", clean)
    if m:
        before = tokens(clean[:m.start(1)])
        if not before or before[-1] not in INDEX_WORDS:
            return float(m.group(1)), "trailing"
    return None, None


# ── key, root, chord ──────────────────────────────────────────────────────
NOTE = r"[A-G](?:#|b|B(?=[Mm]in|[Mm]aj))?"


def _pc(s):
    return pitch_class(s[0] + s[1:].replace("B", "b"))


def _chord(root, q):
    iv = CHORD_Q.get(q)
    if iv is None:
        return None
    r = _pc(root)
    return sorted({(r + i) % 12 for i in iv})


def key_of(title, drums=False):
    """What the title says about pitch: {kind, root, mode, alts, chord, pcs}
    or None. kind: "key" (mode given), "root" (a bare note), "chord" (a
    chord symbol), "progression" (several chords)."""
    if drums:
        return None
    t = re.sub(r"-[A-Z0-9 ]+-", " ", title)                 # Fracture's -DSM- tags
    t = re.sub(r"\s*-?\s*\d{2,3}(?:\.\d+)?\s*bpm\s*\)?\s*$", "", t, flags=re.I).strip(" -_()")
    # "FMin", "GbMin", "EMaj", "Bmin", "DbMin or AbMin"
    found = [(m.group(1), "minor" if m.group(2).lower().startswith("min") else "major")
             for m in re.finditer(rf"(?<![A-Za-z])({NOTE})\s?((?:[Mm]in|[Mm]aj)(?:or)?)(?![a-z])", t)]
    if found:
        (r, md), alts = found[0], found[1:]
        return {"kind": "key", "root": _pc(r), "mode": md,
                "alts": [[_pc(a), m] for a, m in alts], "text": " or ".join(f"{a}{m[:3]}" for a, m in found)}
    # LMajor progressions: "Chord Riff Cm Gm Ab 174"
    t2 = re.sub(r"[\s_]+\d{2,3}$", "", t).strip()
    prog = re.findall(rf"(?:^|\s)({NOTE})(m?)(?=\s|$)", t2)
    if len(prog) >= 3 and re.search(r"(?:riff|progression)", t2, re.I):
        r, q = prog[0]
        return {"kind": "progression", "root": _pc(r), "mode": "minor" if q else "major", "alts": [],
                "chords": [a + b for a, b in prog], "text": " ".join(a + b for a, b in prog)}
    # a chord or note at the end: "_Fm", "_A#m", " C#M11", "Cm-D", "Pad 4.A", "SubC#", "_GC" (no)
    pat = rf"(?:^|[\s_.(]|(?<=Sub)|(?<=Snb))({NOTE})(maj\d*|min|m(?:aj)?\d*|M\d*|dim|sus\d?|add9|\d{{1,2}})?(-[A-G])?$"
    m = re.search(pat, t2) or re.search(rf"(?:^|[\s_(])({NOTE})(m\d*|M\d*)?(?=\s+-\s)", t2)
    if not m:
        return None
    root, q = m.group(1), m.group(2) or ""
    pcs = _chord(root, q) if q else None
    if q and pcs is None:
        return None
    if q in ("m", "min", "Min"):
        return {"kind": "key", "root": _pc(root), "mode": "minor", "alts": [], "text": root + "m",
                "chord": root + q, "pcs": pcs}
    if q:
        return {"kind": "chord", "root": _pc(root), "mode": "minor" if q in MINOR_Q else "major", "alts": [],
                "chord": root + q + (m.group(3) or ""), "pcs": pcs, "text": root + q}
    return {"kind": "root", "root": _pc(root), "mode": None, "alts": [], "text": root}


# ── type ──────────────────────────────────────────────────────────────────
# (type, regex over " lowercase title ", confidence); first match wins
PREFIXES = [
    (r"^dr_", None), (r"^ba_", "bass"), (r"^pa_", "pad"), (r"^st_", "stab"), (r"^fx_", "fx"),
    (r"^amb_", "atmos"), (r"^bas[_ ]", "bass"), (r"^cym_", "cymbal"), (r"^drm_", "break"),
    (r"^etc_", None), (r"^kck_", "kick"), (r"^prc_", "perc"), (r"^snr_", "snare"),
]
WORDS = [
    ("demo", r"\bdemo\b"),
    ("top", r"top ?loop|\btops\b|\btop \d{3}\b|hats? ?loop|\bhats?\b.*\b1[4-9]\d\b|16th hats|\bteenths\b|hat triplets|springhats|dirtyhats|scorpio hats|hatdelay"),
    ("perc-loop", r"perc(ussion)? ?loop|\bpercs?\b.*\b1[0-9]\d\b|\b1[0-9]\d\b.*\bpercs?\b|percverbs|percdelays|technopercs|"
                  r"shaker ?loop|tamb ?l? ?oop|\btambs?\b.*\b1\d\d\b|\b1\d\d\b.*tambs|bongo congas|triangleloop|pots n pans|"
                  r"cabasa|shekere|\bperc loop|rim ?shot loop|tom loop|_perc$"),
    ("break", r"\bbrk\b|\bbreaks?\b|break\b|\bamen\b|\bbeat\b|beats\b|drum ?loop|drumloop|\bdrums\b|"
              r"\bgroove(box)?\b|\bfill(er)?\b|\bdrumfill|^drum \d+$|halftime beat|\bbreakhalf|breaklayer|thinkchops|"
              r"thinktambs|amen\b|\bpolydrums\b|machinic groove|^1\d\d .*pants|chops?\b"),
    ("riser", r"\briser\b|\brise\b|uplifter|downlifter|\bimpact\b|sub ?drop|\bbuild\b|reverse crash|rev china|\bswell\b"),
    ("vocal", r"\bvocals?\b|\bvox\b(?!.*pad)|\bvoices\b(?!.*pad)|\bchoir\b|\bacapella\b|\bspoken\b|vocal ?count|vocalverb"),
    ("kick", r"kick|\bkck\b|\bbd\b|bassdrum|kik\b"),
    ("snare", r"snare|\bsnr\b|sanre"),
    ("clap", r"clap"),
    ("hat", r"\bhats?\b|hihat|hi-hat|\bhh\b|\bclosed\b|\bcls\b|open hat|hatverb|razor hat"),
    ("cymbal", r"crash|\bride\b|clipride|cymbal|china|\bchin\b|splash|\bclash\b|\bcym\b|bell ringer"),
    ("pad", r"\bpads?\b|pad\b|stringpad|ravepad|evolvepad|\bstrings?\b|strings?\b|stringbrass"),
    ("chord", r"chords?\b|\bchrd\b"),
    ("stab", r"\bstabs?\b|stabby|stabbys|\bskank\b"),
    ("keys", r"\bkeys\b|rhodes|piano|\bep\b|\borgan\b|wurli"),
    ("bass", r"\bbass|\bsub\b|subz|\bsubb|reese|\bree\b|\brees\b|\b808\b(?! ?(kick|snare|hat|top|perc))|wubs?\b|wobbler?|donk|bassline|\bsnb\b"),
    ("atmos", r"\bdrone\b"),
    ("lead", r"\blead\b|\briff\b|\barp\b|arpy|sequence|\bmelody\b|\bacid\b|\b303\b|flute|\bhorn\b|\bbrass\b|"
             r"\bgtr|guitar|pluck|\bbleeps?\b|\bphrase\b|^mid \d+$|\bsynth\b|lyra\b"),
    ("perc", r"\bperc|\brim|rim\b|\btoms?\b|widetom|cowbell|\bclave|shaker|\btamb|\btam\b|conga|bongo|tabla|"
             r"woodblock|\bbottle\b|\bcan\b|\bpipe\b|\bmetal\b|\bhit\b|percie|\bgongs?\b|snap|usul|klakk"),
    ("fx", r"\bs?fx\b|noise|glitch|\bzap\b|laser|siren|alarm|warble|boingg?|wowow|creak|door|broom|vroom|scrape|"
           r"\bcaw\b|meow|groan|banshee|feedback|flyby|flanger|\bsweep|swp\b|\bdly\b|mangled|weird"),
    ("atmos", r"atmos|ambien|drone|texture|soundscape|\bhiss\b|\bwind\b|field|steam|dronie|whistl"),
]


def type_of(title, pack):
    """(type, confidence, loop hint True/False/None) from the title."""
    lt = title.lower()
    loop = None
    if re.search(r"\bloop\b|loop$|\bbrk\b|\bbreak|\bbeat\b|\bgroove|\bsequence\b|bassline|\briff\b|\bpattern\b|\bptrn\b|\bphrase\b", lt):
        loop = True
    if re.search(r"\bhit\b|one ?shot|\bshot\b", lt):
        loop = False
    if pack in TITLE_PACKS:
        if re.search(r"_perc$", lt):
            return "perc-loop", 0.8, True
        return TITLE_PACKS[pack], 0.8, True
    prefix_type = None
    for rx, t in PREFIXES:
        if re.search(rx, lt):
            prefix_type = t or "?"
            break
    bare = re.sub(r"\bw(ith)? fx tail\b", "", re.sub(r"^[A-Za-z]{2,3}_", "", title), flags=re.I)
    words = " ".join(tokens(bare))
    m = re.match(r"(bass|break|pad|kick|snare|vocal|s?fx|chord|stab|synth|lead|texture|drum ?loop|drumloop|perc loop)\b\s*-?", lt)
    if m and m.group(1) in ("fx", "sfx", "synth", "texture") and re.search(r"riser|impact|swell|sub ?drop", lt):
        m = None                     # "FX - Reesiest Riser": the specific word wins
    if m and not prefix_type:
        t = {"sfx": "fx", "synth": "lead", "texture": "atmos", "drum loop": "break", "drumloop": "break",
             "perc loop": "perc-loop"}.get(m.group(1), m.group(1))
        return t, 0.9, loop
    for t, rx in WORDS:
        if re.search(rx, words) and not any(w in NOT_WORDS for w in re.findall(r"[a-z]+", words)
                                            if re.search(rx, " " + w + " ")):
            if prefix_type and prefix_type != "?":
                # a prefix fixes the family; words refine within it
                fam = {"cymbal": {"hat", "top", "cymbal"}, "perc": {"perc", "perc-loop", "clap", "top", "riser"},
                       "atmos": {"pad", "atmos", "fx", "riser"}, "stab": {"stab", "chord", "keys", "perc"},
                       "fx": {"fx", "riser", "vocal", "perc", "atmos"}, "bass": {"bass"}, "pad": {"pad"},
                       "kick": {"kick"}, "snare": {"snare", "clap"}, "break": {"break", "top"}}
                if t in fam.get(prefix_type, set()):
                    if prefix_type == "stab" and t == "perc":
                        return "stab", 0.9, loop
                    return t, 0.9, loop
                return prefix_type, 0.9, loop
            if prefix_type == "?" and lt.startswith("dr_"):
                return {"hat": "top", "perc": "perc-loop"}.get(t, t), 0.9, loop
            return t, (0.85 if t not in ("lead", "perc", "fx") else 0.7), loop
    if prefix_type and prefix_type != "?":
        return prefix_type, 0.9, loop
    if prefix_type == "?" and lt.startswith("dr_"):
        return "break", 0.6, loop
    return None, 0.0, loop


# ── keywords ──────────────────────────────────────────────────────────────
# canonical keyword -> spellings in titles; spellings of 4+ letters also
# match glued onto other words ("Ruggedamen", "thinktambs", "Snareverb")
VOCAB = {
    "amen": ["amen", "amn"], "think": ["think"], "apache": ["apache", "apachey"], "hotpants": ["hotpants", "pants"],
    "break": ["break", "breaks", "brk"], "beat": ["beat", "beats"], "drums": ["drums", "drm", "drum"],
    "fill": ["fill", "filler", "drumfill"], "top": ["top", "tops"], "kick": ["kick", "kicks", "kck"],
    "snare": ["snare", "snares", "snr", "sanre"], "clap": ["clap", "claps", "clapsnare"],
    "hat": ["hat", "hats", "hihat", "hh"], "open-hat": ["open"], "cymbal": ["cymbal", "cym"],
    "crash": ["crash"], "ride": ["ride", "clipride"], "china": ["china", "chin"], "perc": ["perc", "percs", "prc", "percussion", "percu"],
    "rim": ["rim", "rimshot", "rims"], "tom": ["tom", "toms", "widetom"], "shaker": ["shaker"],
    "tambourine": ["tamb", "tambs", "tam", "tambourine"], "conga": ["conga", "congas"], "bongo": ["bongo", "bongos"],
    "tabla": ["tabla", "tablaingles"], "clave": ["clave"], "cowbell": ["cowbell", "caubell"], "triangle": ["triangle"],
    "gong": ["gong", "gongs"], "bass": ["bass", "bas"], "sub": ["sub", "subb", "subz", "subby"],
    "reese": ["reese", "ree", "rees", "reesiest"], "808": ["808", "808ish"], "909": ["909"], "303": ["303"],
    "acid": ["acid"], "wobble": ["wobble", "wobbler", "wub", "wubs", "wubby"], "donk": ["donk"],
    "hoover": ["hoover", "mentasm"], "pad": ["pad", "pads"], "stab": ["stab", "stabs", "stabby", "stabbys"],
    "chord": ["chord", "chords", "chrd"], "lead": ["lead"], "arp": ["arp", "arpy"], "riff": ["riff"],
    "sequence": ["sequence", "seq"], "melody": ["melody"], "keys": ["keys"], "rhodes": ["rhodes"],
    "piano": ["piano"], "organ": ["organ"], "strings": ["string", "strings", "stringpad", "stringbrass"],
    "brass": ["brass"], "flute": ["flute", "fluteish"], "horn": ["horn"], "guitar": ["guitar", "gtr"],
    "bell": ["bell"], "choir": ["choir"], "vocal": ["vocal", "vox", "voice", "voices", "vocalverb"],
    "drone": ["drone", "dronie"], "noise": ["noise", "hiss"], "texture": ["texture", "textural"],
    "atmos": ["atmos", "amb", "ambient", "atmosphere"], "riser": ["riser", "rise", "risen", "uplifter"],
    "impact": ["impact"], "sweep": ["sweep", "swp"], "fx": ["fx", "sfx"], "glitch": ["glitch", "glitchy"],
    "laser": ["laser", "lazer"], "siren": ["siren"], "zap": ["zap"], "bleep": ["bleep", "bleeps"],
    "granular": ["granular", "grain"], "rave": ["rave", "raver"], "hardcore": ["hardcore", "hardcoree"],
    "jungle": ["jungle"], "techstep": ["techstep"], "neuro": ["neuro", "neurofunk", "neurality"],
    "dub": ["dub", "dubbin", "dubbage", "dubtek", "dubsteppa"], "techno": ["techno", "tek", "tekky", "tekkerz"],
    "house": ["house", "haus", "housey"], "garage": ["garage"], "hiphop": ["hiphop", "hiphoppy"],
    "funk": ["funk", "funky"], "jazz": ["jazz", "jazzy"], "soul": ["soul"], "gqom": ["gqom"], "gabber": ["gabber"],
    "halftime": ["halftime", "halfy", "half"], "triplet": ["triplet", "triplets", "8t", "trp"],
    "swing": ["swing", "swng", "swn"], "reverb": ["verb", "reverb", "verbd", "vrb", "vrbd"],
    "delay": ["delay", "dly", "echo", "delayed"], "spring": ["spring"], "flange": ["flange", "flanger", "flangey"],
    "phaser": ["phase", "phased", "phaser", "junophase"], "chorus": ["chorus"], "lofi": ["lofi", "lo fi"],
    "12bit": ["12 bit", "12bit"], "vinyl": ["vinyl", "crackly"], "tape": ["tape", "tapefied"],
    "distorted": ["disto", "distorted", "distortion", "dirty", "grit", "gritty", "crushed", "smashed", "clippage", "fucked", "harsh", "gnarly"],
    "sp1200": ["sp12", "sp 12", "sp1200"], "s950": ["s950"], "akai": ["akai", "s3200"], "emu": ["emu"],
    "mpc": ["mpc"], "machinedrum": ["machinedrum"], "m1": ["m1"], "d50": ["d50"], "juno": ["juno"],
    "jupiter": ["jupiter"], "wavestation": ["wavestation"], "obx": ["obx", "ob6"], "diva": ["diva"],
    "moog": ["moog", "moogsluggery"], "monopoly": ["monopoly"], "poly6": ["poly6"], "sh09": ["sh09", "sh 09"],
    "volca": ["volca"], "virus": ["virus"], "serum": ["serum"], "blofeld": ["blofeld"], "fm": ["fm"],
    "tx81z": ["tx81z"], "tanzbar": ["tanzbar"], "operator": ["operator"], "pocket-operator": ["pocket operator"],
    "dark": ["dark", "darker"], "spooky": ["spooky", "creepy", "creeper", "menacing", "demonic", "evil", "anguish"],
    "deep": ["deep", "deeper"], "warm": ["warm"], "lush": ["lush", "lushious", "creamy"], "sad": ["sad"],
    "wide": ["wide"], "dry": ["dry"], "wet": ["wet"], "clean": ["clean"], "detuned": ["detuned"],
    "reverse": ["reverse", "rev"], "sine": ["sine"], "pitched": ["pitched", "pitch"], "atonal": ["atonal"],
}
_SHORT, _LONG = {}, {}
for canon, spells in VOCAB.items():
    for s in spells:
        (_LONG if len(s) >= 4 and " " not in s else _SHORT)[s] = canon
_PHRASES = {s: c for c, ss in VOCAB.items() for s in ss if " " in s}
STOP = {"the", "and", "with", "w", "of", "a", "an", "to", "in", "on", "for", "or", "it", "is", "my", "your",
        "me", "you", "dnb", "lm", "tssp", "loop", "sample", "pack", "bpm", "mackie", "dsm", "1978", "decimated",
        "low", "med", "medium", "high", "gain", "tail", "ft", "vs", "etc", "alt", "edit"}


def keywords(title):
    """(canonical keywords, other words) from a title."""
    toks = tokens(title)
    flat = " ".join(toks)
    canon, rest = [], []
    for ph, c in _PHRASES.items():
        if re.search(rf"\b{re.escape(ph)}\b", flat):
            canon.append(c)
    for w in toks:
        if w in NOT_WORDS:
            rest.append(w)
            continue
        hit = _SHORT.get(w) or _LONG.get(w)
        if not hit and len(w) > 4:
            for s, c in _LONG.items():
                if len(s) >= 4 and (w.startswith(s) or w.endswith(s) or (len(s) >= 5 and s in w)):
                    canon.append(c)
                    hit = True
        if isinstance(hit, str):
            canon.append(hit)
        elif not hit and w not in STOP and not w.isdigit() and len(w) > 1:
            rest.append(w)
    seen = set()
    return [c for c in canon if not (c in seen or seen.add(c))], [w for w in dict.fromkeys(rest) if w not in seen]


def parse(stem, pack):
    num, title = title_of(stem)
    t, conf, loop = type_of(title, pack)
    bpm, how = tempo_of(title)
    key = key_of(title, drums=t in DRUMS or (t is None and pack in TITLE_PACKS))
    kw, words = keywords(title)
    return {"number": num, "title": title, "type": t, "type_conf": conf, "loop": loop,
            "bpm": bpm, "bpm_how": how, "key": key, "keywords": kw, "words": words}


def key_text(root, mode):
    if root is None:
        return None
    return NAMES[root % 12] + ("" if not mode else " " + mode)


if __name__ == "__main__":
    import json
    for s in sys.argv[2:]:
        print(json.dumps(parse(s, sys.argv[1]), ensure_ascii=False))
