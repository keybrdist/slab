#!/usr/bin/env python3
"""samples.py - index a folder of samples by what they are and how they sound.

Every audio file (FLAC, WAV, AIFF, MP3) under a root gets one record: what
its name claims (type, tempo, key, keywords) beside what its audio
measures (type, loop or one-shot, tempo, bars, key, pitch, loudness,
tone, groove, vocals), with a confidence for each, and a flag wherever the
two disagree. Queries read the cached index and answer in well under a
second.

  tools/library/samples.py index                         analyze what's new
  tools/library/samples.py find --type break --bpm 160 --swing 6 --top dark
  tools/library/samples.py find --type pad --key Fm --lacks Eb
  tools/library/samples.py find sub --type bass --oneshot --steady --clean
  tools/library/samples.py find --type top --bpm 174 --tone bright
  tools/library/samples.py find --vocal
  tools/library/samples.py text "dusty rhodes chords"    CLAP free-text search
  tools/library/samples.py like PATH                     sounds like this one
  tools/library/samples.py show PATH ...                 every field
  tools/library/samples.py check                         names vs audio, per pack

From Python (the lab's scripts):

  sys.path.insert(0, "<slab>/tools/library"); import samples
  lib = samples.load()
  for r in lib.find(type="break", bpm=160, swing=6, tone="dark"):
      track.audio(samples.playable(r), ...)     # MP3s decode once to WAV

The packs default to $SLAB_LIBRARY/packs (~/Music/Slab/Library/packs) and
the index to $SLAB_SAMPLE_INDEX, default $SLAB_LIBRARY/.sample-index:
index.json, the per-file caches (cache/, keyed by path, size and mtime, so
a rerun analyzes only new or changed files), the CLAP embeddings, and the
model venv. `index --root` points it at another folder, and is remembered. The audio
measures need numpy and ffmpeg; the AudioSet tags (vocals, instruments)
and free-text search need the venv, made once with `samples.py models
--setup` (torch, transformers and two pretrained models, ~2.5 GB, all
inside the index folder). Without it the index still builds; vocal flags
then come only from names.

Schema, method, accuracy and weak spots: tools/library/SAMPLES.md.
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SLAB = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(SLAB, "tools"))

import sample_names as N  # noqa: E402
from slabkit.theory import NAMES  # noqa: E402

AUDIO = (".flac", ".wav", ".aif", ".aiff", ".mp3")
AUDIO_VERSION = 3      # bump when sample_audio's measures change
MODELS_VERSION = 1     # bump when sample_models' output changes
INDEX_VERSION = 1


def slab_library():
    """$SLAB_LIBRARY, default ~/Music/Slab/Library (as tools/library's other scripts)."""
    return os.path.expanduser(os.environ.get("SLAB_LIBRARY") or "~/Music/Slab/Library")


def index_dir(given=None):
    """$SLAB_SAMPLE_INDEX, default .sample-index in the Slab library: beside
    the packs, outside any checkout, and hidden from slab's browser."""
    return os.path.abspath(os.path.expanduser(given or os.environ.get("SLAB_SAMPLE_INDEX")
                                              or os.path.join(slab_library(), ".sample-index")))


def _key(rel):
    return hashlib.sha1(rel.encode()).hexdigest()[:20]


def _write_json(path, obj):
    tmp = path + ".part"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def scan(root):
    """[(rel, abs, size, mtime)] of every audio file under root, sorted."""
    out = []
    for r, ds, fs in os.walk(root):
        ds[:] = sorted(d for d in ds if not d.startswith("."))
        for f in sorted(fs):
            if f.lower().endswith(AUDIO) and not f.startswith("._"):
                p = os.path.join(r, f)
                st = os.stat(p)
                out.append((os.path.relpath(p, root), p, st.st_size, int(st.st_mtime)))
    return out


def _fresh(entry, size, mtime, version):
    return entry is not None and entry.get("size") == size and entry.get("mtime") == mtime and entry.get("v") == version


def _analyze_one(job):
    rel, path, size, mtime, out = job
    import sample_audio
    t = time.time()
    try:
        feats = sample_audio.analyze(path)
        entry = {"rel": rel, "size": size, "mtime": mtime, "v": AUDIO_VERSION, "features": feats}
    except Exception as e:  # noqa: BLE001 - one bad file must not stop the run
        entry = {"rel": rel, "size": size, "mtime": mtime, "v": AUDIO_VERSION, "error": f"{type(e).__name__}: {e}"}
    entry["secs"] = round(time.time() - t, 2)
    _write_json(out, entry)
    return rel, "error" in entry


def analyze_audio(root, idx, files, jobs, quiet=False):
    cache = os.path.join(idx, "cache", "audio")
    os.makedirs(cache, exist_ok=True)
    todo = []
    for rel, path, size, mtime in files:
        out = os.path.join(cache, _key(rel) + ".json")
        if not _fresh(_read_json(out), size, mtime, AUDIO_VERSION):
            todo.append((rel, path, size, mtime, out))
    if not todo:
        return 0, 0
    if not quiet:
        print(f"audio: {len(todo)} of {len(files)} files to analyze with {jobs} workers", flush=True)
    t0, done, errs = time.time(), 0, 0
    with cf.ProcessPoolExecutor(max_workers=jobs) as ex:
        for rel, err in ex.map(_analyze_one, todo, chunksize=1):
            done += 1
            errs += err
            if err and not quiet:
                print(f"  error: {rel}", flush=True)
            if not quiet and (done % 50 == 0 or done == len(todo)):
                el = time.time() - t0
                print(f"  {done}/{len(todo)} in {el:.0f}s, ~{el / done * (len(todo) - done):.0f}s left", flush=True)
    return len(todo), errs


def venv_python(idx):
    p = os.path.join(idx, "venv", "bin", "python")
    return p if os.path.exists(p) else None


def run_models(root, idx, files, quiet=False):
    """AudioSet tags and CLAP embeddings, in the venv (sample_models.py)."""
    py = venv_python(idx)
    if not py:
        if not quiet:
            print("models: no venv (samples.py models --setup); skipping tags and text search")
        return None
    cache = os.path.join(idx, "cache", "models")
    os.makedirs(cache, exist_ok=True)
    todo = []
    for rel, path, size, mtime in files:
        out = os.path.join(cache, _key(rel) + ".json")
        if not _fresh(_read_json(out), size, mtime, MODELS_VERSION):
            todo.append({"rel": rel, "path": path, "size": size, "mtime": mtime, "out": out, "v": MODELS_VERSION})
    if not todo:
        return 0
    jobs = os.path.join(idx, "models-jobs.json")
    _write_json(jobs, todo)
    if not quiet:
        print(f"models: {len(todo)} files to tag", flush=True)
    env = dict(os.environ, HF_HOME=os.path.join(idx, "hf"), TOKENIZERS_PARALLELISM="false")
    r = subprocess.run([py, os.path.join(HERE, "sample_models.py"), "tag", jobs], env=env)
    if r.returncode != 0:
        print("models: tagging failed; the index is built without the missing tags", file=sys.stderr)
    return len(todo)


def setup_models(idx):
    """Create the venv and fetch the two models (~2.5 GB, all under idx)."""
    uv = shutil.which("uv")
    venv = os.path.join(idx, "venv")
    os.makedirs(idx, exist_ok=True)
    if not os.path.exists(os.path.join(venv, "bin", "python")):
        if uv:
            subprocess.run([uv, "venv", venv, "--python", "3.13", "-q"], check=True)
        else:
            subprocess.run([sys.executable, "-m", "venv", venv], check=True)
    py = os.path.join(venv, "bin", "python")
    if uv:
        subprocess.run([uv, "pip", "install", "-q", "--python", py, "torch", "torchaudio", "transformers", "numpy"], check=True)
    else:
        subprocess.run([py, "-m", "pip", "install", "-q", "torch", "torchaudio", "transformers", "numpy"], check=True)
    env = dict(os.environ, HF_HOME=os.path.join(idx, "hf"))
    subprocess.run([py, os.path.join(HERE, "sample_models.py"), "fetch"], env=env, check=True)


# ── assembly: names beside audio ─────────────────────────────────────────
FAMILY = {"break": "drum-loop", "top": "drum-loop", "perc-loop": "drum-loop",
          "kick": "drum-hit", "snare": "drum-hit", "clap": "drum-hit", "hat": "drum-hit", "cymbal": "drum-hit",
          "perc": "drum-hit", "bass": "tonal", "pad": "tonal", "chord": "tonal", "stab": "tonal", "lead": "tonal",
          "keys": "tonal", "vocal": "vocal", "fx": "fx", "riser": "fx", "atmos": "fx", "demo": "demo"}
NAMES12 = NAMES
MAJOR_SCALE, MINOR_SCALE = [0, 2, 4, 5, 7, 9, 11], [0, 2, 3, 5, 7, 8, 10]
CHORDS = {"": [0, 4, 7], "m": [0, 3, 7], "5": [0, 7], "sus2": [0, 2, 7], "sus4": [0, 5, 7], "7": [0, 4, 7, 10],
          "maj7": [0, 4, 7, 11], "m7": [0, 3, 7, 10], "m9": [0, 3, 7, 10, 2], "maj9": [0, 4, 7, 11, 2],
          "9": [0, 4, 7, 10, 2], "m11": [0, 3, 7, 10, 2, 5], "dim": [0, 3, 6], "madd9": [0, 3, 7, 2],
          "add9": [0, 4, 7, 2], "m6": [0, 3, 7, 9], "6": [0, 4, 7, 9]}
# AudioSet voice labels (Speech, Singing, Choir...) at or above this count as a
# vocal. CLAP zero-shot prompts don't separate processed vocals from drums, so
# CLAP serves search only (SAMPLES.md, Weak spots).
AST_VOCAL = 0.35


def _clip(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def note_text(p):
    if p is None:
        return None
    q = int(round(p))
    return f"{NAMES12[q % 12]}{q // 12 - 1}"


def key_str(root, mode):
    return None if root is None else NAMES12[root % 12] + ({"minor": "m", "major": ""}.get(mode, "?") if mode else "")


def key_relation(a, b):
    """How key a=(root, mode) relates to b: same, relative, parallel,
    fifth, or other. A missing mode matches either."""
    (ra, ma), (rb, mb) = a, b
    if ra == rb and (ma == mb or not ma or not mb):
        return "same"
    if ma and mb and ma != mb:
        if (ma == "minor" and (ra + 3) % 12 == rb) or (ma == "major" and (ra + 9) % 12 == rb):
            return "relative"
        if ra == rb:
            return "parallel"
    if (ra - rb) % 12 in (5, 7) and (ma == mb or not ma or not mb):
        return "fifth"
    return "other"


def scale_fit(chroma, root, mode):
    """Share of the chroma inside the key's scale (either mode if unknown)."""
    if chroma is None or root is None:
        return None
    fits = []
    for sc in ([MINOR_SCALE] if mode == "minor" else [MAJOR_SCALE] if mode == "major" else [MAJOR_SCALE, MINOR_SCALE]):
        fits.append(sum(chroma[(root + i) % 12] for i in sc))
    return round(max(fits), 3)


def best_scale_fit(chroma):
    return max(scale_fit(chroma, k, md) for k in range(12) for md in ("major", "minor")) if chroma else None


def main_pcs(chroma, rel=0.35, most=7):
    if not chroma:
        return []
    top = max(chroma) or 1
    order = sorted(range(12), key=lambda i: -chroma[i])
    return [i for i in order if chroma[i] / top >= rel][:most]


def chord_guess(chroma, bass=None):
    """Best chord for a chroma: template weight minus what it leaves out,
    a cost per note so a triad beats a padded ninth, and a nudge toward
    the strongest bass-register pitch class. -> (root, symbol)."""
    if not chroma:
        return None, None
    top, btop = max(chroma) or 1, (max(bass) or 1) if bass else 1
    best = (None, None, -9.0)
    for r in range(12):
        for q, iv in CHORDS.items():
            pcs = {(r + i) % 12 for i in iv}
            inside = sum(chroma[p] for p in pcs)
            score = inside - (1 - inside) * 0.8 - 0.09 * len(pcs) + 0.02 * chroma[r] / top
            if bass:
                score += 0.1 * bass[r] / btop
            if score > best[2]:
                best = (r, NAMES12[r] + q, score)
    return best[0], best[1]


def audio_key(chroma, root_hint, force=False):
    """Krumhansl key, nudged (+0.3 correlation) toward a root found another
    way: a sustained Em9 is in E minor to a producer, B minor to the bare
    profile. force=True takes the hint as the root and picks only the
    mode (one chord, one note, a bass line's home note).
    -> (root, mode, r, margin)."""
    import numpy as np
    from sample_audio import MAJOR, MINOR
    c = np.array(chroma)
    rs = []
    for k in range(12):
        for md, prof in (("major", MAJOR), ("minor", MINOR)):
            r = float(np.corrcoef(c, np.roll(prof, k))[0, 1])
            if force and k != root_hint:
                continue
            rs.append((r + (0.3 if k == root_hint else 0.0), r, k, md))
    rs.sort(reverse=True)
    return rs[0][2], rs[0][3], rs[0][1], rs[0][0] - rs[1][0]


def tonality(a):
    p = a.get("pitch") or {}
    s_flat = _clip((0.06 - (a.get("flatness") or 0)) / 0.06)
    s_perc = _clip((0.6 - (a.get("percussive") or 0)) / 0.5)
    s_ent = _clip((1.0 - (a.get("chroma_entropy") or 1)) / 0.15)
    s_voiced = _clip((p.get("voiced") or 0) / 0.5)
    return round(0.4 * s_flat + 0.25 * s_perc + 0.2 * s_ent + 0.15 * s_voiced, 3)


def bar_check(a, bpm, tail=False):
    """Does the file hold whole bars at bpm (half bars under six beats)?
    Also whole bars padded to a multiple of 65,536 frames (Homemade
    Weapons), or followed by a tail or silence. -> (exact, beats, loop
    seconds, how)."""
    d, frames, sr = a["duration"], a.get("frames", 0), a.get("sample_rate") or 44100
    if not bpm or not d:
        return False, None, None, None
    beat = 60 / bpm
    tol = 0.015 + 0.002 * d
    beats = d / beat
    n = round(beats / 4) * 4 if beats >= 6 else round(beats / 2) * 2   # whole bars (half bars for a fill)
    if n >= 2 and abs(d - n * beat) <= tol:
        return True, n, n * beat, "exact"
    n4 = int(beats // 4) * 4
    if frames and frames % 65536 == 0 and n4 >= 4 and 0 <= frames - n4 * beat * sr < 65536:
        return True, n4, n4 * beat, "padded"
    act = a.get("active")
    if act and n4 >= 4 and act <= n4 * beat + tol and (tail or (n4 + 4) * beat > d + tol):
        return True, n4, n4 * beat, "tail" if tail else "trailing silence"
    return False, round(beats, 2), None, None


def fold_to(b, target, triplets=False):
    """b times 1/2, 1, 2 (and 2/3, 3/2 with triplets) closest to target,
    and the factor."""
    best = (b, 1.0)
    for f in (0.5, 1.0, 2.0) + ((2 / 3, 1.5) if triplets else ()):
        if abs(b * f - target) < abs(best[0] - target):
            best = (b * f, f)
    return best


def groove(onsets, bpm, secs=20.0):
    """Swing and timing spread on the 16th grid, the lab's way: swing is the
    median lateness of off-16ths behind on-8ths; spread the robust spread
    of every hit around its own position's median. Up to 8 bars or 20 s."""
    import numpy as np
    if not onsets or not bpm:
        return None
    step = 15.0 / bpm
    ts = np.array([t for t, _ in onsets], dtype=float)
    st = np.array([s for _, s in onsets], dtype=float)
    sel = ts < ts[0] + min(secs, 8 * 16 * step)
    ts, st = ts[sel], st[sel]
    if len(ts) < 12:
        return None
    # phase: the 16th grid that best fits all hits, then re-centred on the on-8ths
    phis = np.arange(-step / 2, step / 2, 0.0005)
    cost = [np.sum(np.minimum(np.abs(ts - p - np.round((ts - p) / step) * step), 0.4 * step)) for p in phis]
    phi = phis[int(np.argmin(cost))]
    for _ in range(2):
        k = np.round((ts - phi) / step).astype(int)
        off = ts - phi - k * step
        even = (k % 2 == 0) & (np.abs(off) < 0.4 * step)
        if even.sum() < 6:
            return None
        if even.sum() >= 8:      # least squares on the on-8ths: phase and step together
            A = np.vstack([np.ones(even.sum()), k[even]]).T
            (phi_new, step_new), *_ = np.linalg.lstsq(A, ts[even], rcond=None)
            if abs(step_new - step) / step < 0.02:
                phi, step = phi_new, step_new
            else:
                phi += float(np.median(off[even]))
        else:
            phi += float(np.median(off[even]))
    k = np.round((ts - phi) / step).astype(int)
    off = ts - phi - k * step
    keep = np.abs(off) < 0.4 * step
    # the on-8ths carry the kicks and snares; a loop that starts on an
    # off-16th gets its grid shifted by one
    if st[keep & (k % 2 == 1)].sum() > 1.3 * st[keep & (k % 2 == 0)].sum():
        k = k + 1
    even, odd = keep & (k % 2 == 0), keep & (k % 2 == 1)
    if even.sum() < 6 or odd.sum() < 4:
        return {"groove_hits": int(keep.sum()), "groove_fit": round(float(keep.mean()), 3)}
    me, mo = float(np.median(off[even])), float(np.median(off[odd]))
    resid = np.concatenate([off[even] - me, off[odd] - mo])
    spread = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
    return {"swing_ms": round((mo - me) * 1000, 1), "spread_ms": round(spread * 1000, 1),
            "groove_hits": int(keep.sum()), "groove_fit": round(float(keep.mean()), 3),
            "off16_share": round(float(odd.sum() / keep.sum()), 3)}


def drum_tempo(onsets, tempos):
    """A drum loop's tempo from the detector's candidates. The top one
    stands unless a near-tied runner-up (score >= 0.95) lays the hits on a
    clearly tighter 16th grid (as many hits, under 40% of the spread): a
    live break read at 2/3 speed. Then the octave: under 100 it is doubled
    (the finer grid always fits, so this reads drum loops in the drum & bass
    range; SAMPLES.md, Weak spots); otherwise doubled when the faster grid fits as well and uses its
    off-16ths, halved when the slower grid fits clearly better. Groove is
    measured at both the tempo shown and half of it, so this choice doesn't
    hide a swung 16th."""
    b = tempos[0][0]
    if not onsets or len(onsets) < 12:
        return b
    g0 = groove(onsets, b) or {}
    for c, score in tempos[1:2]:
        if score < 0.95:
            break
        g1 = groove(onsets, c) or {}
        if g0.get("groove_fit") is not None and g1.get("groove_fit") is not None and \
                g1["groove_fit"] >= g0["groove_fit"] - 0.02 and (g1.get("spread_ms") or 99) <= 0.4 * (g0.get("spread_ms") or 0):
            b = c
    if b < 100:
        return b * 2
    g = {c: groove(onsets, c) or {} for c in (b / 2, b, b * 2) if 60 <= c <= 200}
    fit = {c: v.get("groove_fit") for c, v in g.items()}
    if fit.get(b) is None:
        return b
    if fit.get(b * 2) is not None and fit[b * 2] >= fit[b] - 0.1 and (g[b * 2].get("off16_share") or 0) >= 0.1:
        return b * 2
    if fit.get(b / 2) is not None and fit[b / 2] >= fit[b] + 0.1 and (g[b].get("off16_share") or 0) < 0.1:
        return b / 2
    return b


def audio_type(a, loop, tonal, ml):
    """A type from the sound alone (and AudioSet tags when present)."""
    tags = (ml or {}).get("tags") or {}
    b = a.get("bands") or {}
    low = (b.get("sub") or 0) + (b.get("low") or 0)
    top = (b.get("high") or 0) + (b.get("air") or 0)
    d = a.get("duration") or 0
    perc = a.get("percussive") or 0
    p = a.get("pitch") or {}
    voiced = p.get("voiced") or 0
    f0 = p.get("plateau") or p.get("median")
    kick_rate = a.get("kick_rate") or 0
    t_peak, dec40, dec20 = a.get("t_peak") or 0, a.get("decay40") or d, a.get("decay20") or d
    t = lambda *ls: max((tags.get(x, 0) for x in ls), default=0)  # noqa: E731
    keys = t("Piano", "Electric piano", "Keyboard (musical)", "Organ", "Electronic organ", "Hammond organ")
    if t(*sorted(VOCAL_TAGS)) >= 0.5:
        return "vocal"
    if d > 45 and kick_rate > 0.5 and 0.3 <= tonal and perc < 0.5:
        return "demo"
    if loop:
        if low > 0.8 and kick_rate >= 0.5 and voiced < 0.45:
            return "kick"                       # a kick pattern
        if low > 0.55 and voiced >= 0.35 and perc < 0.35:
            return "bass"
        if tonal < 0.45 or perc > 0.45:
            if kick_rate >= 0.4 and low > 0.05:
                return "break"
            if top >= 0.45 or t("Hi-hat", "Cymbal", "Tambourine", "Maraca", "Shaker") > 0.3:
                return "top"
            return "perc-loop" if perc > 0.35 else "fx"
        if f0 and f0 < 52 and low > 0.55:
            return "bass"
        if (a.get("onset_rate") or 0) < 1.0 and (a.get("sustain") or 0) > 0.6:
            return "pad"
        if keys > 0.4:
            return "keys"
        return "lead"
    # one-shots
    if (a.get("level_slope") or 0) > 4 and (a.get("centroid_slope") or 0) > 0.3 and d > 1.0:
        return "riser"
    if low >= 0.6 and t_peak <= 0.1 and voiced < 0.3 and (dec40 <= 0.8 or t("Bass drum", "Kick drum") >= 0.3):
        return "kick"                           # a fast low thump: basses are voiced, kicks barely
    if tonal < 0.45:
        if top > 0.55:
            return "cymbal" if dec40 > 0.8 else "hat"
        if t("Clapping", "Hands") > 0.3:
            return "clap"
        if t("Snare drum", "Rimshot") > 0.3 or ((b.get("lowmid") or 0) + (b.get("mid") or 0) > 0.5 and (a.get("flatness") or 0) > 0.05):
            return "snare"
        if d > 2.5 and (a.get("sustain") or 0) > 0.5:
            return "atmos" if (a.get("flatness") or 0) < 0.1 else "fx"
        return "perc"
    if t_peak <= 0.02 and dec20 < 0.25 and len(main_pcs(a.get("chroma"))) <= 2:
        return "perc"                           # a pitched hit: bottle, pipe, metal
    if f0 and f0 < 48 or low > 0.75:
        return "bass"
    if keys > 0.4:
        return "keys"
    if d > 3 and (a.get("sustain") or 0) > 0.6:
        return "pad"
    if len(main_pcs(a.get("chroma"))) >= 3 and d < 3:
        return "stab"
    return "lead"


def compatible(name_type, audio_type):
    """Whether two types describe the same sound closely enough: one family,
    a hit and a pattern of it, or an fx/atmos/vocal that may be tonal."""
    fa, fb = FAMILY.get(name_type), FAMILY.get(audio_type)
    if fa == fb:
        return True
    pair = {fa, fb}
    return pair in ({"drum-hit", "drum-loop"}, {"vocal", "tonal"}, {"vocal", "fx"}, {"fx", "tonal"}) or \
        (fa == "fx" and fb != "demo") or {name_type, audio_type} == {"stab", "perc"}


VOCAL_TAGS = {"Speech", "Male speech, man speaking", "Female speech, woman speaking", "Child speech, kid speaking",
              "Narration, monologue", "Singing", "Choir", "Chant", "Male singing", "Female singing",
              "Synthetic singing", "Rapping", "Humming", "Vocal music", "A capella", "Human voice", "Speech synthesizer"}


def assemble(root, rel, size, mtime, nm, a, ml):
    """One record: identity, the name's claims, the audio's measures, the
    resolved values with their sources, and the disagreements."""
    pack = rel.split(os.sep)[0]
    r = {"id": rel, "pack": pack, "rel": rel, "path": os.path.join(root, rel), "file": os.path.basename(rel),
         "title": nm["title"], "format": os.path.splitext(rel)[1][1:].lower(), "size": size, "mtime": mtime}
    flags = []
    for k in ("codec", "sample_rate", "channels", "bit_depth", "duration", "frames"):
        r[k] = a.get(k)
    r["name_type"], r["name_type_conf"] = nm["type"], nm["type_conf"]
    r["name_bpm"] = nm["bpm"]
    nk = nm["key"]
    r["name_key"] = key_str(nk["root"], nk["mode"]) if nk else None
    r["name_key_text"] = nk["text"] if nk else None
    r["name_key_kind"] = nk["kind"] if nk else None
    r["name_chord"] = nk.get("chord") if nk and nk["kind"] in ("chord", "progression") else None
    if nk and nk.get("chords"):
        r["name_chord"] = " ".join(nk["chords"])
    if a.get("silent"):
        r.update({"silent": True, "type": nm["type"], "loop": None, "flags": ["silent"], "keywords": nm["keywords"],
                  "words": nm["words"], "tags": [], "family": FAMILY.get(nm["type"])})
        return r
    # levels, tone
    for k in ("lufs", "peak_db", "rms_db", "crest_db", "centroid", "rolloff", "flatness", "width", "correlation",
              "percussive", "onset_rate", "kick_rate", "tuning_cents", "tuning_drift"):
        r[k] = a.get(k)
    for k, v in (a.get("bands") or {}).items():
        r["band_" + k] = v
    tags = (ml or {}).get("tags") or {}
    ton = tonality(a)
    r["tonality"] = ton
    r["tonal"] = ton >= 0.5

    # loop or one-shot, tempo
    tail = bool(re.search(r"\b(fx )?tail\b", nm["title"], re.I))
    n_on = len(a.get("onsets") or [])
    d = a["duration"]
    tempos = a.get("tempo") or []
    bpm_audio = tempos[0][0] if tempos else None
    if bpm_audio and (FAMILY.get(nm["type"]) in ("drum-loop", "drum-hit") or (a.get("percussive") or 0) >= 0.35):
        bpm_audio = round(drum_tempo(a.get("onsets"), tempos), 1)
    peak = a.get("tempo_peak") or 0
    r["bpm_audio"] = bpm_audio
    r["bpm_audio_conf"] = round(_clip((peak - 1.2) / 1.0), 2) if bpm_audio else None
    exact_n, beats_n, loop_n, how_n = bar_check(a, nm["bpm"], tail) if nm["bpm"] else (False, None, None, None)
    # tempo from the length alone: whole bars nearest the detected tempo
    bar_bpm = None
    if bpm_audio and d >= 1.0:
        cands = []
        for bars in range(1, 65):
            b = 240 * bars / d
            if 60 <= b <= 200:
                f_b, _ = fold_to(bpm_audio, b)
                # fit first (to 0.5%), then the octave the detector chose
                cands.append((round(abs(f_b - b) / b / 0.005), abs(math.log2(b / bpm_audio)), b))
        if cands:
            err, _, b = min(cands)
            if err * 0.005 < 0.03:
                bar_bpm = round(b, 2)
    r["bpm_bars"] = bar_bpm
    oneshot_audio = d < 0.9 or n_on <= 2 or (n_on <= 4 and d < 3 and (a.get("level_slope") or 0) < -6)
    name_loopy = nm["type"] in ("break", "top", "perc-loop")
    if nm["loop"] is not None:
        loop, loop_conf, loop_src = nm["loop"], 0.85, "name"
    elif exact_n and d >= 1.2:
        loop, loop_conf, loop_src = True, 0.9, "name tempo, whole bars"     # a held pad too
    elif name_loopy and not (d < 1.0 and n_on <= 2):
        loop, loop_conf, loop_src = True, 0.8, "name type"
    elif nm["bpm"] and d >= 1.2 and n_on >= 4:
        loop, loop_conf, loop_src = True, 0.7, "name tempo"
    elif oneshot_audio:
        loop, loop_conf, loop_src = False, 0.8, "audio"
    elif bar_bpm and n_on >= 6 and d >= 1.2 and (a.get("percussive") or 0) >= 0.25:
        loop, loop_conf, loop_src = True, 0.7, "audio, whole bars"
    elif n_on >= 8 and d >= 2 and (a.get("percussive") or 0) >= 0.3:
        loop, loop_conf, loop_src = True, 0.55, "audio"
    else:
        loop, loop_conf, loop_src = False, 0.5, "audio"
    r["loop"], r["loop_conf"], r["loop_source"] = loop, loop_conf, loop_src

    bpm, bpm_conf, bpm_src, agree = None, None, None, None
    if nm["bpm"] and bpm_audio:
        f_b, fac = fold_to(bpm_audio, nm["bpm"], triplets=True)
        if abs(f_b - nm["bpm"]) / nm["bpm"] <= 0.02:
            agree = "same" if fac == 1.0 else ("half/double" if fac in (0.5, 2.0) else "3:2")
        else:
            alts = [b for b, _ in tempos[1:]]
            agree = "alt" if any(abs(fold_to(b, nm["bpm"])[0] - nm["bpm"]) / nm["bpm"] <= 0.02 for b in alts) else "other"
    r["bpm_agree"] = agree
    if nm["bpm"]:
        bpm, bpm_src = nm["bpm"], "name"
        bpm_conf = 0.95 if exact_n else (0.9 if agree in ("same", "half/double") else 0.7)
        if loop and not exact_n and n_on >= 6:
            ex_a = bar_check(a, bar_bpm)[0] if bar_bpm else False
            if ex_a and abs(bar_bpm - nm["bpm"]) / nm["bpm"] > 0.01:
                flags.append(f"bpm: name {nm['bpm']:g}, but the length is whole bars at {bar_bpm:g}")
            elif agree in ("other", None) and (r["bpm_audio_conf"] or 0) >= 0.3:
                flags.append(f"bpm: name {nm['bpm']:g}, audio {bpm_audio:g}")
    elif loop and bpm_audio:
        if bar_bpm:
            bpm, bpm_conf, bpm_src = bar_bpm, 0.8, "audio + whole bars"
        else:
            bpm, bpm_conf, bpm_src = round(bpm_audio, 1), round(0.3 + 0.4 * (r["bpm_audio_conf"] or 0), 2), "audio"
    r["bpm"], r["bpm_conf"], r["bpm_source"] = bpm, bpm_conf, bpm_src
    if loop and bpm:
        ex, beats, loop_secs, how = bar_check(a, bpm, tail) if bpm != nm["bpm"] else (exact_n, beats_n, loop_n, how_n)
        r["bars_exact"] = bool(ex)
        r["bars"] = round((beats if ex else d * bpm / 60) / 4, 2)
        r["loop_secs"] = round(loop_secs, 4) if loop_secs else None
        r["bars_how"] = how
        if how == "padded":
            r["padded"] = True

    # type
    t_audio = audio_type(a, loop, ton, ml)
    r["audio_type"] = t_audio
    if nm["type"] and nm["type_conf"] >= 0.6:
        t, t_src, t_conf = nm["type"], "name", nm["type_conf"]
    else:
        t, t_src, t_conf = t_audio, "audio", 0.5
    if loop and t == "hat":          # loops of one-shot types become their loop types
        t = "top"
    if loop and t == "perc":
        t = "perc-loop"
    if not loop and t in ("top", "perc-loop") and d < 1.0:
        t = "hat" if t == "top" else "perc"
    r["type"], r["type_source"], r["type_conf"] = t, t_src, t_conf
    r["family"] = fam = FAMILY.get(t)
    if t_src == "name" and not compatible(nm["type"], t_audio):
        flags.append(f"type: name {nm['type']}, audio {t_audio}")

    # pitch: a one-shot note's own pitch; a chord's root; a key
    chroma, bass = a.get("chroma"), a.get("bass_chroma")
    r["chroma"] = chroma
    pcs = main_pcs(chroma) if chroma else []
    r["pcs"] = [NAMES12[i] for i in pcs]
    p = a.get("pitch") or {}
    drumish = fam in ("drum-loop", "drum-hit")
    chordal = t in ("pad", "chord", "stab") or (t is None and len(pcs) >= 3)
    is_note = (not loop and not chordal and ton >= 0.45 and "plateau" in p
               and (p.get("voiced", 0) >= 0.5 if t == "kick" else (not drumish and p.get("voiced", 0) >= 0.3)))
    root_hint = None
    if is_note:
        glide, curve = p["glide"], p.get("curve") or []
        mono = 0.0
        if len(curve) >= 4 and abs(glide) >= 1.0:
            import numpy as np
            cc = np.corrcoef(np.arange(len(curve)), curve)[0, 1]
            mono = 0.0 if np.isnan(cc) else abs(float(cc))
        r["pitch_start"], r["pitch_end"] = note_text(p["start"]), note_text(p["end"])
        r["pitch_start_midi"], r["pitch_end_midi"] = p["start"], p["end"]
        r["glide_st"] = glide
        r["pitch_wobble_cents"] = p.get("wobble")
        r["purity"] = p.get("purity")
        if abs(glide) >= 1.0 and mono >= 0.7:
            r["pitch_hold"] = "glide down" if glide < 0 else "glide up"
            root_midi = p["end"]          # where the note lands
        elif (p.get("range") or 0) > 1.5:
            r["pitch_hold"] = "moving"    # several notes: a phrase, not one pitch
            vals = [round(v) for v in curve]
            root_midi = max(set(vals), key=vals.count) if vals else p["plateau"]
        elif (p.get("wobble") or 0) > 40:
            r["pitch_hold"] = "wobbly"
            root_midi = p["plateau"]
        else:
            r["pitch_hold"] = "steady"
            root_midi = p["plateau"]
        r["root_audio"] = note_text(root_midi)
        r["root_audio_midi"] = round(root_midi, 2)
        r["root_audio_cents"] = round((root_midi - round(root_midi)) * 100)
        root_hint = int(round(root_midi)) % 12
    if chroma and not drumish and ton >= 0.45:
        cr, csym = chord_guess(chroma, bass)
        if chordal or (loop and fam == "tonal"):
            r["chord_audio"] = csym
        # where the root comes from, best first against the names (SAMPLES.md):
        # a note's own pitch, a bass line's longest steady note, a single
        # chord's root; loops of chords keep the profile, nudged
        force = True
        if root_hint is None:
            if t == "bass" and p.get("plateau") and p.get("voiced", 0) >= 0.3:
                root_hint = int(round(p["plateau"])) % 12
            elif chordal and (not loop or t == "pad"):      # a pad holds its chord, looped or not
                root_hint = cr
            else:
                root_hint, force = cr, False
        kr, km, kcorr, kmargin = audio_key(chroma, root_hint, force)
        if force:
            kmargin = 0.1
        r["key_audio"] = key_str(kr, km)
        r["key_audio_conf"] = round(_clip((kcorr - 0.3) / 0.5) * _clip(kmargin / 0.1) ** 0.5 * _clip((ton - 0.3) / 0.4), 2)
    else:
        kr = km = None
    if r.get("tuning_cents") is not None and ton >= 0.5 and (abs(r["tuning_cents"]) >= 15 or (r.get("tuning_drift") or 0) >= 25):
        r["detuned"] = True

    # agreement with the name
    if nk and not drumish:
        nroot, nmode = nk["root"], nk["mode"]

        def cents_to(midi):
            return abs(((midi - nroot + 6) % 12) - 6)
        if is_note and nk["kind"] in ("root", "key"):
            near = min(cents_to(x) for x in (p["start"], p["end"], r["root_audio_midi"]))
            r["key_agree"] = "same" if near <= 0.55 else ("near" if near <= 1.55 else "other")
            if r["key_agree"] != "same" and p.get("voiced", 0) >= 0.4:
                flags.append(f"root: name {NAMES12[nroot]}, audio {r['root_audio']}"
                             + (f" ({r['pitch_start']} to {r['pitch_end']})" if r.get("pitch_hold", "").startswith("glide") else ""))
        elif kr is not None:
            rel_ = key_relation((nroot, nmode), (kr, km))
            if nk.get("alts") and rel_ != "same":
                for ar, am in nk["alts"]:
                    if key_relation((ar, am), (kr, km)) == "same":
                        rel_ = "same (alt)"
            r["key_agree"] = rel_
            r["key_fit"] = scale_fit(chroma, nroot, nmode)
            best_fit = best_scale_fit(chroma)
            r["key_fits"] = r["key_fit"] is not None and r["key_fit"] >= best_fit - 0.03
            if not rel_.startswith(("same", "relative")) and not r["key_fits"] and (r.get("key_audio_conf") or 0) >= 0.25:
                flags.append(f"key: name {r['name_key']}, audio {r['key_audio']} ({rel_}; the name's scale holds "
                             f"{round(100 * r['key_fit'])}%, the best key's {round(100 * best_fit)}%)")
        if nk.get("pcs") and chroma and nk["kind"] == "chord":
            want = set(nk["pcs"])
            have = set(pcs[:max(3, len(want))])
            r["chord_overlap"] = round(len(want & have) / len(want | have), 2)
            if r["chord_overlap"] < 0.5 and r.get("chord_audio"):
                flags.append(f"chord: name {nk['chord']} ({' '.join(NAMES12[i] for i in sorted(want))}), "
                             f"audio {r.get('chord_audio')} ({' '.join(r['pcs'][:5])})")
    # resolved key / root: the name's when it has one, else the audio's
    if not drumish:
        if nk and (nk["kind"] in ("key", "progression") or (nk["kind"] in ("chord", "root") and not is_note)):
            r["key"], r["key_source"] = r["name_key"], "name"
            r["key_conf"] = 0.9 if (r.get("key_agree") or "").startswith(("same", "relative")) or r.get("key_fits") else 0.6
        elif kr is not None and not is_note and ton >= 0.5 and (r.get("key_audio_conf") or 0) >= 0.2:
            r["key"], r["key_source"], r["key_conf"] = r["key_audio"], "audio", r["key_audio_conf"]
        if is_note:
            if nk and nk["kind"] in ("root", "key"):
                r["root"], r["root_source"] = NAMES12[nk["root"]], "name"
            else:
                r["root"], r["root_source"] = NAMES12[int(round(r["root_audio_midi"])) % 12], "audio"
    # envelope
    for k in ("t_peak", "decay20", "decay40", "sustain", "level_slope", "centroid_slope", "active"):
        r[k] = a.get(k)
    # content
    vocal_ast = (ml or {}).get("vocal")
    r["vocal_ast"] = vocal_ast[0] if vocal_ast else None
    r["vocal_ast_label"] = vocal_ast[1] if vocal_ast else None
    vocal_name = nm["type"] == "vocal" or any(k in nm["keywords"] for k in ("vocal", "choir"))
    vocal_audio = bool(vocal_ast and vocal_ast[0] >= AST_VOCAL)
    r["vocal"] = vocal_name or vocal_audio
    r["vocal_source"] = "name+audio" if vocal_name and vocal_audio else ("name" if vocal_name else ("audio" if vocal_audio else None))
    r["kick"] = (a.get("kick_rate") or 0) >= 0.3 if loop else None
    r["tags"] = [[k, v] for k, v in tags.items() if v >= 0.1 and k not in ("Music", "Musical instrument")][:10]
    # drums under a melodic part: AudioSet hears a kit, or a drum genre with
    # kicks and percussive energy, or a whole track keeps a steady kick. Low
    # pad and stab notes set off the kick count, and gated stabs look
    # percussive, so neither decides by itself.
    drum_tag = max([tags.get(k, 0) for k in ("Drum kit", "Drum", "Snare drum", "Drum machine", "Hi-hat", "Percussion",
                                             "Drum roll", "Bass drum", "Cymbal")] + [0])
    genre = max([tags.get(k, 0) for k in ("Drum and bass", "Hip hop music", "Breakbeat", "Dubstep", "Techno", "House music")] + [0])
    kr, pc = a.get("kick_rate") or 0, a.get("percussive") or 0
    r["drums"] = bool((fam in ("tonal", "vocal") or t in ("demo",) or t is None) and t != "bass" and n_on >= 6 and
                      (loop or d >= 4) and
                      (drum_tag >= 0.25 or (genre >= 0.15 and kr >= 1.0 and pc >= 0.1)
                       or (t in ("demo", None) and d >= 20 and kr >= 1.0)))     # a whole track with a steady kick
    if r["drums"] and fam in ("tonal", "vocal"):
        flags.append("drums under the melodic part")
    if t == "top" and r["kick"]:
        flags.append("kick in a top loop")
    # groove: drum loops, and loops with drums in them
    if loop and bpm and (fam == "drum-loop" or r["drums"] or t == "demo"):
        # two levels: 16ths at the tempo shown, and 8ths (the 16th grid at half
        # tempo), where a 94 BPM break shown at 188 keeps its swung 16ths
        levels = {"16th": groove(a.get("onsets"), bpm), "8th": groove(a.get("onsets"), bpm / 2) if bpm / 2 >= 50 else None}
        best = None
        for lv, g in levels.items():
            if not g:
                continue
            if g.get("swing_ms") is not None:
                r[f"swing{lv[:-2]}_ms"], r[f"spread{lv[:-2]}_ms"] = g["swing_ms"], g["spread_ms"]
            r[f"fit{lv[:-2]}"] = g.get("groove_fit")
            if g.get("swing_ms") is not None and (g.get("groove_fit") or 0) >= 0.85 and \
                    (best is None or g["swing_ms"] > best[1]["swing_ms"]):
                best = (lv, g)
        g16 = levels["16th"] or {}
        r["groove_hits"], r["off16_share"] = g16.get("groove_hits"), g16.get("off16_share")
        if best:
            r["swing_ms"], r["spread_ms"], r["groove_fit"], r["swing_level"] = \
                best[1]["swing_ms"], best[1]["spread_ms"], best[1]["groove_fit"], best[0]
    r["width_class"] = "mono" if (a.get("channels") == 1 or (a.get("width") or 0) < 0.03) else ("wide" if (a.get("width") or 0) > 0.45 else "stereo")
    # keywords: the name's, plus what was measured
    derived = [t, fam, "loop" if loop else "oneshot", "tonal" if r["tonal"] else "atonal"]
    if r["vocal"]:
        derived.append("vocal")
    if r.get("pitch_hold"):
        derived.append(r["pitch_hold"].replace(" ", "-"))
    if r.get("purity") and r["purity"] >= 0.8:
        derived += ["clean", "sine"]
    if fam == "drum-loop":
        derived.append("kick" if r["kick"] else "no-kick")
    if r.get("swing_ms") is not None:
        derived.append("swung" if r["swing_ms"] >= 6 else ("straight" if abs(r["swing_ms"]) < 3 else "light-swing"))
    if r.get("detuned"):
        derived.append("detuned")
    if r["width_class"] != "stereo":
        derived.append(r["width_class"])
    if (r.get("rolloff") or 20000) < 9000 and fam in ("drum-loop", "drum-hit"):
        derived.append("lofi")
    r["keywords"] = list(dict.fromkeys(nm["keywords"] + [x for x in derived if x]))
    r["words"] = nm["words"]
    r["flags"] = flags
    if p.get("curve") and is_note:
        r["pitch_curve"] = p["curve"]
    return r


def classes(records):
    """Library-relative classes: brightness by type and loop (thirds of
    centroid), energy by family (thirds of loudness, plus hit density for
    drum loops)."""
    import numpy as np
    live = [r for r in records if not r.get("silent") and r.get("centroid")]
    groups = {}
    for r in live:
        groups.setdefault((r["type"], r["loop"]), []).append(r)
    fam = {}
    for r in live:
        fam.setdefault((r["family"], r["loop"]), []).append(r)
    for key, rs in groups.items():
        pool = rs if len(rs) >= 15 else fam[(rs[0]["family"], rs[0]["loop"])]
        c = np.log([x["centroid"] for x in pool])
        lo, hi = np.percentile(c, [33.3, 66.7])
        for r in rs:
            v = math.log(r["centroid"])
            r["tone"] = "dark" if v < lo else ("bright" if v > hi else "neutral")
            r["brightness"] = round(float(np.mean(c <= v)), 2)
    for key, rs in fam.items():
        lu = np.array([x["lufs"] if x.get("lufs") is not None else -40 for x in rs])
        on = np.array([x.get("onset_rate") or 0 for x in rs])
        z = (lu - lu.mean()) / (lu.std() or 1)
        if key[0] == "drum-loop":
            z = 0.5 * z + 0.5 * (on - on.mean()) / (on.std() or 1)
        lo, hi = np.percentile(z, [33.3, 66.7])
        for r, v in zip(rs, z):
            r["energy"] = "low" if v < lo else ("high" if v > hi else "medium")
    # the top end on its own: air (10-20 kHz) against high (4-10 kHz), so a
    # break with a heavy kick can still have a bright top, or a dull one
    for key, rs in groups.items():
        pool = rs if len(rs) >= 15 else fam[(rs[0]["family"], rs[0]["loop"])]
        ratio = lambda x: (x.get("band_air") or 0) / max((x.get("band_high") or 0) + (x.get("band_air") or 0), 1e-6)  # noqa: E731
        v_all = np.array([ratio(x) for x in pool])
        lo, hi = np.percentile(v_all, [33.3, 66.7])
        for r in rs:
            v = ratio(r)
            r["top"] = "dark" if v < lo else ("bright" if v > hi else "neutral")
    for r in live:
        r["keywords"] = list(dict.fromkeys(r["keywords"] + [r.get("tone"), f"{r.get('energy')}-energy", f"{r.get('top')}-top"]))


def text_embeddings(idx, texts):
    """CLAP text embeddings, cached in text-cache.json (a query costs a
    model load the first time, ~5 s)."""
    import numpy as np
    path = os.path.join(idx, "text-cache.json")
    cache = _read_json(path) or {}
    missing = [t for t in texts if t not in cache]
    if missing:
        py = venv_python(idx)
        if not py:
            raise SystemExit("free-text search needs the model venv: samples.py models --setup")
        env = dict(os.environ, HF_HOME=os.path.join(idx, "hf"), TOKENIZERS_PARALLELISM="false")
        out = subprocess.run([py, os.path.join(HERE, "sample_models.py"), "text", *missing], env=env,
                             capture_output=True, text=True)
        if out.returncode != 0:
            raise SystemExit(out.stderr.strip().splitlines()[-1] if out.stderr.strip() else "text embedding failed")
        for t, e in zip(missing, json.loads(out.stdout)):
            cache[t] = e
        _write_json(path, cache)
    return np.array([cache[t] for t in texts], dtype=np.float32)


def build(root, idx, quiet=False):
    """Assemble index.json (and clap.npy) from the caches."""
    import numpy as np
    files = scan(root)
    ml_entries = {}
    for rel, path, size, mtime in files:
        e = _read_json(os.path.join(idx, "cache", "models", _key(rel) + ".json"))
        if _fresh(e, size, mtime, MODELS_VERSION) and "clap" in e:
            ml_entries[rel] = e
    records, errors, ids, embs = [], [], [], []
    for rel, path, size, mtime in files:
        e = _read_json(os.path.join(idx, "cache", "audio", _key(rel) + ".json"))
        nm = N.parse(os.path.splitext(os.path.basename(rel))[0], rel.split(os.sep)[0])
        if not _fresh(e, size, mtime, AUDIO_VERSION) or "error" in (e or {}):
            err = (e or {}).get("error", "not analyzed")
            errors.append([rel, err])
            records.append({"id": rel, "pack": rel.split(os.sep)[0], "rel": rel, "path": path, "file": os.path.basename(rel),
                            "title": nm["title"], "format": os.path.splitext(rel)[1][1:].lower(), "type": nm["type"],
                            "name_type": nm["type"], "name_bpm": nm["bpm"], "bpm": nm["bpm"], "keywords": nm["keywords"],
                            "words": nm["words"], "flags": [f"not analyzed: {err}"], "tags": [], "loop": nm["loop"],
                            "family": FAMILY.get(nm["type"])})
            continue
        ml = ml_entries.get(rel)
        rec = assemble(root, rel, size, mtime, nm, e["features"], ml)
        records.append(rec)
        if ml:
            ids.append(rel)
            embs.append(ml["clap"])
    classes(records)
    out = {"version": INDEX_VERSION, "root": root, "built": time.strftime("%Y-%m-%d %H:%M:%S"), "count": len(records),
           "errors": errors, "models": bool(ml_entries), "records": records}
    _write_json(os.path.join(idx, "index.json"), out)
    if embs:
        np.save(os.path.join(idx, "clap.npy"), np.array(embs, dtype=np.float32))
        _write_json(os.path.join(idx, "clap-ids.json"), ids)
    _write_json(os.path.join(idx, "config.json"), {"root": root})
    if not quiet:
        print(f"index: {len(records)} records ({len(errors)} not analyzed), {len(embs)} with CLAP embeddings -> "
              f"{os.path.join(idx, 'index.json')}")
    return out


# ── queries ───────────────────────────────────────────────────────────────
TYPE_ALIASES = {"drumloop": "break", "drums": "break", "breaks": "break", "beat": "break",
                "hat-loop": "top", "top-loop": "top", "tops": "top", "hats": "top", "perc-loop": "perc-loop",
                "percussion": "perc", "impact": "riser", "texture": "atmos", "ambient": "atmos", "sfx": "fx",
                "synth": "lead", "riff": "lead", "piano": "keys", "rhodes": "keys", "voice": "vocal", "vox": "vocal"}
FAMILIES = {"drum-loop", "drum-hit", "tonal", "fx"}


def _parse_key(s):
    m = re.fullmatch(r"\s*([A-Ga-g])([#b]?)\s*(m|min|minor|maj|major|M)?\s*", s or "")
    if not m:
        raise ValueError(f"key {s!r}: like Fm, F minor, C#, Ab major")
    root = N.pitch_class(m.group(1).upper() + m.group(2))
    q = m.group(3)
    mode = None if not q else ("minor" if q.lower().startswith("m") and q not in ("M", "maj", "major") else "major")
    return root, mode


def _rec_key(r, field):
    """(root, mode) of a record's key field; a bare root from the name
    ("_E") carries no mode, also once resolved into `key`."""
    v = r.get(field)
    if not v:
        return None
    root = N.pitch_class(re.match(r"[A-G]#?", v).group(0))
    if v.endswith("m"):
        return root, "minor"
    if r.get("name_key_kind") == "root" and (field == "name_key" or (field == "key" and r.get("key_source") == "name")):
        return root, None
    return root, "major"


def _norm_word(w):
    w = w.lower()
    return N._SHORT.get(w) or N._LONG.get(w) or w


def _where(expr):
    m = re.fullmatch(r"\s*([a-z_0-9]+)\s*(<=|>=|!=|=|<|>)\s*(.+?)\s*", expr)
    if not m:
        raise ValueError(f"--where {expr!r}: like lufs>-10 or band_high<0.1")
    f, op, v = m.groups()
    try:
        val = float(v)
    except ValueError:
        val = v
    ops = {"<": lambda a, b: a < b, ">": lambda a, b: a > b, "<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b,
           "=": lambda a, b: a == b, "!=": lambda a, b: a != b}

    def test(r):
        x = r.get(f)
        if x is None:
            return False
        if isinstance(val, float) and not isinstance(x, (int, float)):
            return False
        return ops[op](x, val)
    return test


class Library:
    def __init__(self, data, idx):
        self.idx = idx
        self.data = data
        self.records = data["records"]
        self.by_id = {r["id"]: r for r in self.records}
        self.by_path = {r["path"]: r for r in self.records}
        self._clap = None

    def __len__(self):
        return len(self.records)

    def get(self, ref):
        """A record by id (path under the root), absolute path, or a unique
        title fragment."""
        if isinstance(ref, dict):
            return ref
        if ref in self.by_id:
            return self.by_id[ref]
        p = os.path.abspath(os.path.expanduser(ref))
        if p in self.by_path:
            return self.by_path[p]
        hits = [r for r in self.records if ref.lower() in r["rel"].lower()]
        if len(hits) == 1:
            return hits[0]
        raise KeyError(f"{ref!r}: {'no match' if not hits else f'{len(hits)} matches'}")

    def find(self, *words, type=None, loop=None, bpm=None, bpm_tol=None, fold=False, key=None, key_source="any",
             has=(), lacks=(), root=None, tonal=None, vocal=None, kick=None, drums=None, steady=None, glide=None,
             clean=None, swing=None, max_swing=None, spread=None, tone=None, top=None, energy=None, width=None, pack=None,
             min_dur=None, max_dur=None, flagged=None, where=(), sort=None, limit=25, pool=None):
        """Records matching every filter given, best first. words match the
        keywords, title words, AudioSet tags, type and pack."""
        if isinstance(where, str):
            where = [where]
        if isinstance(has, str):
            has = [has]
        if isinstance(lacks, str):
            lacks = [lacks]
        tests = []
        if type:
            types = set()
            fams = set()
            for t in (type if isinstance(type, (list, tuple, set)) else str(type).split(",")):
                t = t.strip().lower()
                if t in ("bass-loop", "bass-oneshot"):
                    types.add("bass")
                    loop = t == "bass-loop" if loop is None else loop
                    continue
                if t in FAMILIES:
                    fams.add(t)
                    continue
                t = TYPE_ALIASES.get(t, t)
                if t not in N.TYPES:
                    raise ValueError(f"type {t!r}: one of {', '.join(N.TYPES)}, or a family {', '.join(sorted(FAMILIES))}")
                types.add(t)
            tests.append(lambda r: r.get("type") in types or r.get("family") in fams)
        if loop is not None:
            tests.append(lambda r: bool(r.get("loop")) == bool(loop))
        def bpm_cands(r):
            # a detected tempo's octave is a guess (87 or 174?); a named one isn't
            b = r.get("bpm") or 0
            return [b, b * 2, b / 2] if fold or r.get("bpm_source") != "name" else [b]

        if bpm:
            tol = bpm_tol if bpm_tol is not None else max(1.0, bpm * 0.012)

            def bpm_ok(r):
                b = r.get("bpm")
                if not b:
                    return False
                return any(abs(c - bpm) <= tol for c in bpm_cands(r))
            tests.append(bpm_ok)
        if key:
            want = _parse_key(key)

            def key_ok(r):
                fields = {"any": ("key", "name_key", "key_audio"), "name": ("name_key",), "audio": ("key_audio",),
                          "resolved": ("key",)}[key_source]
                for f in fields:
                    k = _rec_key(r, f)
                    if k and k[0] == want[0] and (not want[1] or not k[1] or k[1] == want[1]):
                        if f == "key_audio" and (r.get("key_audio_conf") or 0) < 0.2:
                            continue
                        return True
                return False
            tests.append(key_ok)
        for pc in has:
            v = N.pitch_class(pc[0].upper() + pc[1:])
            tests.append(lambda r, v=v: NAMES12[v] in (r.get("pcs") or []))
        for pc in lacks:
            v = N.pitch_class(pc[0].upper() + pc[1:])
            tests.append(lambda r, v=v: bool(r.get("pcs")) and NAMES12[v] not in r["pcs"])
        if root:
            m = re.fullmatch(r"([A-Ga-g][#b]?)(-?\d)?", root)
            if not m:
                raise ValueError(f"root {root!r}: like F#, B0")
            rpc = N.pitch_class(m.group(1)[0].upper() + m.group(1)[1:])
            octave = int(m.group(2)) if m.group(2) else None

            def root_ok(r):
                if r.get("root") is None:
                    return False
                if N.pitch_class(r["root"]) != rpc:
                    return False
                return octave is None or (r.get("root_audio_midi") is not None and int(round(r["root_audio_midi"])) // 12 - 1 == octave)
            tests.append(root_ok)
        for field, want_v in (("tonal", tonal), ("vocal", vocal), ("kick", kick), ("drums", drums)):
            if want_v is not None:
                tests.append(lambda r, f=field, w=want_v: bool(r.get(f)) == bool(w))
        if steady:
            tests.append(lambda r: r.get("pitch_hold") == "steady")
        if glide:
            tests.append(lambda r: (r.get("pitch_hold") or "").startswith("glide"))
        if clean:
            tests.append(lambda r: (r.get("purity") or 0) >= (0.8 if clean is True else float(clean)))
        # groove numbers only count where the grid holds the hits (swing_ms is
        # set only then: the larger of the 16th and 8th levels)
        fits = lambda r: r.get("swing_ms") is not None  # noqa: E731
        if swing is not None:
            tests.append(lambda r: fits(r) and r["swing_ms"] >= swing)
        if max_swing is not None:
            tests.append(lambda r: fits(r) and r["swing_ms"] <= max_swing)
        if spread is not None:
            tests.append(lambda r: fits(r) and r["spread_ms"] >= spread)
        if tone:
            tests.append(lambda r: r.get("tone") == tone)
        if top:
            tests.append(lambda r: r.get("top") == top)
        if energy:
            tests.append(lambda r: r.get("energy") == energy)
        if width:
            tests.append(lambda r: r.get("width_class") == width)
        if pack:
            tests.append(lambda r: pack.lower() in r["pack"].lower())
        if min_dur is not None:
            tests.append(lambda r: (r.get("duration") or 0) >= min_dur)
        if max_dur is not None:
            tests.append(lambda r: (r.get("duration") or 0) <= max_dur)
        if flagged is not None:
            tests.append(lambda r: bool(r.get("flags")) == bool(flagged))
        for w in where:
            tests.append(_where(w))
        for w in words:
            for part in w.split():
                nw = _norm_word(part)

                def word_ok(r, nw=nw, raw=part.lower()):
                    if nw in r.get("keywords", ()) or raw in r.get("words", ()) or nw == r.get("type"):
                        return True
                    if raw in r["title"].lower() or raw in r["pack"].lower():
                        return True
                    return any(raw in t.lower() for t, p in r.get("tags", ()) if p >= 0.2)
                tests.append(word_ok)
        hits = [r for r in (pool if pool is not None else self.records) if all(t(r) for t in tests)]
        if sort:
            f = sort.lstrip("-")
            have = sorted((r for r in hits if r.get(f) is not None), key=lambda r: r[f], reverse=sort.startswith("-"))
            hits = have + [r for r in hits if r.get(f) is None]
        else:
            def rank(r):
                k = []
                if bpm:
                    k.append(min(abs(c - bpm) for c in bpm_cands(r)))
                if top == "dark":
                    k.append((r.get("band_air") or 0) / max((r.get("band_high") or 0) + (r.get("band_air") or 0), 1e-6))
                elif top == "bright":
                    k.append(-(r.get("band_air") or 0) / max((r.get("band_high") or 0) + (r.get("band_air") or 0), 1e-6))
                if tone == "dark":
                    k.append(r.get("brightness") or 0)
                elif tone == "bright":
                    k.append(-(r.get("brightness") or 0))
                if swing is not None:
                    k.append(-(r.get("swing_ms") or 0))
                if clean:
                    k.append(-(r.get("purity") or 0))
                k.append(len(r.get("flags") or ()))
                k.append(r["rel"])
                return k
            hits.sort(key=rank)
        return hits[:limit] if limit else hits

    def _clap_matrix(self):
        import numpy as np
        if self._clap is None:
            ids = _read_json(os.path.join(self.idx, "clap-ids.json")) or []
            mat = np.load(os.path.join(self.idx, "clap.npy")) if ids else np.zeros((0, 512), np.float32)
            self._clap = (ids, mat)
        return self._clap

    def _ranked(self, q, limit, filters):
        import numpy as np
        ids, mat = self._clap_matrix()
        if not ids:
            raise SystemExit("no CLAP embeddings in this index: samples.py models --setup, then samples.py index")
        sims = mat @ q
        pool = None
        if filters:
            pool = self.find(limit=0, **filters)
            ok = {r["id"] for r in pool}
            order = [i for i in np.argsort(-sims) if ids[i] in ok]
        else:
            order = list(np.argsort(-sims))
        out = []
        for i in order[:limit]:
            r = dict(self.by_id[ids[i]])
            r["similarity"] = round(float(sims[i]), 3)
            out.append(r)
        return out

    def text(self, query, limit=25, **filters):
        """Records whose sound best matches a description (CLAP)."""
        q = text_embeddings(self.idx, [query])[0]
        return self._ranked(q, limit, filters)

    def like(self, ref, limit=25, **filters):
        """Records that sound most like a given one (CLAP)."""
        ids, mat = self._clap_matrix()
        r = self.get(ref)
        if r["id"] not in ids:
            raise SystemExit(f"{r['title']}: no CLAP embedding (samples.py models --setup, then samples.py index)")
        i = ids.index(r["id"])
        out = self._ranked(mat[i], limit + 1, filters)
        return [x for x in out if x["id"] != r["id"]][:limit]


def load(index=None):
    """The library from index.json (build it with `samples.py index`)."""
    idx = index_dir(index)
    data = _read_json(os.path.join(idx, "index.json"))
    if not data:
        raise SystemExit(f"no index at {idx}: run samples.py index --root <samples folder>")
    return Library(data, idx)


def playable(rec, index=None):
    """A path Slab can open: FLAC and WAV as they are, anything else (MP3)
    decoded once to a WAV in the index's wav/ folder."""
    path = rec["path"] if isinstance(rec, dict) else rec
    if path.lower().endswith((".flac", ".wav")):
        return path
    out = os.path.join(index_dir(index), "wav", _key(path) + ".wav")
    if not os.path.exists(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-map", "0:a:0", "-c:a", "pcm_s24le", out + ".part.wav"],
                       check=True)
        os.replace(out + ".part.wav", out)
    return out


# ── output ────────────────────────────────────────────────────────────────
def _short_pack(p):
    p = re.sub(r"^(Fanu - FatGyver - |Total Science - (Total Science - )?)", "", p)
    return {"Beats- Bass and Pads Sample Pack 2": "TotalSci 2023", "Beats- Bass and Pads Sample Pack": "TotalSci 2020",
            "D&B BREAKS SAMPLE PACK by Fanu": "Fanu DnB Breaks", "FatGyver's Hip Hop Breaks Sample Pack": "FatGyver HipHop",
            "Sample Pack by Fanu Vol. 1": "Fanu Vol.1"}.get(p, p.split(" - ")[0][:15])


def line(r):
    t = (r.get("type") or "?")[:9]
    lp = "loop" if r.get("loop") else "shot"
    bars = f"{r['bars']:g}b" if r.get("loop") and r.get("bars") else ""
    bpm = (f"{r['bpm']:.1f}" + ("" if r.get("bpm_source") == "name" else "~")) if r.get("bpm") else ""
    if r.get("root"):
        k = f"{r['root']}" + (f" {r['pitch_hold']}" if r.get("pitch_hold") else "")
    else:
        k = r.get("key") or ""
        if r.get("key_source") == "name" and r.get("key_audio") and r.get("key_agree") not in ("same", "same (alt)"):
            k += f"/{r['key_audio']}?"
    groove = (f"sw{r['swing_ms']:+.0f}{'/8' if r.get('swing_level') == '8th' else ''} sp{r['spread_ms']:.0f}"
              if r.get("swing_ms") is not None else "")
    lu = f"{r['lufs']:.0f}LU" if r.get("lufs") is not None else ""
    sim = f"{r['similarity']:.2f} " if "similarity" in r else ""
    fl = "!" if r.get("flags") else " "
    extra = " vocal" if r.get("vocal") else ""
    return (f"{sim}{t:9} {lp} {bars:>4} {bpm:>7} {k:16.16} {lu:>5} {r.get('tone') or '':7} {r.get('energy') or '':6} "
            f"{groove:12} {fl} {_short_pack(r['pack'])[:16]:16} {r['title'][:60]}{extra}")


def show(r, full=False):
    skip = set() if full else {"chroma", "pitch_curve", "path", "id", "rel"}
    print(f"\n{r['path']}")
    for k, v in r.items():
        if k in skip or v is None or v == [] or v == "":
            continue
        print(f"  {k:18} {json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v}")


# ── validation ────────────────────────────────────────────────────────────
def check(lib, root):
    import collections
    rs = lib.records
    on_disk = len(scan(root))
    print(f"files on disk {on_disk}, indexed {len(rs)}, not analyzed {len(lib.data.get('errors') or [])}, "
          f"with AudioSet tags {sum(1 for r in rs if r.get('vocal_ast') is not None)}\n")
    packs = collections.defaultdict(list)
    for r in rs:
        packs[r["pack"]].append(r)
    print(f"{'pack':22} {'files':>5} | {'tempo named':>11} {'same':>5} {'x2':>4} {'whole':>6} | "
          f"{'keys named':>10} {'same':>5} {'rel':>4} {'5th':>4} {'fits':>5} | {'roots':>5} {'same':>5} | {'type':>5} {'agree':>5}")
    tot = collections.Counter()
    for pack, xs in sorted(packs.items()):
        c = collections.Counter()
        for r in xs:
            if r.get("name_bpm") and r.get("loop") and r.get("bpm_agree"):
                c["t"] += 1
                c["t_same"] += r["bpm_agree"] == "same"
                c["t_oct"] += r["bpm_agree"] == "half/double"
                c["t_bars"] += bool(r.get("bars_exact"))
            ka = r.get("key_agree")
            if ka and r.get("name_key_kind") in ("key", "chord", "progression") or (ka and r.get("name_key_kind") == "root" and not r.get("root")):
                c["k"] += 1
                c["k_same"] += ka.startswith("same")
                c["k_rel"] += ka == "relative"
                c["k_5"] += ka == "fifth"
                c["k_fit"] += bool(r.get("key_fits"))
            if ka and r.get("root") and r.get("name_key_kind") in ("root", "key"):
                c["r"] += 1
                c["r_same"] += ka == "same"
            if r.get("name_type") and r.get("name_type_conf", 0) >= 0.6 and r.get("audio_type"):
                c["ty"] += 1
                c["ty_ok"] += compatible(r["name_type"], r["audio_type"])
        tot.update(c)

        def pct(a, b):
            return f"{100 * c[a] / c[b]:.0f}%" if c[b] else "-"
        print(f"{_short_pack(pack)[:22]:22} {len(xs):5} | {c['t']:11} {pct('t_same', 't'):>5} {pct('t_oct', 't'):>4} "
              f"{pct('t_bars', 't'):>6} | {c['k']:10} {pct('k_same', 'k'):>5} {pct('k_rel', 'k'):>4} {pct('k_5', 'k'):>4} "
              f"{pct('k_fit', 'k'):>5} | {c['r']:5} {pct('r_same', 'r'):>5} | {c['ty']:5} {pct('ty_ok', 'ty'):>5}")
    c = tot

    def pct(a, b):
        return f"{100 * c[a] / c[b]:.0f}%" if c[b] else "-"
    print(f"{'all':22} {len(rs):5} | {c['t']:11} {pct('t_same', 't'):>5} {pct('t_oct', 't'):>4} {pct('t_bars', 't'):>6} | "
          f"{c['k']:10} {pct('k_same', 'k'):>5} {pct('k_rel', 'k'):>4} {pct('k_5', 'k'):>4} {pct('k_fit', 'k'):>5} | "
          f"{c['r']:5} {pct('r_same', 'r'):>5} | {c['ty']:5} {pct('ty_ok', 'ty'):>5}")
    print("\ntempo: loops with a tempo in the name; same = detected within 2%, x2 = at half or double, whole = the"
          "\nlength is whole bars at the named tempo. keys: same includes either alternative the name offers;"
          "\nrel = relative major/minor; fits = the name's scale holds the notes as well as any key's (within 3%)."
          "\nroots: tonal one-shots, the note they start or land on, within 55 cents. type: name and audio types"
          "\ncompatible: one family (drum loop, drum hit, tonal, fx, vocal), a hit and a pattern of it, or an"
          "\nfx/atmos/vocal name on tonal audio.")
    flagged = collections.Counter(f.split(":")[0] for r in rs for f in r.get("flags") or ())
    print("\nflags: " + ", ".join(f"{k} {v}" for k, v in flagged.most_common()))


# ── command line ──────────────────────────────────────────────────────────
def _filters(a):
    f = {}
    for k in ("type", "bpm", "bpm_tol", "key", "key_source", "root", "swing", "max_swing", "spread", "tone", "top", "energy",
              "width", "pack", "min_dur", "max_dur", "sort", "limit"):
        v = getattr(a, k, None)
        if v is not None:
            f[k] = v
    if a.fold:
        f["fold"] = True
    if a.has:
        f["has"] = a.has
    if a.lacks:
        f["lacks"] = a.lacks
    if a.where:
        f["where"] = a.where
    for k in ("loop", "tonal", "vocal", "kick", "drums", "flagged"):
        v = getattr(a, k)
        if v is not None:
            f[k] = v
    for k in ("steady", "glide", "clean"):
        if getattr(a, k):
            f[k] = True
    return f


def _filter_args(p):
    g = p.add_argument_group("filters")
    g.add_argument("--type", help="break top perc-loop kick snare clap hat cymbal perc bass pad chord stab lead keys "
                                   "vocal fx riser atmos demo; bass-loop, bass-oneshot; families drum-loop drum-hit "
                                   "tonal fx; comma-separated")
    g.add_argument("--loop", dest="loop", action="store_const", const=True)
    g.add_argument("--oneshot", dest="loop", action="store_const", const=False)
    g.add_argument("--bpm", type=float)
    g.add_argument("--bpm-tol", type=float, help="default 1.2%%")
    g.add_argument("--fold", action="store_true", help="also match half and double tempo")
    g.add_argument("--key", help="Fm, F minor, C#, Ab major (matches the name's or the audio's key)")
    g.add_argument("--key-source", choices=("any", "name", "audio", "resolved"))
    g.add_argument("--has", action="append", metavar="PC", help="pitch class among the main ones (repeatable)")
    g.add_argument("--lacks", action="append", metavar="PC", help="pitch class absent (repeatable)")
    g.add_argument("--root", help="a one-shot's note: F#, or B0 with octave")
    g.add_argument("--steady", action="store_true", help="one-shot holds its pitch")
    g.add_argument("--glide", action="store_true", help="one-shot slides")
    g.add_argument("--clean", action="store_true", help="nearly pure tone (fundamental >= 80%% of the energy)")
    g.add_argument("--swing", type=float, metavar="MS",
                   help="off-16ths at least MS behind the on-8ths (at the tempo or half of it)")
    g.add_argument("--max-swing", type=float, metavar="MS")
    g.add_argument("--spread", type=float, metavar="MS", help="timing spread at least MS")
    g.add_argument("--tone", choices=("dark", "neutral", "bright"), help="overall brightness, thirds within the type")
    g.add_argument("--top", choices=("dark", "neutral", "bright"), help="the top end alone: air against highs")
    g.add_argument("--energy", choices=("low", "medium", "high"))
    g.add_argument("--width", choices=("mono", "stereo", "wide"))
    for name, hlp in (("tonal", "pitched material"), ("vocal", "a voice"), ("kick", "kick hits"),
                      ("drums", "drums under a melodic part"), ("flagged", "name and audio disagree")):
        g.add_argument(f"--{name}", dest=name, action="store_const", const=True, help=hlp)
        g.add_argument(f"--no-{name}", dest=name, action="store_const", const=False)
    g.add_argument("--atonal", dest="tonal", action="store_const", const=False)
    g.add_argument("--pack")
    g.add_argument("--min-dur", type=float)
    g.add_argument("--max-dur", type=float)
    g.add_argument("--where", action="append", metavar="EXPR", help="any numeric field: lufs>-10, band_air<0.05")
    g.add_argument("--sort", metavar="FIELD", help="a field; --sort=-FIELD for descending")
    g.add_argument("--limit", type=int, default=25)
    o = p.add_argument_group("output")
    o.add_argument("--paths", action="store_true", help="absolute paths only")
    o.add_argument("--json", action="store_true")
    o.add_argument("--index", help="index folder (default $SLAB_SAMPLE_INDEX or $SLAB_LIBRARY/.sample-index)")


def _print(hits, a):
    if a.json:
        print(json.dumps(hits, ensure_ascii=False, indent=1))
    elif a.paths:
        for r in hits:
            print(r["path"])
    else:
        for r in hits:
            print(line(r))
        if not hits:
            print("(nothing matches)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("index", help="analyze new or changed files and rebuild index.json")
    p.add_argument("--root", help="the samples folder (remembered; default $SLAB_LIBRARY/packs)")
    p.add_argument("--index")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    p.add_argument("--no-models", action="store_true", help="skip AudioSet tags and CLAP")
    p = sub.add_parser("models", help="AudioSet tags and CLAP embeddings")
    p.add_argument("--setup", action="store_true", help="create the venv and download the models (~2.5 GB)")
    p.add_argument("--index")
    p = sub.add_parser("find", help="filter the index")
    p.add_argument("words", nargs="*")
    _filter_args(p)
    p = sub.add_parser("text", help="free-text search (CLAP), with the same filters")
    p.add_argument("query")
    _filter_args(p)
    p = sub.add_parser("like", help="what sounds like this file (CLAP), with the same filters")
    p.add_argument("ref")
    _filter_args(p)
    p = sub.add_parser("show", help="every field of some records")
    p.add_argument("refs", nargs="+")
    p.add_argument("--all", action="store_true", help="include chroma and pitch curve")
    p.add_argument("--index")
    p = sub.add_parser("check", help="how far names and audio agree, per pack")
    p.add_argument("--index")
    a = ap.parse_args(argv)
    idx = index_dir(getattr(a, "index", None))

    if a.cmd == "index":
        cfg = _read_json(os.path.join(idx, "config.json")) or {}
        packs = os.path.join(slab_library(), "packs")
        root = a.root or os.environ.get("SLAB_SAMPLES") or cfg.get("root") or (packs if os.path.isdir(packs) else "")
        if not root:
            ap.error(f"index: --root <samples folder> (or put the packs in {packs})")
        root = os.path.abspath(os.path.expanduser(root))
        if not os.path.isdir(root):
            ap.error(f"index: {root} is not a folder")
        t0 = time.time()
        files = scan(root)
        print(f"{len(files)} audio files under {root}")
        analyze_audio(root, idx, files, a.jobs)
        if not a.no_models:
            run_models(root, idx, files)
        build(root, idx)
        print(f"done in {time.time() - t0:.0f}s")
    elif a.cmd == "models":
        if a.setup:
            setup_models(idx)
        cfg = _read_json(os.path.join(idx, "config.json")) or {}
        if not cfg.get("root"):
            ap.error("models: no index yet; run samples.py index --root <samples folder> first")
        files = scan(cfg["root"])
        run_models(cfg["root"], idx, files)
        build(cfg["root"], idx)
    elif a.cmd in ("find", "text", "like"):
        lib = load(idx)
        f = _filters(a)
        try:
            if a.cmd == "find":
                hits = lib.find(*a.words, **f)
            elif a.cmd == "text":
                lim = f.pop("limit", 25)
                hits = lib.text(a.query, limit=lim, **f)
            else:
                lim = f.pop("limit", 25)
                hits = lib.like(a.ref, limit=lim, **f)
        except (ValueError, KeyError) as e:
            ap.error(str(e))
        _print(hits, a)
    elif a.cmd == "show":
        lib = load(idx)
        for ref in a.refs:
            try:
                show(lib.get(ref), a.all)
            except KeyError as e:
                print(e)
    elif a.cmd == "check":
        lib = load(idx)
        check(lib, lib.data["root"])


if __name__ == "__main__":
    main()
