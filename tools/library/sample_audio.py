"""sample_audio.py - what a sample sounds like, measured with numpy.

The audio half of the sample index (tools/library/samples.py): one call,
`analyze(path)`, decodes a file with ffmpeg and returns a dict of plain
numbers and short lists, safe to cache as JSON. Nothing here reads the
file name; samples.py sets the two side by side.

Measured:
  levels      integrated loudness (BS.1770, slabkit.analyze's K-filter),
              peak, RMS, crest
  spectrum    centroid, 99% rolloff, flatness, slabkit's six band shares,
              stereo width and correlation
  onsets      the jungle-break lab's spectral-flux detector (hop 128), so
              swing numbers line up with the lab's: times and strengths
              over the first minute, kept for groove
  tempo       autocorrelation of the onset envelope: the best few tempos
  harmonic/   median-filter HPSS: the percussive share of the energy
  percussive
  chroma      32,768-point segments from 80 Hz up, spectral peaks only,
              after a tuning estimate (4,096-point frames smear every file
              toward one pitch class); a Krumhansl key with its margin;
              the same for 35-260 Hz (the bass note); tuning drift
  pitch       YIN over the first 12 s: start, end, median, the longest
              steady stretch, how pure the tone is (a sine sub is ~1)
  envelope    time to peak, decay to -20 and -40 dB, level and centroid
              slopes (a riser climbs), active length without trailing
              silence, the exact frame count
  kick        onsets dominated by 40-160 Hz, per second
"""

import json
import math
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from slabkit.analyze import BANDS, _blocks, _k_weight, integrated  # noqa: E402

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

LONG = 90.0        # files longer than this are measured on three 30 s windows
GROOVE_SECS = 60.0  # onsets and tempo come from the first minute
PITCH_SECS = 12.0


def probe(path):
    """Container facts: codec, rate, channels, bit depth, frames."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_streams",
                        "-show_format", "-of", "json", path], capture_output=True, text=True)
    info = json.loads(r.stdout or "{}")
    s = (info.get("streams") or [{}])[0]
    fmt = info.get("format", {})
    bits = int(s.get("bits_per_raw_sample") or s.get("bits_per_sample") or 0) or None
    return {
        "codec": s.get("codec_name"),
        "sample_rate": int(s.get("sample_rate") or 0),
        "channels": int(s.get("channels") or 0),
        "bit_depth": bits,
        "bitrate": int(fmt.get("bit_rate") or 0) or None,
    }


def decode(path, channels):
    """(frames, ch) float32 at the file's own rate, at most two channels."""
    ch = 2 if channels >= 2 else 1
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-map", "0:a:0", "-ac", str(ch),
                        "-f", "f32le", "-"], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode(errors="replace").strip() or "ffmpeg failed")
    return np.frombuffer(r.stdout, "<f4").reshape(-1, ch)


def _r(v, n=3):
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return None if not math.isfinite(v) else round(v, n)
    return v


def _db(v):
    return 20 * math.log10(max(float(v), 1e-9))


def _excerpt(x, sr):
    """The file, or three 30 s windows of a long one (start, middle, end)."""
    n = len(x)
    if n <= LONG * sr:
        return x
    w = int(30 * sr)
    starts = [int(n * 0.05), int(n * 0.45), max(0, n - w - int(n * 0.05))]
    return np.concatenate([x[a:a + w] for a in starts])


# ── levels and spectrum ───────────────────────────────────────────────────
def levels(x, sr):
    st = x if x.shape[1] == 2 else np.repeat(x, 2, axis=1)   # mono plays on both sides
    kx = _k_weight(st.astype(np.float64), sr)
    lufs = integrated(_blocks(kx, sr)[0])
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    return {"lufs": _r(lufs, 2), "peak_db": _r(_db(peak), 2), "rms_db": _r(_db(rms), 2),
            "crest_db": _r(_db(peak) - _db(rms), 2)}


def spectrum(x, sr):
    m = x.mean(axis=1).astype(np.float64)
    n = 4096
    if len(m) < n:
        m = np.pad(m, (0, n - len(m)))
    frames = np.lib.stride_tricks.sliding_window_view(m, n)[::n // 2] * np.hanning(n)
    mag = np.abs(np.fft.rfft(frames, axis=1))
    pw = mag ** 2
    f = np.fft.rfftfreq(n, 1 / sr)
    band = (f >= 20) & (f < 20000)
    mean_pw = pw.mean(axis=0)
    total = mean_pw[band].sum() or 1e-20
    bands = {name: _r(mean_pw[(f >= lo) & (f < hi)].sum() / total, 4) for name, lo, hi in BANDS}
    mean_mag = mag.mean(axis=0)
    centroid = float((mean_mag[band] * f[band]).sum() / (mean_mag[band].sum() or 1e-20))
    c = np.cumsum(mean_pw[band]) / total
    rolloff = float(f[band][min(np.searchsorted(c, 0.99), band.sum() - 1)])
    fb = (f >= 100) & (f < 10000)
    loud = pw[:, fb].sum(axis=1) > pw[:, fb].sum(axis=1).max() * 1e-4
    p = pw[loud][:, fb] + 1e-20
    flat = float(np.mean(np.exp(np.mean(np.log(p), axis=1)) / np.mean(p, axis=1))) if len(p) else 0.0
    out = {"centroid": _r(centroid, 0), "rolloff": _r(rolloff, 0), "flatness": _r(flat, 4), "bands": bands}
    if x.shape[1] == 2:
        l, r = x[:, 0].astype(np.float64), x[:, 1].astype(np.float64)
        mid, side = (l + r) / 2, (l - r) / 2
        den = math.sqrt(float(np.sum(l * l) * np.sum(r * r))) or 1e-12
        out["correlation"] = _r(float(np.sum(l * r)) / den, 3)
        out["width"] = _r(math.sqrt(float(np.mean(side ** 2))) / max(math.sqrt(float(np.mean(mid ** 2))), 1e-12), 3)
    else:
        out["correlation"], out["width"] = 1.0, 0.0
    return out


# ── onsets and tempo ──────────────────────────────────────────────────────
def flux(m, sr, hop=128, nfft=1024):
    """The lab's onset function (drumfunk/analyze.py): log-magnitude flux."""
    win = np.hanning(nfft).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(m, (nfft, nfft)), nfft)[::hop] * win
    mag = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32)
    fl = np.maximum(0, np.diff(np.log1p(mag * 50), axis=0)).sum(axis=1)
    return np.concatenate([[0], fl]), hop, nfft


def onsets(m, sr, thresh=1.6):
    fl, hop, nfft = flux(m, sr)
    med = np.convolve(fl, np.ones(31) / 31, mode="same")
    peaks = []
    for i in range(2, len(fl) - 2):
        if fl[i] > fl[i - 1] and fl[i] >= fl[i + 1] and fl[i] > thresh * med[i] + 1.0:
            t = (i * hop - nfft) / sr + nfft / 2 / sr
            if not peaks or t - peaks[-1][0] > 0.045:
                peaks.append((max(t, 0.0), float(fl[i])))
            elif fl[i] > peaks[-1][1]:
                peaks[-1] = (max(t, 0.0), float(fl[i]))
    return peaks, fl, sr / hop


def tempo(fl, fps):
    """Best tempos (60-200) from the autocorrelation of the onset envelope:
    [(bpm, score)], score 0..1 relative to the best."""
    if len(fl) < fps * 2:
        return []
    k = int(fps * 0.5)
    env = fl - np.convolve(fl, np.ones(k) / k, mode="same")
    env = np.maximum(env, 0)
    if not env.any():
        return []
    n = 1 << int(np.ceil(np.log2(len(env) * 2)))
    acf = np.fft.irfft(np.abs(np.fft.rfft(env, n)) ** 2)[:len(env)]
    acf /= acf[0] or 1
    bpms = np.arange(60, 200.01, 0.1)

    def at(lag):
        i = np.floor(lag).astype(int)
        ok = i + 1 < len(acf)
        fr = lag - i
        v = np.zeros_like(lag)
        v[ok] = acf[i[ok]] * (1 - fr[ok]) + acf[i[ok] + 1] * fr[ok]
        return v

    lag = 60 * fps / bpms
    score = at(lag) + 0.8 * at(2 * lag) + 0.6 * at(3 * lag) + 0.5 * at(4 * lag) + 0.5 * at(lag / 2)
    prior = np.exp(-0.5 * (np.log2(bpms / 130) / 1.0) ** 2)
    s = score * prior
    out = []
    order = np.argsort(-s)
    for i in order:
        b = bpms[i]
        if all(abs(b - o) / o > 0.03 for o, _ in out):
            out.append((float(b), float(s[i])))
        if len(out) == 4:
            break
    top = out[0][1] or 1
    mean = float(np.mean(s)) or 1e-9
    return [(round(b, 1), round(v / top, 3)) for b, v in out], round(float(out[0][1] / mean), 2)


# ── harmonic / percussive ─────────────────────────────────────────────────
def _median_filter(a, size, axis):
    pad = [(0, 0), (0, 0)]
    pad[axis] = (size // 2, size // 2)
    ap = np.pad(a, pad, mode="edge")
    out = np.empty_like(a)
    step = 400
    if axis == 0:
        for i in range(0, a.shape[0], step):
            w = np.lib.stride_tricks.sliding_window_view(ap[i:i + step + size - 1], size, axis=0)
            out[i:i + step] = np.median(w, axis=-1)
    else:
        for i in range(0, a.shape[0], step):
            w = np.lib.stride_tricks.sliding_window_view(ap[i:i + step], size, axis=1)
            out[i:i + step] = np.median(w, axis=-1)
    return out


def percussive_share(m, sr):
    """Share of energy in the percussive part of a median-filter HPSS, on up
    to 30 s (three 10 s windows of longer audio)."""
    n, hop = 2048, 512
    if len(m) > 30 * sr:
        w = int(10 * sr)
        m = np.concatenate([m[int(a):int(a) + w] for a in np.linspace(0, len(m) - w, 3)])
    if len(m) < n * 2:
        m = np.pad(m, (0, n * 2 - len(m)))
    frames = np.lib.stride_tricks.sliding_window_view(m, n)[::hop] * np.hanning(n).astype(np.float32)
    S = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32)
    H = _median_filter(S, 17, 0)
    P = _median_filter(S, 17, 1)
    h2, p2 = H ** 2, P ** 2
    den = h2 + p2 + 1e-12
    eh = float((S ** 2 * h2 / den).sum())
    ep = float((S ** 2 * p2 / den).sum())
    return ep / (eh + ep + 1e-12)


# ── chroma and key ────────────────────────────────────────────────────────
def chroma(m, sr, fmin=80.0, fmax=2500.0, nk=1 << 15):
    """(12-bin chroma, tuning in cents, spread in semitones, drift in cents)
    from spectral peaks of long segments. Drift is the 10-90% range of
    each segment's own tuning: a pitch that wanders over the file."""
    if len(m) < nk:
        m = np.pad(m, (0, nk - len(m)))
    f = np.fft.rfftfreq(nk, 1 / sr)
    band = np.nonzero((f >= fmin) & (f < fmax))[0]
    win = np.hanning(nk)
    pk_f, pk_w, seg_tune = [], [], []
    starts = range(0, max(1, len(m) - nk + 1), nk // 2)
    for a in list(starts)[:120]:
        X = np.abs(np.fft.rfft(m[a:a + nk] * win))
        top = X[band].max()
        if top <= 1e-6:
            continue
        seg = X[band[0] - 1:band[-1] + 2]
        c = seg[1:-1]
        is_pk = (c > seg[:-2]) & (c >= seg[2:]) & (c > top * 0.03)
        idx = band[is_pk]
        if not len(idx):
            continue
        # parabolic interpolation for a sub-bin frequency
        a0, b0, c0 = np.log(X[idx - 1] + 1e-12), np.log(X[idx] + 1e-12), np.log(X[idx + 1] + 1e-12)
        d = 0.5 * (a0 - c0) / np.where(np.abs(a0 - 2 * b0 + c0) < 1e-12, 1e-12, a0 - 2 * b0 + c0)
        fq, w = f[idx] + d * (f[1] - f[0]), X[idx] / top
        pk_f.append(fq)
        pk_w.append(w)
        fr = 69 + 12 * np.log2(fq / 440)
        fr -= np.round(fr)
        if w.sum() > 0:
            seg_tune.append(np.angle(np.sum(w * np.exp(2j * np.pi * fr))) / (2 * np.pi))
    if not pk_f:
        return None
    pf, pw = np.concatenate(pk_f), np.concatenate(pk_w)
    midi = 69 + 12 * np.log2(pf / 440)
    frac = midi - np.round(midi)
    ang = np.angle(np.sum(pw * np.exp(2j * np.pi * frac)))
    tuning = ang / (2 * np.pi)                      # semitones, -0.5..0.5
    pcs = np.round(midi - tuning).astype(int) % 12
    c = np.zeros(12)
    np.add.at(c, pcs, pw)
    resid = (frac - tuning + 0.5) % 1 - 0.5          # how far peaks sit off the tuned grid
    spread = float(np.sqrt(np.average(resid ** 2, weights=pw)))
    drift = None
    if len(seg_tune) >= 3:
        st = (np.array(seg_tune) - tuning + 0.5) % 1 - 0.5
        drift = float((np.percentile(st, 90) - np.percentile(st, 10)) * 100)
    return c / (c.sum() or 1), float(tuning * 100), spread, drift


def key_of(c):
    """Krumhansl key: (pc, mode, r, margin over the next different key)."""
    rs = [(float(np.corrcoef(c, np.roll(MAJOR, k))[0, 1]), k, "major") for k in range(12)] + \
         [(float(np.corrcoef(c, np.roll(MINOR, k))[0, 1]), k, "minor") for k in range(12)]
    rs.sort(reverse=True)
    return rs[0][1], rs[0][2], rs[0][0], rs[0][0] - rs[1][0], [(k, md, round(r, 3)) for r, k, md in rs[:3]]


# ── pitch ─────────────────────────────────────────────────────────────────
def yin(m, sr, fmin=27.0, fmax=1600.0, thresh=0.15):
    """Per-frame (time, f0 or nan, aperiodicity, rms) with YIN."""
    tau_max = int(sr / fmin)
    frame = 1 << int(np.ceil(np.log2(2.2 * tau_max)))
    hop = frame // 8
    if len(m) < frame + hop:
        return np.zeros((0, 4))
    F = np.lib.stride_tricks.sliding_window_view(m, frame)[::hop].astype(np.float64)
    W = frame - tau_max
    n = 1 << int(np.ceil(np.log2(frame + W)))
    acf = np.fft.irfft(np.conj(np.fft.rfft(F[:, :W], n)) * np.fft.rfft(F, n), n)[:, :tau_max + 1]
    cs = np.concatenate([np.zeros((len(F), 1)), np.cumsum(F ** 2, axis=1)], axis=1)
    e0 = cs[:, W][:, None]
    taus = np.arange(tau_max + 1)
    et = cs[:, taus + W] - cs[:, taus]
    d = np.maximum(e0 + et - 2 * acf, 0)
    d[:, 0] = 0
    cum = np.cumsum(d[:, 1:], axis=1)
    dn = np.ones_like(d)
    dn[:, 1:] = d[:, 1:] * np.arange(1, tau_max + 1) / np.where(cum == 0, 1, cum)
    tau_min = max(2, int(sr / fmax))
    out = np.full((len(F), 4), np.nan)
    rms = np.sqrt(np.mean(F ** 2, axis=1))
    for i in range(len(F)):
        row = dn[i]
        below = np.nonzero(row[tau_min:] < thresh)[0]
        if len(below):
            t = tau_min + below[0]
            while t + 1 <= tau_max and row[t + 1] < row[t]:
                t += 1
        else:
            t = tau_min + int(np.argmin(row[tau_min:]))
        ap = row[t]
        if 1 <= t < tau_max:
            a, b, c = row[t - 1], row[t], row[t + 1]
            den = a - 2 * b + c
            tt = t + (0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0)
        else:
            tt = t
        out[i] = ((i * hop + frame / 2) / sr, sr / tt if tt > 0 else np.nan, ap, rms[i])
    return out


def _midi(f):
    return 69 + 12 * np.log2(f / 440.0)


def note_name(p):
    q = int(round(p))
    return f"{NAMES[q % 12]}{q // 12 - 1}"


def refine_f0(m, sr, f0, a, b):
    """f0 read from the harmonics of a long FFT over [a, b) samples: the
    weighted median of each partial's frequency over its number. A single
    peak misreads two beating sources; the partials outvote it."""
    seg = m[a:b]
    n = 1 << 16
    if len(seg) < 4096:
        return f0, None
    seg = seg[:n]
    X = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n))
    fr = np.fft.rfftfreq(n, 1 / sr)
    ests, ws, amps = [], [], []
    for k in range(1, 9):
        lo, hi = k * f0 * 0.97, k * f0 * 1.03
        if hi > sr / 2 - 100:
            break
        sel = np.nonzero((fr >= lo) & (fr <= hi))[0]
        if not len(sel):
            continue
        i = sel[np.argmax(X[sel])]
        if 0 < i < len(X) - 1:
            al, bl, cl = np.log(X[i - 1] + 1e-12), np.log(X[i] + 1e-12), np.log(X[i + 1] + 1e-12)
            den = al - 2 * bl + cl
            fi = fr[i] + (0.5 * (al - cl) / den if abs(den) > 1e-12 else 0) * (fr[1] - fr[0])
        else:
            fi = fr[i]
        ests.append(fi / k)
        ws.append(X[i])
        amps.append(X[i] ** 2)
    if not ests:
        return f0, None
    ests, ws = np.array(ests), np.array(ws)
    o = np.argsort(ests)
    cw = np.cumsum(ws[o])
    f = float(ests[o][np.searchsorted(cw, cw[-1] / 2)])
    # purity: the fundamental's share of the harmonic energy
    purity = float(amps[0] / (sum(amps) or 1e-12)) if abs(ests[0] - f) / f < 0.03 else None
    return f, purity


def pitch(m, sr):
    seg = m[:int(PITCH_SECS * sr)].astype(np.float64)
    tr = yin(seg, sr)
    if not len(tr):
        return None
    rms = tr[:, 3]
    top = np.nanmax(rms) if len(rms) else 0
    if top <= 1e-6:
        return None
    voiced = (tr[:, 2] < 0.2) & (rms > top * 10 ** (-35 / 20)) & np.isfinite(tr[:, 1])
    if voiced.sum() < 4:
        return {"voiced": round(float(voiced.mean()), 3)}
    t, f = tr[voiced, 0], tr[voiced, 1]
    body = rms[voiced] > top * 0.1          # within 20 dB of the peak: the note, not its fade
    p = _midi(f)
    if len(p) >= 5:   # 5-frame median: drop single-frame octave slips
        pp = np.pad(p, 2, mode="edge")
        p = np.median(np.lib.stride_tricks.sliding_window_view(pp, 5), axis=1)
    n = len(p)
    pb = p[body] if body.sum() >= 2 else p
    k = max(1, len(pb) // 10)
    start, end = float(np.median(pb[:k])), float(np.median(pb[-k:]))
    med = float(np.median(p))
    # longest run within half a semitone of its own median
    best = (0, 0)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(p[j + 1] - np.median(p[i:j + 2])) < 0.5 and t[j + 1] - t[j] < 0.15:
            j += 1
        if j - i > best[1] - best[0]:
            best = (i, j)
        i = j + 1
    a, b = best
    plateau = float(np.median(p[a:b + 1]))
    plateau_secs = float(t[b] - t[a]) if b > a else 0.0
    hop_s = float(np.median(np.diff(tr[:, 0]))) if len(tr) > 1 else 0.01
    a_s = max(0, int((t[a] - hop_s * 4) * sr))
    b_s = int((t[b] + hop_s * 4) * sr)
    f_ref, purity = refine_f0(seg, sr, 440 * 2 ** ((plateau - 69) / 12), a_s, max(b_s, a_s + 8192))
    ref = float(_midi(f_ref))
    if abs(ref - plateau) > 0.6:     # the harmonics disagree badly: keep YIN's
        ref = plateau
    curve = [round(float(v), 2) for v in p[np.linspace(0, n - 1, min(n, 24)).astype(int)]]
    return {
        "voiced": round(float(voiced.mean()), 3),
        "start": round(start, 2), "end": round(end, 2), "median": round(med, 2),
        "plateau": round(ref, 2), "plateau_secs": round(plateau_secs, 2),
        "glide": round(end - start, 2),
        "wobble": round(float(np.std(p[a:b + 1])) * 100, 1),
        "range": round(float(np.percentile(p, 95) - np.percentile(p, 5)), 2),
        "purity": _r(purity, 3),
        "curve": curve,
    }


# ── envelope, kick ────────────────────────────────────────────────────────
def envelope(m, sr):
    hop = int(sr * 0.01)
    n = len(m) // hop
    if n < 2:
        return {}
    e = np.sqrt(np.mean(m[:n * hop].reshape(n, hop).astype(np.float64) ** 2, axis=1)) + 1e-9
    db = 20 * np.log10(e)
    pk = int(np.argmax(db))
    top = db[pk]

    def fall(level):
        below = np.nonzero(db[pk:] < top + level)[0]
        return round(below[0] * 0.01, 3) if len(below) else None

    active = np.nonzero(db > top - 50)[0]
    t = np.arange(n) * 0.01
    sel = db > top - 40
    slope = float(np.polyfit(t[sel], db[sel], 1)[0]) if sel.sum() > 3 else 0.0
    # how the level is spread: share of 10 ms frames within 20 dB of the peak
    sustain = float(np.mean(db[active[0]:active[-1] + 1] > top - 20)) if len(active) else 0.0
    return {"t_peak": round(pk * 0.01, 3), "decay20": fall(-20), "decay40": fall(-40),
            "level_slope": round(slope, 2), "sustain": round(sustain, 3),
            "lead_in": round(active[0] * 0.01, 3) if len(active) else 0.0,
            "active": round((active[-1] + 1) * 0.01, 3) if len(active) else 0.0}


def centroid_slope(m, sr):
    n = 4096
    if len(m) < n * 4:
        return 0.0
    frames = np.lib.stride_tricks.sliding_window_view(m, n)[::n] * np.hanning(n)
    mag = np.abs(np.fft.rfft(frames, axis=1))
    f = np.fft.rfftfreq(n, 1 / sr)
    e = mag.sum(axis=1)
    sel = e > e.max() * 0.01
    if sel.sum() < 4:
        return 0.0
    c = (mag[sel] * f).sum(axis=1) / e[sel]
    t = np.nonzero(sel)[0] * n / sr
    return round(float(np.polyfit(t / max(t[-1], 1e-6), np.log2(np.maximum(c, 20)), 1)[0]), 3)


def kick(m, sr, times):
    """Kick hits per second: onsets whose first 40 ms put a lot of energy in
    40-160 Hz against 160 Hz-8 kHz: above 1.0 outright (a kick pattern,
    every hit low), or above 0.15 and three times the loop's own median
    onset, so a hat loop's rumble doesn't count and a thin old break's kick
    still does."""
    n = 4096
    f = np.fft.rfftfreq(n, 1 / sr)
    lo_b, hi_b = (f >= 40) & (f < 160), (f >= 160) & (f < 8000)
    w = int(0.04 * sr)
    rs = []
    for t in times:
        a = int(t * sr)
        seg = m[a:a + w]
        if len(seg) < 64:
            continue
        X = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n)) ** 2
        rs.append(X[lo_b].sum() / (X[hi_b].sum() + 1e-12))
    if not rs:
        return {"kick_rate": 0.0, "kick_max": 0.0}
    rs = np.array(rs)
    hits = int(np.sum((rs >= 1.0) | (rs >= max(0.15, 3 * float(np.median(rs))))))
    return {"kick_rate": round(hits / max(len(m) / sr, 0.05), 3), "kick_max": round(float(rs.max()), 3)}


# ── all together ──────────────────────────────────────────────────────────
def analyze(path):
    info = probe(path)
    x = decode(path, info["channels"] or 2)
    sr = info["sample_rate"]
    out = dict(info)
    out["frames"] = len(x)
    out["duration"] = round(len(x) / sr, 4) if sr else 0.0
    if len(x) == 0 or float(np.max(np.abs(x))) < 1e-5:
        out["silent"] = True
        return out
    ex = _excerpt(x, sr)
    m_ex = ex.mean(axis=1)
    m = x.mean(axis=1)
    out.update(levels(ex, sr))
    out.update(spectrum(ex, sr))
    out.update(envelope(m, sr))
    out["centroid_slope"] = centroid_slope(m_ex, sr)
    lead = int(out.get("lead_in", 0) * sr)
    g = m[lead:lead + int(GROOVE_SECS * sr)]
    pk, fl, fps = onsets(g, sr)
    out["onset_rate"] = round(len(pk) / max(len(g) / sr, 0.05), 3)
    out["onsets"] = [[round(t + lead / sr, 4), round(s, 2)] for t, s in pk[:1200]]
    tp = tempo(fl, fps)
    if tp:
        out["tempo"], out["tempo_peak"] = tp
    out["percussive"] = round(percussive_share(m_ex, sr), 4)
    ch = chroma(m_ex, sr)
    if ch is not None:
        c, tuning, spread, drift = ch
        k, mode, r, margin, top3 = key_of(c)
        out["tuning_drift"] = _r(drift, 1)
        out["chroma"] = [round(float(v), 4) for v in c]
        out["tuning_cents"] = round(tuning, 1)
        out["tuning_spread"] = round(spread, 3)
        out["key"] = [k, mode, round(r, 3), round(margin, 3)]
        out["key_top"] = top3
        ent = -float(np.sum(c * np.log(c + 1e-12))) / math.log(12)
        out["chroma_entropy"] = round(ent, 3)
    bc = chroma(m_ex, sr, 35.0, 260.0)            # the bass register: a chord's lowest note is usually its root
    if bc is not None:
        out["bass_chroma"] = [round(float(v), 4) for v in bc[0]]
    out.update(kick(g, sr, [t for t, _ in pk]))
    pt = pitch(m, sr)
    if pt:
        out["pitch"] = pt
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(json.dumps(analyze(p), indent=1, ensure_ascii=False)[:4000])
