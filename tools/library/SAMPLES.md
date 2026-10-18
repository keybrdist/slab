# Sample index

`tools/library/samples.py` indexes every audio file (FLAC, WAV, AIFF, MP3)
under a folder of sample packs and answers questions like "a 160 BPM
break with real swing and a dark top" or "a clean sine sub that holds its
pitch" in about a tenth of a second. Each file gets one record. The record
holds what its name claims (type, tempo, key, keywords) beside what its
audio measures (type, loop or one-shot, tempo, bars, key or root, pitch
movement, loudness, tone, groove, vocals, drums), with a confidence for
each. Where name and audio disagree, the record carries a flag; the
disagreement is not resolved silently.

The packs are yours, so the index never goes into Git: it lives in
`$SLAB_SAMPLE_INDEX`, by default `.scratch/sample-index/` in this checkout.

## Setup

```sh
tools/library/samples.py index --root ~/path/to/samples   # first time; the root is remembered
tools/library/samples.py models --setup                   # optional, once: ~2.5 GB, see below
```

`index` decodes each file with ffmpeg and measures it with numpy, 12
files at a time. The lab's 1,715 files (3 GB) take about 90 s, and the
models another 200 s. Results are cached per file, keyed by path, size
and mtime, so a rerun only analyzes new or changed files and otherwise
rebuilds `index.json` in about 3 s.

`models --setup` makes a venv inside the index folder (torch and
transformers, with `HF_HOME` there too, so nothing is installed system
wide) and downloads two pretrained models:

- the AST AudioSet tagger (`MIT/ast-finetuned-audioset-10-10-0.4593`) gives
  the vocal flag and instrument tags (Piano, Hi-hat, Drum kit...);
- CLAP (`laion/larger_clap_music_and_speech`) gives free-text and
  sounds-like search.

Without the venv the index still builds: vocals then come only from names,
and `text` and `like` aren't available.

## Asking it

```sh
S=tools/library/samples.py
$S find --type break --bpm 160 --swing 6 --top dark        # real swing, dull hats
$S find --type pad --key Fm --lacks D#                     # F minor pad, no E♭ (no 7th)
$S find sub --type bass --oneshot --steady --clean         # a sine sub that holds its pitch
$S find --type top --bpm 174 --tone bright
$S find --vocal
$S text "dusty rhodes chords" --limit 10                   # CLAP, with any find filter
$S like "TSSP2023 174 Grand Pad_Em"                        # what sounds like this one
$S show "Traitors Sub B"                                   # every field
$S check                                                   # names against audio, per pack
```

Words match the keywords, title words, AudioSet tags, type and pack.
Filters:

| filter | meaning |
|---|---|
| `--type` | `break top perc-loop kick snare clap hat cymbal perc bass pad chord stab lead keys vocal fx riser atmos demo`, a family (`drum-loop drum-hit tonal fx`), `bass-loop`, `bass-oneshot`; comma-separated |
| `--loop` / `--oneshot` | |
| `--bpm N` `--bpm-tol` `--fold` | within 1.2%; `--fold` also matches half and double. A tempo measured from audio always folds (its octave is a guess); one from the name doesn't |
| `--key Fm` `--key-source` | matches the name's key, the audio's key (when confident) or the resolved one; `--key-source name\|audio\|resolved` narrows it |
| `--has PC` `--lacks PC` | a pitch class among the file's main ones, or absent |
| `--root F#` / `B0` | a tonal one-shot's note, optionally with octave |
| `--steady` `--glide` `--clean` | one-shot holds its pitch, slides, or is nearly a pure tone |
| `--swing MS` `--max-swing` `--spread MS` | groove (drum loops whose 16th grid fits) |
| `--tone` `--top` | dark, neutral or bright: overall, or the top end alone |
| `--energy` `--width` | low/medium/high; mono/stereo/wide |
| `--vocal` `--kick` `--drums` `--tonal` `--flagged` | each with a `--no-` form (`--atonal`) |
| `--pack` `--min-dur` `--max-dur` | |
| `--where EXPR` | any numeric field: `lufs>-10`, `band_air<0.05`, `purity>=0.9` |
| `--sort FIELD` | `-FIELD` for descending; the default ranks by the filters given |
| `--paths` `--json` `--limit N` | output |

A result line reads: type, loop or shot, bars, tempo (`~` when it came from
the audio), key or root, loudness, tone, energy, swing and spread, `!` when
the record has flags, pack, title.

## From Python

```python
import sys
sys.path.insert(0, "<slab>/tools/library")
import samples

lib = samples.load()                       # index.json: ~0.1 s
for r in lib.find(type="break", bpm=170, swing=6, top="dark"):
    print(r["path"], r["bpm"], r["bars"], r["swing_ms"])
hits = lib.text("airy pad with tape hiss", type="pad", limit=5)
near = lib.like(hits[0])
r = lib.get("Traitors Sub B")               # by id, path or a unique title fragment
```

`find`, `text` and `like` take the CLI's filters as keyword arguments
(`loop=False`, `has=["F"]`, `where=["lufs>-10"]`) and return record
dicts. `r["path"]` is absolute. Slab plays WAV and FLAC audio clips
directly; `samples.playable(r)` returns the path, or decodes an MP3 once
to a WAV in the index's `wav/` folder:

```python
t.audio(samples.playable(r), section=drop, warp=r["bpm"])
t.audio(samples.playable(r), dur_sec=r["loop_secs"])   # a padded loop's real length
```

## Adding packs

Import a pack into the samples folder (the lab's
`.scratch/samples/import_packs.py` unpacks pack zips there),
then run `samples.py index`. It analyzes only the new files. A pack whose
names follow a new convention may need a rule in `sample_names.py`:
`samples.py check` shows how far its names and audio agree.

## What a record holds

Fields are flat, so `--where` can test any number. Fields that don't
apply to a file are absent.

| group | fields |
|---|---|
| identity | `id` (path under the root), `pack`, `rel`, `path`, `file`, `title`, `format`, `codec`, `sample_rate`, `channels`, `bit_depth`, `duration`, `frames`, `size`, `mtime` |
| the name | `name_type`, `name_type_conf`, `name_bpm`, `name_key` (`Fm`, `C#`, or `E?` when the mode is unknown), `name_key_text`, `name_key_kind` (`key`, `root`, `chord`, `progression`), `name_chord`, `keywords` (normalized: `amen`, `reese`, `rhodes`, `808`, `sub`, `rave`, `dub`, `m1`, `d50`... plus measured tags such as `swung`, `dark-top`, `clean`), `words` (the rest of the title) |
| type | `type`, `type_source` (`name` or `audio`), `type_conf`, `audio_type`, `family` (`drum-loop`, `drum-hit`, `tonal`, `vocal`, `fx`, `demo`) |
| loop | `loop`, `loop_conf`, `loop_source`; for loops `bars`, `bars_exact` (whole bars at the tempo), `bars_how` (`exact`, `padded`, `tail`, `trailing silence`), `loop_secs` (the musical length), `padded` |
| tempo | `bpm`, `bpm_conf`, `bpm_source` (`name`, `audio + whole bars`, `audio`), `bpm_audio`, `bpm_audio_conf`, `bpm_bars` (whole bars nearest the detected tempo), `bpm_agree` (`same`, `half/double`, `3:2`, `alt`, `other`) |
| key and pitch | `key`, `key_source`, `key_conf`; `key_audio`, `key_audio_conf`, `chord_audio`, `pcs` (main pitch classes, strongest first), `chroma`; `key_agree` (`same`, `same (alt)`, `relative`, `parallel`, `fifth`, `near`, `other`), `key_fit` and `key_fits` (the name's scale holds the notes as well as any key's), `chord_overlap`; `tonal`, `tonality` (0..1), `tuning_cents`, `tuning_drift`, `detuned` |
| one-shot notes | `root`, `root_source`, `root_audio` (`B0`), `root_audio_midi`, `root_audio_cents`, `pitch_start`, `pitch_end` (and `_midi`), `glide_st`, `pitch_hold` (`steady`, `glide down`, `glide up`, `wobbly`, `moving`), `pitch_wobble_cents`, `purity` (fundamental's share of the harmonic energy: a sine is ~1), `pitch_curve` |
| energy | `lufs` (BS.1770, slabkit's meter), `peak_db`, `rms_db`, `crest_db`, `onset_rate` (hits per second), `energy` (low, medium, high), envelope: `t_peak`, `decay20`, `decay40`, `sustain`, `level_slope`, `active` |
| tone | `centroid`, `rolloff` (99% of the energy below it), `flatness`, `band_sub` `band_low` `band_lowmid` `band_mid` `band_high` `band_air` (slabkit's bands, shares of 20 Hz-20 kHz), `width`, `correlation`, `width_class`, `tone` and `brightness` (thirds of centroid within the type), `top` (air against high, thirds) |
| content | `percussive` (HPSS share), `kick` (a loop has kick hits), `kick_rate`, `drums` (drums under a melodic part), `vocal`, `vocal_source`, `vocal_ast`, `vocal_ast_label`, `tags` (AudioSet labels at 0.1 or more) |
| groove | `swing_ms` (off-16ths behind on-8ths, ms), `spread_ms` (timing spread around each position), `groove_fit` (share of hits on the grid), `swing_level` (`16th`, or `8th`: the 16th grid at half tempo); both levels' raw values `swing16_ms` `spread16_ms` `fit16` `swing8_ms` `spread8_ms` `fit8`; `groove_hits`, `off16_share` |
| flags | `flags`: `bpm: ...`, `key: ...`, `root: ...`, `chord: ...`, `type: ...`, `drums under the melodic part`, `kick in a top loop` |

## How it's measured

The audio half is `sample_audio.py`, the name half `sample_names.py`, the
models `sample_models.py`; `samples.py` sets them side by side.

- **Names.** The lab packs' file names read `<pack> - NN <title>`. Then each
  pack has its own conventions: Fracture writes `TYPE_NAME_[PROCESSOR]_KEY/TEMPO`
  with DR, BA, PA, ST, FX; Homemade Weapons writes `AMB_ BAS_ CYM_ DRM_ ETC_
  KCK_ PRC_ SNR_`; LMajor writes `LM DNB <TYPE> NN <key> <bpm>`; Total
  Science writes `TSSPyyyy <bpm> Name_Fm`; Pugilist writes `FMin - 130 BPM`
  and `DbMin or AbMin`; Om Unit writes `160bpm name`. Homemade Weapons'
  `1725` is 172.5. In the two Fanu breaks packs every title is a track
  name, so every file is a break.
- **Tempo.** Autocorrelation of the lab's onset function. For drum loops,
  a near-tied runner-up wins when it lays the hits on a much tighter 16th
  grid, and tempos under 100 are doubled (in this library named loops
  mostly sit at 160-175). Without a named tempo, the loop's length picks the
  whole-bar tempo nearest the detection.
- **Loops.** A loop is named as one, or is whole bars at its named tempo,
  or has the hits and the length of one. Bars count whole bars, also when
  padded to a multiple of 65,536 frames (Homemade Weapons) or followed by
  a tail.
- **Key.** Chroma comes from 32,768-point segments, spectral peaks from
  80 Hz up, after a tuning estimate; a bass-register chroma (35-260 Hz)
  sits beside it. The Krumhansl key is nudged toward a root found another
  way: a one-shot's own pitch, a bass line's longest steady note, or a
  pad's or single chord's root (best chord template, nudged by the bass
  note).
- **Pitch.** YIN over the first 12 s. Start and end are read from the
  frames within 20 dB of the peak, so a note's fade doesn't drag them, and
  the frequency is refined from the harmonics of a long FFT, so two beating
  sources don't misread it. A glide's root is where it lands; a name that
  gives the starting note still agrees.
- **Groove.** The lab's measure: how far the off-16ths sit behind the
  on-8ths, and the timing spread of every hit around its position's median.
  It is read on the 16th grid at the tempo shown and again at half tempo
  (the 8th level), with phase and step fitted by least squares on the
  on-8ths and parity checked against which hits are strongest, over up to
  8 bars or 20 s. `swing_ms` is the larger of the two levels whose grid
  holds 85% of the hits, so a 94 BPM break shown at 188 keeps its swung
  16ths. A result line marks the 8th level `sw+19/8`.
- **Vocal.** AudioSet voice labels (Speech, Singing, Choir...) at 0.35 or
  more, or a vocal word in the name.
- **Drums under a melodic part.** AudioSet hears a kit (0.25 or more), or a
  drum genre with kicks and percussive energy, or a whole track keeps a
  steady kick. Low pad notes set off the kick count and gated stabs look
  percussive, so neither decides alone.

## Accuracy

Measured 2026-10-18 on the jungle-break lab's library: 14 packs, 1,715
files, all indexed without errors. Names are the reference, and names are
sometimes wrong, so a disagreement is a lead to check, not proof of an
error. `samples.py check` prints this table for the current index.

| pack | tempo named | same | x2 | whole bars | keys named | same | 5th | fits | roots | same | type agree |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fracture | 37 | 73% | 14% | 97% | 62 | 58% | 13% | 63% | 25 | 100% | 93% |
| Homemade Weapons | 118 | 53% | 11% | 98% | 3 | 33% | 33% | 67% | 13 | 85% | 94% |
| LMajor | 48 | 81% | 4% | 98% | 54 | 61% | 9% | 65% | 41 | 88% | 94% |
| Om Unit | 86 | 60% | 24% | 97% | - | | | | - | | 90% |
| Pugilist Lo-fi Magic | 85 | 74% | 2% | 88% | - | | | | - | | 76% |
| Pugilist Outer Space | 40 | 25% | 2% | 100% | 32 | 50% | 16% | 78% | - | | 67% |
| Total Science 2023 | 67 | 58% | 4% | 96% | 39 | 54% | 3% | 44% | 14 | 79% | 96% |
| Total Science 2020 | 33 | 88% | 0% | 100% | 33 | 45% | 24% | 48% | 14 | 86% | 95% |
| all 14 packs | 555 | 63% | 9% | 96% | 228 | 55% | 12% | 61% | 108 | 88% | 92% |

- **Tempo.** 96% of named loops are whole bars at the named tempo, so for
  loops the name and the length agree almost always. Detection from audio
  alone matches the name on 86% of drum loops (91% counting half or double)
  but only 31% of tonal loops (pads and stabs have few onsets).
- **Key.** 55% of named keys match the detected key; 62% where the detector
  is confident (`key_audio_conf` 0.3 or more) and 76% counting a fifth. On
  61% the name's scale holds the notes as well as any key does.
- **Roots.** Tonal one-shots match the named root 88% of the time,
  within 55 cents, at either end of a glide.
- **Types.** 92% compatible.
- **Spot check.** 21 files of every type were compared with their
  spectrograms. What that turned up (kick flags on one-shots, a gated stab
  taken for drums, a riser typed as FX, a live break read at two-thirds
  tempo) is fixed in the numbers above.

The traps measured in the lab come out like this:

- *Homemade Weapons "BAS_Traitors Sub B"*: a glide down from A1 to B0, root
  B, agreeing with the name.
- *YUKU "Flute - Argoul"*: C♯4 (two pipes beating around 272-278 Hz).
- *Homemade Weapons loops*: 89 files padded to 65,536-frame multiples; all
  40 drum loops come out as whole bars at the named tempo, with `loop_secs`
  their real length.
- *Total Science pads labelled `_Em`* (Grand Pad, Dimensional): notes
  B D F♯ G E, chord Em9 and Em11, key E minor. A plain Krumhansl profile
  reads B minor; the label is right.
- *LMajor "Chord 03 Cm9"*: flagged; the audio holds F and C, an F minor
  chord.
- *LMajor drift*: 8 of the 20 Chord Pad and Chord Stab files are marked
  `detuned` (15 cents or more off, or 25 cents of drift); the rest are
  within 14 cents.
- *FatGyver Sample Dumpster Diving*: all 13 tracks are flagged with drums
  under them, and the one beat (CS Beat) is a break.
- *Mako*: generic names, tempos 172-175 from whole-bar lengths.

## Weak spots

- **Tempo octave.** A loop with no tempo in its name gets its octave from
  the rule above, so FatGyver's hip hop breaks (85-96) show at 170-192.
  Queries fold measured tempos; groove reads both levels.
- **Swing** assumes the loop starts on a downbeat (parity is checked, not
  guaranteed). Named pack loops are mostly grid-tight (median spread
  2.7 ms, swing 3 ms). The swung material is FatGyver's hip hop breaks
  (median 19 ms at the 8th level, spread 7 ms) and a quarter of Fanu's
  breaks (9 ms or more). For a live feel ask for spread too:
  `--swing 8 --spread 6`.
- **Keys of tonal loops** with several chords, distorted reese basses
  (Total Science 2020's bass loops read near-flat chroma) and atonal FX.
  `key_audio_conf` says when not to trust it.
- **Vocals that are heavily processed** (verbed, granular, pads named
  "voices") are found by name only: AudioSet scores them low, and CLAP's
  zero-shot prompts scored kicks above them.
- **808 kicks and 808 basses**, and pitched percussion and short stabs,
  look alike; `type` follows the name when it has one.
- **Free-text search** loads CLAP for the first new query (about 3 s);
  repeated queries are cached.
