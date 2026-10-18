# AGENTS.md: the lab

Notes for agents making music in this fork. The root [AGENTS.md](../AGENTS.md)
covers Slab itself; [README.md](README.md) says what the lab is. Read this
before choosing samples, adding packs or reaching for a tutorial.

## Where things live

```
~/Music/Slab/                      Slab's home folder ($SLAB_HOME)
  Library/                         $SLAB_LIBRARY
    packs/                         the sample library: one folder per pack, as the pack ships
      INDEX.md                     files per pack by category guessed from the title
      import_packs.py              brings new pack zips in, rewrites INDEX.md
    .sample-index/                 the sample index (hidden from Slab's browser)
      index.json                   one record per audio file
      cache/                       per-file analysis, keyed by path, size and mtime
      clap.npy, clap-ids.json      CLAP embeddings for text and sounds-like search
      venv/, hf/                   the model venv and model weights (~2.5 GB)
      wav/                         MP3s decoded once for Slab
  Knowledge/                       the tutorial library (see Tutorials below)
  Projects/ Samples/ Presets/ ...  Slab's own user folders
~/Sync/sample-packs/               new pack zips land here; imported ones move to imported/

<this repo>
  lab/                             these notes
  tools/library/samples.py         index and search the sample library (+ sample_audio.py,
                                   sample_names.py, sample_models.py; SAMPLES.md documents them)
  tools/slabkit/                   write Slab projects in Python (README.md)
  docs/33-composing.md             from a brief to an editable sketch
  docs/21-production-guide.md      mixing and production in Slab
  .worktrees/slab-lab/<task>/      one worktree per line of work
  .scratch/                        git-ignored working files: handoffs, renders, analyses
```

The jungle-break lab worktree (`.worktrees/slab-lab/jungle-break-lab`, branch
`research/jungle-break-lab`) holds the drum studies. Its
`docs/35-jungle-break-lab.md` is the session log: what was tried, what the
listener said, where every file is. Its untracked `.scratch/` holds:

| path | what |
|---|---|
| `drumfunk/df.py` | the core helper: onset-sliced breaks as one choked sampler kit, a groove model (swing, jitter, push), clips from step patterns |
| `drumfunk/analyze.py`, `notes.py`, `survey.py` | onset maps of a break, note maps of a phrase, quick sample surveys |
| `drumfunk/mixcheck.py`, `check.py` | judge a render: loudness, stems, drum checks |
| `drumfunk/iterate.py`, `iterpage.py`, `iterations.json` | build a version of an idea, measure it, log it; the living iteration page |
| `drumfunk/*_build.py`, `studies.py`, `paradox.py` | the builders of each round of ideas and studies |
| `samples/` | the lab's earlier copy of the packs; new work uses `~/Music/Slab/Library/packs` |
| `reference/`, `external/`, `generated/` | reference analyses; Strudel, makebreakbeat and Beat-Gen experiments |

## Finding samples

Ask the index, not the file names. Queries take about 0.1 s.

```sh
S=tools/library/samples.py
$S find --type break --bpm 160 --swing 6 --top dark        # a swung break with dull hats
$S find --type break --swing 8 --spread 6                  # live-played feel
$S find --type pad --key Fm --lacks D#                     # F minor, no 7th
$S find sub --type bass --oneshot --steady --clean         # a sine sub that holds its pitch
$S find --type top --bpm 174 --tone bright --no-kick
$S find --vocal
$S find amen --pack Homemade                               # words match names, keywords, tags
$S text "dusty rhodes chords"                              # free text (CLAP; first new query ~3 s)
$S like "TSSP2023 174 Grand Pad_Em"                        # what sounds like this one
$S show "Traitors Sub B"                                   # every field of a record
```

From Python:

```python
import sys; sys.path.insert(0, "<repo>/tools/library")
import samples
lib = samples.load()
hits = lib.find(type="break", bpm=170, swing=6, limit=10)
t.audio(samples.playable(hits[0]), section=drop, warp=hits[0]["bpm"])
```

What to trust:

- **Name and audio side by side.** `name_type`, `name_bpm` and `name_key` are
  what the file name claims; `audio_type`, `bpm_audio` and `key_audio` are
  measured. `type`, `bpm` and `key` are resolved, with `*_source` saying
  which side won. `flags` lists every disagreement; `--flagged` finds them.
- **Tempo.** A named tempo is reliable: 96% of named loops are whole bars at
  it. A tempo marked `~` in results came from the audio and may be half or
  double; `--bpm` matches both octaves for those.
- **Loops.** `bars` and `loop_secs` give the musical length. Some packs pad
  their files past the loop, so place `dur_sec=r["loop_secs"]` when it is set.
- **Key.** `key_audio_conf` below 0.3 means the chroma was unclear: distorted
  basses, several chords, atonal FX. One-shots carry `root`, `pitch_start`,
  `pitch_end` and `pitch_hold` (steady, glide, wobbly, moving) instead.
- **Groove.** `swing_ms` is how far the off-16ths sit behind the on-8ths,
  `spread_ms` the timing scatter. Pack loops are mostly grid-tight (spread
  2-4 ms). Real swing is in FatGyver's hip hop breaks and some of Fanu's.
  `swing_level` `8th` means it was read at half tempo.
- **Vocals.** Plainly audible voices are found from the audio; heavily
  processed ones only by name.
- **In Slab.** FLAC and WAV play directly. `samples.playable()` decodes an MP3
  once. Slab's own browser lists only `.wav` files, so use the index.

The full field list, method and measured accuracy are in
[tools/library/SAMPLES.md](../tools/library/SAMPLES.md).

## Managing the library

- **Add a pack.** Drop its zip in `~/Sync/sample-packs/`, then:

  ```sh
  python3 ~/Music/Slab/Library/packs/import_packs.py   # unzip into packs/, rewrite INDEX.md
  tools/library/samples.py index                      # analyze only the new files (~0.05 s each)
  tools/library/samples.py check                      # how far names and audio agree, per pack
  ```

  A pack with a new naming convention may need a rule in
  `tools/library/sample_names.py`; `check` shows when it does.
- **Models.** Vocal tags and text search need the venv:
  `tools/library/samples.py models --setup` makes it inside the index folder.
  The index builds without it.
- **Moving files.** Copy with `rsync -a` so mtimes survive; the cache is keyed
  by path under the packs folder, size and mtime, so a re-rooted index
  rebuilds in seconds: `samples.py index --root <new folder>`.
- **Removing a pack.** Delete its folder, then run `samples.py index`.
- **Never commit** pack audio or the index.
  `.scratch/` is git-ignored; `~/Music/Slab` is outside the repo.

## Tutorials

`~/Music/Slab/Knowledge/` holds six tutorials (Paradox parts 1 and 2,
flobama's layering and record-quantize lessons, STRANJAH's start-to-finish
DnB, UK Bass Tutorials' LMajor breaks). Each has a short `notes.md`, full
transcripts and timed captions. Load them on demand, never whole:

1. Read `context/drum-funk-thread.md` before generating anything. It holds
   the listener's feedback from earlier rounds.
2. Search for a technique instead of reading transcripts:
   `python3 ~/Music/Slab/Knowledge/library.py search 'pitch bend'` prints the
   matching lines and their files.
3. Open one tutorial's `notes.md` when the task calls for its technique; the
   index and reading paths are in `Knowledge/README.md`.
4. Go to a transcript or timed text only for the passage a search found.

To add a tutorial, follow `Knowledge/README.md` and run `library.py rebuild`
and `library.py check`.

## Working rules from earlier rounds

- At most two versions of an idea, unless the person asks for more; past v2
  the results got worse, not better.
- Choose sources by listening against the reference (A/B), not by name.
- Play a strong break as a loop with its own groove. Mining it into hits on
  a grid and running every piece through a clipper chain removed the swing
  that rolling jungle needs: the reference measured about 10 ms swing with
  11-16 ms spread, the rebuilt drums 2.5-5 ms.
- Tutorial techniques are options for a musical need, not one recipe.
- Prefer sampled keys and layers to Slab's synth keys, which sounded synthetic.
- Measurements diagnose; they don't prove a piece sounds right. Ask the
  person to listen at decision points.
