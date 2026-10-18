# The lab

This fork of [Slab](../README.md) (`keybrdist/slab`, from `nooga/slab`) is
also a music lab. AI agents write jungle, drumfunk and drum & bass tracks
here as Slab projects, in Python with [slabkit](../tools/slabkit/README.md),
from a library of sample packs. They render the projects headless, measure
the renders, and a person listens and decides what to keep.

Slab is the instrument. The lab is what's built around it:

- **A sample library**: 14 packs, 1,715 files, in `~/Music/Slab/Library/packs/`.
  Every file is analyzed and indexed by type, tempo, bars, key, pitch,
  loudness, tone, swing and vocals, so an agent can ask for "a 160 BPM
  break with real swing and a dark top" and get paths back in a tenth of a
  second ([tools/library/SAMPLES.md](../tools/library/SAMPLES.md)).
- **A tutorial library**: production tutorials on jungle, drumfunk and DnB in
  `~/Music/Slab/Knowledge/`, with notes, transcripts and a search tool.
- **Lab sessions**: worktrees under `.worktrees/slab-lab/<task>/`, one per
  line of work. The jungle-break lab holds the drum studies, the iteration
  pages and the scripts that built them.
- **Notes for agents**: [AGENTS.md](AGENTS.md) maps all of it, and says how
  to find samples, add packs and use the tutorials.

## How a piece gets made

1. A brief: a style, a reference track, a tempo, a mood.
2. Sources: search the sample index, then audition candidates against the
   reference. File names say what a sample claims to be, not how it sounds.
3. A Slab project, written with slabkit: breaks as loops or sliced kits,
   bass, keys, pads, arrangement, mix.
4. A headless render (`slab song.slab --render out.wav`) and measurements:
   loudness, band balance, swing.
5. Listening. The numbers are proxies, so the person listening decides, and
   each idea gets at most two versions before it changes direction.

## What stays out of Git

The sample packs are purchases. Their audio, the index built from them and
the model venv (about 2.5 GB) live under `~/Music/Slab/`, never in this
repository. Renders made from them may be published.
