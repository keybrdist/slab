"""sample_models.py - AudioSet tags and CLAP embeddings for the sample index.

Runs in the index's own venv (made by `samples.py models --setup`), with
HF_HOME pointing inside the index folder, so nothing is installed system
wide. Two pretrained models:

  AST   MIT/ast-finetuned-audioset-10-10-0.4593: 527 AudioSet labels
        (Speech, Singing, Choir, Drum kit, Hi-hat, Piano, Synthesizer...);
        the vocal flag and instrument tags come from it
  CLAP  laion/larger_clap_music_and_speech: audio and text in one
        512-d space, for free-text and sounds-like search

Each file is heard as up to three 10 s windows (start, middle, end); a
label's score is its best window, the embedding the mean of the windows'.

  sample_models.py fetch              download both models
  sample_models.py tag JOBS.json      tag the files listed, writing each
                                      one's cache entry (samples.py does this)
  sample_models.py text QUERY ...     CLAP text embeddings, JSON on stdout
  sample_models.py embed FILE ...     CLAP audio embeddings, JSON on stdout
"""

import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

AST = "MIT/ast-finetuned-audioset-10-10-0.4593"
CLAP = "laion/larger_clap_music_and_speech"
SR = 48000
WIN = 10.0

VOCAL = {"Speech", "Male speech, man speaking", "Female speech, woman speaking", "Child speech, kid speaking",
         "Conversation", "Narration, monologue", "Babbling", "Speech synthesizer", "Shout", "Yell", "Whispering",
         "Singing", "Choir", "Yodeling", "Chant", "Mantra", "Male singing", "Female singing", "Child singing",
         "Synthetic singing", "Rapping", "Humming", "Vocal music", "A capella", "Human voice", "Beatboxing"}


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def decode(path):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-map", "0:a:0", "-ac", "1", "-ar", str(SR),
                        "-f", "f32le", "-"], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode(errors="replace").strip())
    return np.frombuffer(r.stdout, "<f4").copy()


def windows(x):
    n = int(WIN * SR)
    if len(x) <= n:
        return [x if len(x) >= SR // 10 else np.pad(x, (0, SR // 10 - len(x)))]
    starts = [0, (len(x) - n) // 2, len(x) - n]
    return [x[a:a + n] for a in dict.fromkeys(starts)]


class Models:
    """Each model loads on first use: a text query needs only CLAP."""

    def __init__(self):
        self.dev = device()
        self._ast = self._clap = None

    def _load_ast(self):
        if self._ast is None:
            from transformers import ASTFeatureExtractor, ASTForAudioClassification
            self.ast_fx = ASTFeatureExtractor.from_pretrained(AST)
            self._ast = ASTForAudioClassification.from_pretrained(AST).eval().to(self.dev)
            self.labels = [self._ast.config.id2label[i] for i in range(self._ast.config.num_labels)]
        return self._ast

    def _load_clap(self):
        if self._clap is None:
            from transformers import ClapModel, ClapProcessor
            self.clap_proc = ClapProcessor.from_pretrained(CLAP)
            self._clap = ClapModel.from_pretrained(CLAP).eval().to(self.dev)
        return self._clap

    @torch.inference_mode()
    def tags(self, wins):
        import torchaudio.functional as AF
        w16 = [AF.resample(torch.from_numpy(w), SR, 16000).numpy() for w in wins]
        ast = self._load_ast()
        feats = self.ast_fx(w16, sampling_rate=16000, return_tensors="pt")
        logits = ast(**{k: v.to(self.dev) for k, v in feats.items()}).logits
        return torch.sigmoid(logits).max(dim=0).values.float().cpu().numpy()

    @torch.inference_mode()
    def embed_audio(self, wins):
        clap = self._load_clap()
        inp = self.clap_proc(audio=wins, sampling_rate=SR, return_tensors="pt")
        e = clap.get_audio_features(**{k: v.to(self.dev) for k, v in inp.items()})
        e = getattr(e, "pooler_output", e)
        e = torch.nn.functional.normalize(e.float(), dim=-1).mean(dim=0)
        return torch.nn.functional.normalize(e, dim=0).cpu().numpy()

    @torch.inference_mode()
    def embed_text(self, texts):
        clap = self._load_clap()
        inp = self.clap_proc(text=texts, return_tensors="pt", padding=True)
        e = clap.get_text_features(**{k: v.to(self.dev) for k, v in inp.items()})
        e = getattr(e, "pooler_output", e)
        return torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()


def tag(jobs_path):
    jobs = json.load(open(jobs_path))
    m = Models()
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(decode, j["path"]) for j in jobs[:8]]
        for i, j in enumerate(jobs):
            if i + 8 < len(jobs):
                futs.append(ex.submit(decode, jobs[i + 8]["path"]))
            entry = {"rel": j["rel"], "size": j["size"], "mtime": j["mtime"], "v": j["v"]}
            try:
                x = futs[i].result()
                futs[i] = None
                if len(x) == 0 or float(np.max(np.abs(x))) < 1e-5:
                    entry["silent"] = True
                else:
                    wins = windows(x)
                    p = m.tags(wins)
                    labels = m.labels
                    order = np.argsort(-p)
                    entry["tags"] = {labels[k]: round(float(p[k]), 3) for k in order if p[k] >= 0.02}
                    vi = [k for k, lab in enumerate(labels) if lab in VOCAL]
                    best = max(vi, key=lambda k: p[k])
                    entry["vocal"] = [round(float(p[best]), 3), labels[best]]
                    entry["clap"] = [round(float(v), 5) for v in m.embed_audio(wins)]
            except Exception as e:  # noqa: BLE001 - one bad file must not stop the run
                entry["error"] = f"{type(e).__name__}: {e}"
            tmp = j["out"] + ".part"
            with open(tmp, "w") as f:
                json.dump(entry, f, separators=(",", ":"))
            os.replace(tmp, j["out"])
            if (i + 1) % 50 == 0 or i + 1 == len(jobs):
                el = time.time() - t0
                print(f"  {i + 1}/{len(jobs)} in {el:.0f}s, ~{el / (i + 1) * (len(jobs) - i - 1):.0f}s left", flush=True)


def main():
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "fetch":
        m = Models()
        m._load_ast()
        m._load_clap()
        print("models ready")
    elif cmd == "tag":
        tag(args[0])
    elif cmd == "text":
        print(json.dumps([[round(float(v), 5) for v in e] for e in Models().embed_text(args)]))
    elif cmd == "embed":
        m = Models()
        print(json.dumps([[round(float(v), 5) for v in m.embed_audio(windows(decode(p)))] for p in args]))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
