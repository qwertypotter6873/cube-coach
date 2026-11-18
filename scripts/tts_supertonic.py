"""Supertonic 3 (ONNX)로 한국어 안내 문장을 음성으로 만든다.

공식 예제(vendor/supertonic-src/py/helper.py)의 추론 과정을 읽고 필요한 부분만 다시 썼다:
  글자 → 길이 예측 → 글자 인코더 → 흐름 정합(flow matching) 몇 단계 → 보코더.
가중치: Hugging Face Supertone/supertonic-3 (OpenRAIL-M), vendor/supertonic-3 에 받아 둔다.

실행 예:
  .venv/bin/python scripts/tts_supertonic.py --voice F1 --out reports/tts/sample.wav "오른쪽 면을 위로 올려 주세요."
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "vendor" / "supertonic-3"


class Supertonic:
    def __init__(self, model_dir: Path = MODEL, threads: int = 4) -> None:
        onnx = model_dir / "onnx"
        self.cfg = json.loads((onnx / "tts.json").read_text())
        self.indexer = json.loads((onnx / "unicode_indexer.json").read_text())
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        load = lambda name: ort.InferenceSession(str(onnx / f"{name}.onnx"), so, providers=["CPUExecutionProvider"])
        self.dp, self.enc, self.vf, self.voc = (load(n) for n in ("duration_predictor", "text_encoder", "vector_estimator", "vocoder"))
        self.sr = self.cfg["ae"]["sample_rate"]
        self.chunk = self.cfg["ae"]["base_chunk_size"] * self.cfg["ttl"]["chunk_compress_factor"]
        self.ldim = self.cfg["ttl"]["latent_dim"] * self.cfg["ttl"]["chunk_compress_factor"]
        self.styles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.model_dir = model_dir

    def style(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        if name not in self.styles:
            d = json.loads((self.model_dir / "voice_styles" / f"{name}.json").read_text())
            ttl = np.array(d["style_ttl"]["data"], dtype=np.float32).reshape(1, *d["style_ttl"]["dims"][1:])
            dp = np.array(d["style_dp"]["data"], dtype=np.float32).reshape(1, *d["style_dp"]["dims"][1:])
            self.styles[name] = (ttl, dp)
        return self.styles[name]

    def _ids(self, text: str, lang: str) -> np.ndarray:
        text = unicodedata.normalize("NFKD", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not re.search(r"[.!?,]$", text):
            text += "."
        text = f"<{lang}>{text}</{lang}>"
        return np.array([[self.indexer[ord(c)] for c in text]], dtype=np.int64)

    def __call__(self, text: str, voice: str = "F1", lang: str = "ko", steps: int = 8, speed: float = 1.05, seed: int = 0) -> np.ndarray:
        ttl, dp = self.style(voice)
        ids = self._ids(text, lang)
        mask = np.ones((1, 1, ids.shape[1]), dtype=np.float32)
        dur = self.dp.run(None, {"text_ids": ids, "style_dp": dp, "text_mask": mask})[0] / speed
        emb = self.enc.run(None, {"text_ids": ids, "style_ttl": ttl, "text_mask": mask})[0]
        wav_len = int(dur[0] * self.sr)
        n_lat = (wav_len + self.chunk - 1) // self.chunk
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((1, self.ldim, n_lat)).astype(np.float32)
        lmask = np.ones((1, 1, n_lat), dtype=np.float32)
        total = np.array([steps], dtype=np.float32)
        for i in range(steps):
            x = self.vf.run(None, {
                "noisy_latent": x, "text_emb": emb, "style_ttl": ttl, "text_mask": mask,
                "latent_mask": lmask, "current_step": np.array([i], dtype=np.float32), "total_step": total,
            })[0]
        wav = self.voc.run(None, {"latent": x})[0][0]
        return wav[:wav_len]


def save_wav(path: Path, wav: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--voice", default="F1")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--speed", type=float, default=1.05)
    ap.add_argument("--out", default=str(ROOT / "reports" / "tts" / "sample.wav"))
    args = ap.parse_args()
    tts = Supertonic()
    t = time.perf_counter()
    wav = tts(args.text, args.voice, steps=args.steps, speed=args.speed)
    dt = time.perf_counter() - t
    save_wav(Path(args.out), wav, tts.sr)
    print(f"{args.out}: {len(wav) / tts.sr:.2f}초 분량, 합성 {dt:.2f}초 (실시간의 {len(wav) / tts.sr / dt:.1f}배)")


if __name__ == "__main__":
    main()
