"""코치 대사를 미리 음성 파일로 만든다 (실시간 안내를 지연 없이 내려고).

입력: reports/tts_inventory_train.json — web/test/inventory.ts 가 모은 조각 목록 [{text, tts?, n}]
  text: 앱이 찾는 열쇠 글 / tts: 모델에 넣을 글 (없으면 text. 이어지는 조각은 끝에 쉼표가 붙어 있다)
출력: web/public/tts/<목소리>/<내용 해시>.m4a, manifest.json { text: 파일 }, meta.json

- 파일 이름은 (목소리, 넣은 글, 설정, 자르기 방식)의 해시다. 같은 내용은 다시 만들지 않고,
  설정을 바꾸면 이름이 달라져 저절로 새로 만든다.
- 앞뒤 무음을 잘라 둔다. 조각을 이어 붙일 때 사이가 벌어지지 않게 하려는 것이고, 쉼은 앱이 넣는다.
- 만드는 동안에도 manifest 를 조금씩 고쳐 써서 앱이 바로 쓸 수 있다.
  목록에서 빠진 문장과 파일은 끝까지 잘 마친 뒤에만 지운다.

실행: .venv/bin/python scripts/tts_build_clips.py --voice F2   (앱의 목소리: F2 여성, M1 남성)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tts_supertonic import Supertonic, save_wav  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TTS = ROOT / "web" / "public" / "tts"
MODEL_NAME = "Supertone/supertonic-3 (OpenRAIL-M)"

# 앞뒤 무음 자르기. 값을 바꾸면 TRIM["v"] 도 올린다 (파일 이름이 바뀌어 다시 만든다).
TRIM = {
    "v": 1,
    "speech_db": 30,  # 봉우리보다 이만큼 작은 데까지는 확실한 말소리
    "onset_db": 50,  # 말 앞쪽은 이만큼까지 거슬러 올라간다 (ㅅ·ㅎ·ㅊ 같은 약한 첫소리를 살리려고)
    "offset_db": 45,  # 말 뒤쪽은 이만큼까지 따라간다
    "pad_in": 0.015,  # 초
    "pad_out": 0.04,
    "fade_in": 0.005,
    "fade_out": 0.02,
}


def trim_silence(wav: np.ndarray, sr: int, p: dict = TRIM) -> np.ndarray:
    """앞뒤 무음과 끝에 남는 잔소리를 잘라 낸다. 5ms 마디의 소리 세기(dB)로 판단한다."""
    hop = int(sr * 0.005)
    n = len(wav) // hop
    if n < 3:
        return wav
    e = 20 * np.log10(np.sqrt((wav[: n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9)
    top = e.max()
    loud = np.where(e > top - p["speech_db"])[0]
    a, b = loud[0], loud[-1]
    while a > 0 and e[a - 1] > top - p["onset_db"]:
        a -= 1
    while b < n - 1 and e[b + 1] > top - p["offset_db"]:
        b += 1
    s = max(0, a * hop - int(p["pad_in"] * sr))
    t = min(len(wav), (b + 1) * hop + int(p["pad_out"] * sr))
    y = wav[s:t].astype(np.float32).copy()
    fi, fo = int(p["fade_in"] * sr), int(p["fade_out"] * sr)
    y[:fi] *= (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fi))).astype(np.float32)
    y[-fo:] *= (0.5 + 0.5 * np.cos(np.linspace(0, np.pi, fo))).astype(np.float32)
    return y


# 짧은 조각은 조금 천천히 읽힌다. 1.08배로 읽으면 F2가 "팔 초."의 "초", "준비 끝."의 "끝"을 삼켰다.
# 받아 적기(whisper)로 확인한 글자 오류율, 1.08배 → 0.95배:
#   2~3음절 F2 0.18 → 0.08, M1 0.23 → 0.00 / 4~5음절 F2 0.05 → 0.03, M1 같음 / F2 "N 초." 6개 중 1개 → 6개 모두 "초"가 들림
# 6~7음절은 차이가 작아(F2 15개 중 1개 → 0개) 속도가 갑자기 바뀌지 않게 중간값 1.0으로 둔다.
# 8음절 넘는 문장은 원래 속도(--speed)를 그대로 쓴다. 한글 음절 수 기준, 두 목소리 같은 규칙.
SHORT_SPEED = [(5, 0.95), (7, 1.0)]


# 받아 적기 검사(자주 나오는 조각 200개)에서 뭉개져 들린 조각만 다른 잡음 씨앗으로 다시 만든다 (글·속도는 같다).
# F2 씨앗 0: "디 프라임, 아랫면 왼쪽으로." → "디프라인 RM이 왼쪽으로", "디, 아랫면 오른쪽으로." → "뒤 아랫면…"
SEED_FIX: dict[str, dict[str, int]] = {"F2": {"디 프라임, 아랫면 왼쪽으로.": 1, "디, 아랫면 오른쪽으로.": 1}}


def speed_for(tts_text: str, base: float) -> float:
    n = len(re.findall(r"[가-힣]", tts_text))
    for most, speed in SHORT_SPEED:
        if n <= most:
            return min(speed, base)
    return base


def clip_name(voice: str, tts_text: str, steps: int, speed: float, seed: int) -> str:
    key = f"{voice}|{steps}|{speed}|{seed}|trim{TRIM['v']}|{tts_text}"
    return hashlib.sha1(key.encode()).hexdigest()[:16] + ".m4a"


def encode_m4a(wav: np.ndarray, sr: int, dst: Path, tmp: Path) -> None:
    """wav → m4a. 다 쓴 다음에 옮겨 놓는다 (중간에 멈춰도 반쯤 쓴 파일이 남지 않게)."""
    w = tmp / "x.wav"
    part = dst.with_name(dst.name + ".part")
    save_wav(w, wav, sr)
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(w), "-ac", "1", "-ar", "24000", "-c:a", "aac", "-b:a", "40k", "-f", "ipod", str(part)],
        check=True,
    )
    os.replace(part, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="F2")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--speed", type=float, default=1.08)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--inventory", default=str(ROOT / "reports" / "tts_inventory_train.json"))
    ap.add_argument("--out", default="", help="출력 폴더 (기본: web/public/tts/<목소리>)")
    ap.add_argument("--no-prune", action="store_true", help="목록에 없는 파일을 지우지 않는다")
    args = ap.parse_args()

    items = json.loads(Path(args.inventory).read_text())
    want = {x["text"]: x.get("tts") or x["text"] for x in items}
    out = Path(args.out) if args.out else TTS / args.voice
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "manifest.json"
    old: dict[str, str] = json.loads(mpath.read_text()) if mpath.exists() else {}

    speeds = {text: speed_for(tts, args.speed) for text, tts in want.items()}
    seeds = {text: SEED_FIX.get(args.voice, {}).get(text, args.seed) for text in want}
    names = {text: clip_name(args.voice, tts, args.steps, speeds[text], seeds[text]) for text, tts in want.items()}
    # 만드는 동안 쓰는 manifest: 이미 있는 파일은 바로 새 이름으로, 아직 없는 것은 옛 파일로 둔다
    manifest = {t: f for t, f in old.items() if (out / f).exists()}
    manifest.update({t: f for t, f in names.items() if (out / f).exists()})
    todo = [t for t in want if not (out / names[t]).exists()]
    print(f"조각 {len(want)}개 중 새로 만들 것 {len(todo)}개 (목소리 {args.voice})", flush=True)

    t0 = time.time()
    audio_s = 0.0
    if todo:
        tts = Supertonic(threads=args.threads)
        with tempfile.TemporaryDirectory() as tmp:
            for i, text in enumerate(todo, 1):
                if (out / names[text]).exists():  # 넣을 글이 같은 조각은 파일 하나를 같이 쓴다
                    manifest[text] = names[text]
                    continue
                raw = tts(want[text], args.voice, steps=args.steps, speed=speeds[text], seed=seeds[text])
                wav = trim_silence(raw, tts.sr)
                audio_s += len(wav) / tts.sr
                encode_m4a(wav, tts.sr, out / names[text], Path(tmp))
                manifest[text] = names[text]
                if i % 25 == 0 or i == len(todo):
                    mpath.write_text(json.dumps(manifest, ensure_ascii=False))
                    rate = i / (time.time() - t0)
                    print(f"  {i}/{len(todo)}  ({rate:.1f}개/초, 남은 시간 약 {(len(todo) - i) / rate / 60:.1f}분)", flush=True)

    # 다 만들었다: 지금 목록에 있는 것만 남긴다
    manifest = {t: names[t] for t in want}
    mpath.write_text(json.dumps(manifest, ensure_ascii=False))
    meta = {"voice": args.voice, "steps": args.steps, "speed": args.speed, "short_speed": SHORT_SPEED, "seed": args.seed, "seed_fix": SEED_FIX.get(args.voice, {}), "trim": TRIM, "model": MODEL_NAME}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
    removed = 0
    if not args.no_prune:
        keep = set(manifest.values())
        for f in [*out.glob("*.m4a"), *out.glob("*.part")]:
            if f.name not in keep:
                f.unlink()
                removed += 1
    size = sum((out / f).stat().st_size for f in set(manifest.values()))
    took = time.time() - t0
    print(f"완료: 조각 {len(manifest)}개, 새로 만든 것 {len(todo)}개 ({audio_s / 60:.1f}분 분량, {took / 60:.1f}분 걸림), "
          f"안 쓰는 파일 {removed}개 지움, 전체 {size / 1e6:.1f}MB")


if __name__ == "__main__":
    main()
