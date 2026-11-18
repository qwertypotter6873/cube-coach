"""안내 음성 후보 비교: 같은 코치 문장을 여러 목소리로 합성하고 속도와 명료도를 잰다.

- 속도: 합성에 걸린 시간 대비 만들어진 음성 길이 (실시간의 몇 배인가)
- 명료도: 음성 인식기(Whisper large-v3-turbo)로 다시 받아 적었을 때의 글자 오류율(CER).
  사람이 듣는 자연스러움(MOS)을 대신하지는 못하지만, 발음이 뭉개지거나 단어를 빠뜨리는 목소리를 걸러낸다.

결과
  reports/tts_compare.json            수치
  web/public/voices/*.m4a, voices.json  들어 보기 페이지(web/voices.html)용 음성

실행: .venv/bin/python scripts/tts_compare.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tts_supertonic import Supertonic, save_wav  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "reports" / "tts" / "compare"
PUBLIC = ROOT / "web" / "public" / "voices"

# 코치가 실제로 하는 말 가운데 종류별로 골랐다 (지시 / 바로잡기 / 설명 / 정리)
SENTENCES = [
    "오른쪽 면을 위로 올려 주세요.",
    "윗면을 왼쪽으로 돌려 주세요.",
    "앞면을 반시계 방향으로 돌려 주세요.",
    "잠깐, 반대 방향이에요!",
    "앗, 방금은 왼쪽 면을 돌렸어요. 먼저 되돌릴게요. 왼쪽 면을 위로 올려 주세요.",
    "두 번째. 섹시무브.",
    "노랑이 위, 초록이 앞에 오게 잡아 주세요.",
    "흰색, 주황, 파랑 꼭짓점이 아래층 다른 자리에 끼어 있어요. 섹시무브 한 번으로 먼저 꺼낼게요.",
    "지금은 ㄱ자 모양이에요. 노란 십자 공식을 할게요.",
    "흰색 십자 완성! 다음은 1층 꼭짓점이에요.",
    "PLL 제이비 공식이에요.",
    "완성! 총 일 분 사십이 초, 여든여섯 수 걸렸어요.",
]

SUPERTONIC_VOICES = ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"]


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    return re.sub(r"[^가-힣a-zA-Z0-9]", "", s).lower()


def cer(ref: str, hyp: str) -> float:
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, b in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (a != b))
    return d[-1] / max(len(r), 1)


def wav_duration(path: Path) -> float:
    out = subprocess.run(["afinfo", str(path)], capture_output=True, text=True).stdout
    m = re.search(r"estimated duration: ([\d.]+)", out)
    return float(m.group(1)) if m else 0.0


def to_m4a(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-ac", "1", "-ar", "24000", "-c:a", "aac", "-b:a", "40k", str(dst)], check=True)


def main() -> None:
    import mlx_whisper

    systems: dict[str, dict] = {}
    tts = Supertonic(threads=4)
    tts("준비.", "F1", steps=2)  # 첫 실행 준비

    for voice in SUPERTONIC_VOICES:
        name = f"supertonic3-{voice}"
        rows = []
        for i, text in enumerate(SENTENCES):
            t = time.perf_counter()
            wav = tts(text, voice, steps=8, speed=1.05, seed=i)
            dt = time.perf_counter() - t
            path = WORK / name / f"{i:02d}.wav"
            save_wav(path, wav, tts.sr)
            rows.append({"i": i, "path": str(path.relative_to(ROOT)), "audio_s": len(wav) / tts.sr, "synth_s": dt})  # 저장소 기준 경로
        systems[name] = {"rows": rows, "kind": "Supertonic 3 (99M, ONNX, 온디바이스)", "voice": voice}
        print(f"{name}: 합성 완료", flush=True)

    # 기준선: macOS 내장 한국어 음성 (브라우저 Web Speech가 이 목소리를 쓰는 경우가 많다)
    rows = []
    for i, text in enumerate(SENTENCES):
        path = WORK / "macos-Yuna" / f"{i:02d}.aiff"
        path.parent.mkdir(parents=True, exist_ok=True)
        t = time.perf_counter()
        subprocess.run(["say", "-v", "Yuna", "-o", str(path), text], check=True)
        dt = time.perf_counter() - t
        rows.append({"i": i, "path": str(path.relative_to(ROOT)), "audio_s": wav_duration(path), "synth_s": dt})
    systems["macos-Yuna"] = {"rows": rows, "kind": "macOS 내장 음성 (브라우저 기본 음성의 예)", "voice": "Yuna"}
    print("macos-Yuna: 합성 완료", flush=True)

    # 명료도: 음성 인식으로 받아 적기
    for name, sysd in systems.items():
        for r in sysd["rows"]:
            out = mlx_whisper.transcribe(str(ROOT / r["path"]), path_or_hf_repo="mlx-community/whisper-large-v3-turbo", language="ko", temperature=0.0)
            r["asr"] = out["text"].strip()
            r["cer"] = cer(SENTENCES[r["i"]], r["asr"])
        c = np.array([r["cer"] for r in sysd["rows"]])
        sysd["cer_mean"] = float(c.mean())
        sysd["rtf"] = float(sum(r["audio_s"] for r in sysd["rows"]) / sum(r["synth_s"] for r in sysd["rows"]))
        print(f"{name:<18} CER {sysd['cer_mean']:.3f}  실시간의 {sysd['rtf']:.1f}배", flush=True)

    # 웹 들어 보기용
    manifest = {"sentences": SENTENCES, "systems": []}
    for name, sysd in systems.items():
        files = []
        for r in sysd["rows"]:
            dst = PUBLIC / f"{name}_{r['i']:02d}.m4a"
            to_m4a(ROOT / r["path"], dst)
            files.append(f"voices/{dst.name}")
        manifest["systems"].append({"name": name, "kind": sysd["kind"], "cer": sysd["cer_mean"], "rtf": sysd["rtf"], "files": files})
    (PUBLIC.parent / "voices.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    (ROOT / "reports" / "tts_compare.json").write_text(json.dumps(systems, ensure_ascii=False, indent=1))

    print("\n=== 받아 적기가 틀린 문장 (CER > 0.1) ===")
    for name, sysd in systems.items():
        for r in sysd["rows"]:
            if r["cer"] > 0.1:
                print(f"{name} #{r['i']}: '{SENTENCES[r['i']]}' → '{r['asr']}' ({r['cer']:.2f})")


if __name__ == "__main__":
    main()
