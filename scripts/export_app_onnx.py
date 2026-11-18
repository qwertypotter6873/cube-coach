"""학습한 가중치에서 앱이 읽는 인식 모델 ONNX 를 다시 내보낸다 (models/cube_seg_net.onnx 와 앱 사본 web/public/models/cube_seg_net.onnx).

모델은 192x192 합성 그림으로 학습했지만 합성곱만이라 입력 크기를 바꿀 수 있다. 앱은 면 읽기 단계에 192x192를,
추적 단계에 256x256을 넣고(칸이 커져 실제 프레임 정확도가 오른다), 나온 64x64 지도를 48x48로 줄여 쓴다
(web/src/vision/seg.ts, maps.ts). 그래서 입력 크기를 열어 두고 192·256 모두에서 torch 와 같은지 확인한다
(scripts/train_seg.py 와 같은 함수 cubecoach.segnet.export_onnx). 두 파일은 같은 바이트다.
실행: .venv/bin/python scripts/export_app_onnx.py
"""

import argparse
import hashlib
import shutil
from pathlib import Path

from cubecoach.segnet import export_onnx, load_segnet

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models" / "cube_seg_net.pt"))
    ap.add_argument("--out", nargs="+", default=[str(ROOT / "models" / "cube_seg_net.onnx"), str(ROOT / "web" / "public" / "models" / "cube_seg_net.onnx")])
    args = ap.parse_args()
    first = Path(args.out[0])
    diff = export_onnx(load_segnet(args.weights), first)
    for o in args.out[1:]:
        shutil.copyfile(first, o)
    for o in args.out:
        h = hashlib.sha256(Path(o).read_bytes()).hexdigest()
        print(f"{o}: {Path(o).stat().st_size / 1024:.0f}KB, SHA-256 {h[:16]}, torch 와 최대 차이 {diff:.1e} (입력 192·256)")


if __name__ == "__main__":
    main()
