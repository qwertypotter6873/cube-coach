"""입력 크기를 열어 둔 인식 모델 ONNX 를 내보낸다 (입력 256 실험 갈래용, README 5.2절).

모델은 192x192 합성 그림으로 학습했지만 합성곱만이라 입력 크기를 바꿀 수 있다. 실험 갈래(커밋 5154035)는 면 읽기 단계에
192x192를, 추적 단계에 256x256을 넣고 나온 64x64 지도를 48x48로 줄여 썼다(web/src/vision/maps.ts). 시험셋에서 나빠서
지금 앱은 넣지 않았고, 앱 사본(web/public/models/cube_seg_net.onnx)은 192 고정으로 내보낸 원래 파일 그대로다.
192·256 모두에서 torch 와 같은지 확인한다(scripts/train_seg.py 와 같은 함수 cubecoach.segnet.export_onnx).
실행: .venv/bin/python scripts/export_app_onnx.py [--out 경로 ...]   (기본은 models/exp/ 에만 쓴다)
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
    ap.add_argument("--out", nargs="+", default=[str(ROOT / "models" / "exp" / "cube_seg_net_dyn.onnx")])
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
