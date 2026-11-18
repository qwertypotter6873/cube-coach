"""합성 영상 하나를 프레임 단위로 따라가며 추적기 상태를 출력한다 (정답 지도 사용).

실행: .venv/bin/python scripts/debug_track.py [--seed 5000] [--moves 6] [--fingers] [--all]
"""

from __future__ import annotations

import argparse

import numpy as np

from cubecoach.cube import FACES
from cubecoach.geometry import rotation_angle
from cubecoach.render3d import N_CLASSES, render
from cubecoach.simseq import simulate
from cubecoach.tracker import Tracker


def oracle(fr):
    prob = np.full((N_CLASSES,) + fr.seg.shape, 0.03 / (N_CLASSES - 1), dtype=np.float32)
    np.put_along_axis(prob, fr.seg[None].astype(np.int64), 0.97, axis=0)
    return np.log(prob), fr.heat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--moves", type=int, default=6)
    ap.add_argument("--fingers", action="store_true")
    ap.add_argument("--all", action="store_true", help="모든 프레임 출력 (기본은 사건이 있는 프레임만)")
    args = ap.parse_args()

    rng, noise = np.random.default_rng(args.seed), np.random.default_rng(1)
    tr = None
    for i, step in enumerate(simulate(rng, args.moves, fingers=args.fingers)):
        if tr is None:
            tr = Tracker(step.scene.colors, f=0.95, seed=0)
        roi = tr.roi()
        logp, heat = oracle(render(step.scene, roi, noise))
        events = tr.update(logp, heat, roi)
        err = rotation_angle(step.scene.R, tr.R) if tr.R is not None else -1.0
        tag = ""
        if step.scene.turn:
            tag += f"  실제: {step.scene.turn[0]} {np.degrees(step.scene.turn[1]):+.0f}도"
        if tr.layer >= 0:
            tag += f"  추적: {FACES[tr.layer]} {np.degrees(tr.phi):+.0f}도"
        if step.done:
            tag += f"  ★정답 {step.done}"
        if events:
            tag += f"  ●판정 {[(e.center, e.quarter_turns) for e in events]}"
        if args.all or tag:
            print(f"f{i:3d} 자세오차 {err:5.1f}도 품질 {tr.quality:.2f}{tag}")


if __name__ == "__main__":
    main()
