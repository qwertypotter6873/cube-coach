"""입력 256 비교용 브라우저 실행(2026-11-18)을 모은다: reports/browser_runs_input256/ → reports/browser_input256_summary.json

모두 scripts/browser_sample_check.sh 로 '카메라 · 영상' → '샘플 영상' 버튼(사람과 같은 길)으로 46초 샘플을 처음부터 돌렸다.
  p1_192_k / p1_256_k : 192 빌드(그때의 main) 와 모든 단계 256 빌드를 번갈아 4번씩
  p2_192_k / p2_hyb_k : 192 빌드와 지금 빌드(면 읽기 192, 추적 256→48)를 번갈아 4번씩
  stall_*             : 지금 빌드 2번 + 192 빌드 1번, 추적으로 넘어가는 첫 프레임의 모델 시간 등 멈칫함 진단을 더한 실행
같은 빌드는 같은 길에서 같은 회전열을 자주 되풀이한다(독립 표본이 아니다). 그래서 이 실행들은 속도·면 읽기·오류·상태 일치 확인용이고,
회전 수는 참고로만 적는다(추적 비교는 시드를 바꾼 파이썬 벤치, README 5.2·5.4절).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from track_video import SEXY_MOVE_GT, SEXY_MOVE_STATE, lcs  # noqa: E402

D = ROOT / "reports" / "browser_runs_input256"
TRUTH = "".join(str("WRGYOB".index(c)) for c in SEXY_MOVE_STATE)
A, B = SEXY_MOVE_GT["A"][3], SEXY_MOVE_GT["B"][3]


def row(p: Path) -> dict:
    r = json.loads(json.loads(p.read_text().strip().splitlines()[-1]))
    mv = r["moves"]
    w = lambda a, b: [(m[1], m[2]) for m in mv if a <= m[0] < b]
    hold = [m for m in mv if 8.2 <= m[0] < 19.5]
    check = p.with_suffix(".err").read_text().strip().splitlines()[-1]
    return {"면_읽기_시각": r["scanT"], "면_읽기_정답": r["scanned"] == TRUTH,
            "들고_있기_판정(8.2~19.5초, 정답 1)": len(hold),
            "떨어뜨림_잡음": any(m[1] == 0 and m[2] == 1 and 12.5 <= m[0] <= 15.5 for m in hold),
            "섹시무브_판정/정답과_짝지음(19.5~27.5초, 24수)": [len(w(19.5, 27.5)), lcs(w(19.5, 27.5), A)],
            "F_U_F'_U'_앞부분_판정/짝지음(30.5~46초)": [len(w(30.5, 46.0)), lcs(w(30.5, 46.0), B)],
            "fps": r["fps"], "처리한_프레임": r["frames"], "건너뛴_프레임": r["skipped"], "오류": len(r["errs"]),
            "프레임당_ms": r.get("ms"), "상태_확인": check}


def main() -> None:
    out = {"설명": __doc__.strip(), "실행": {}}
    for p in sorted(D.glob("*.json")):
        out["실행"][p.stem] = row(p)
    (ROOT / "reports" / "browser_input256_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for k, v in out["실행"].items():
        print(k, v["면_읽기_시각"], "정답" if v["면_읽기_정답"] else "틀림", v["들고_있기_판정(8.2~19.5초, 정답 1)"],
              v["섹시무브_판정/정답과_짝지음(19.5~27.5초, 24수)"], v["fps"], v["처리한_프레임"], (v["프레임당_ms"] or {}).get("model50"))


if __name__ == "__main__":
    main()
