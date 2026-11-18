"""scripts/browser_sample_check.sh 를 여러 번 돌린 결과를 reports/browser_sample_runs.json 으로 정리한다.

입력: 한 줄에 실행 하나씩, {"run": 번호, "check": 확인 결과 문장, "out": browser_sample_check.sh 의 마지막 줄 JSON}
실행: .venv/bin/python scripts/summarize_browser_runs.py 실행결과.jsonl [내부함수로_연_실행.jsonl]
둘째 파일(선택)은 같은 빌드를 버튼 대신 globalThis.__vision.startUrl() 로 바로 연 실행이다 (한 줄에 그 실행의 결과 JSON 문자열).
구간 경계는 sexy_move.mov 정답 구간과 같다. 8.2~19.5초의 정답은 13초쯤 떨어뜨릴 때 돌아간 흰 한 번(가운데 색 0, 90도 1)이다.
예전 기록(이전_기본, 20프레임 예외 설정)은 그대로 둔다.
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "browser_sample_runs.json"
SEGS = {"hold_8.2_19.5": (8.2, 19.5), "sexy_19.5_26.5": (19.5, 26.5), "regrip_26.5_34.5": (26.5, 34.5), "fufu_34.5_46": (34.5, 46.0)}
KEY = "지금_기본(늘_엄격, relaxWithin 0)"


def main() -> None:
    rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
    truth = json.loads(OUT.read_text()) if OUT.exists() else {}
    runs = []
    for r in rows:
        o = r["out"] if isinstance(r["out"], dict) else json.loads(r["out"])
        mv = o["moves"]
        runs.append({
            "run": r["run"], "scan_ok": bool(o.get("scanned")), "scan_t": o.get("scanT"), "fps": o.get("fps"),
            "errors": len(o.get("errs") or []), "moves_total": len(mv),
            **{k: sum(1 for m in mv if a <= m[0] < b) for k, (a, b) in SEGS.items()},
            # 떨어뜨릴 때 돌아간 흰 면 한 번 (13초쯤)을 잡았나
            "drop_turn_found": any(12.5 <= m[0] <= 15.0 and m[1] == 0 and m[2] == 1 for m in mv),
            "inserted_turns": len(o.get("backs") or []),
            "state_check": " ".join(x for x in r["check"].splitlines() if x.startswith("확인")),
            "mode": o.get("mode"), "level": o.get("level"), "moves": mv,
        })
    truth["설명"] = ("배포용 빌드(vite preview)에서 '카메라 · 영상' → '샘플 영상' 버튼으로 web/public/samples/sexy_move_46s.mp4 를 "
                   "처음부터 끝까지 재생하며 앱이 받은 회전(시각, 가운데 색, 90도 단위)을 센 결과 (scripts/browser_sample_check.sh, "
                   f"헤드리스 Chrome, M1 Pro, {datetime.date.today().isoformat()}). 구간 경계는 sexy_move.mov 정답 구간과 같다. "
                   "8.2~19.5초의 정답은 13초쯤 떨어뜨릴 때 돌아간 흰 한 번(가운데 색 0, 90도 1)이다. "
                   "state_check 는 끝에서 화면·코치가 아는 큐브가 추적기 상태와 같은지 본 결과다. "
                   "이전_기본은 2026-11-17 옛 빌드(내부 함수로 바로 연 확인)에서 잰 기록이다.")
    truth.setdefault("구간", {"hold_8.2_19.5": "정답 1 (떨어뜨림)", "sexy_19.5_26.5": "정답 24", "regrip_26.5_34.5": "정답 0", "fufu_34.5_46": "정답 약 5"})
    truth[KEY] = runs
    if len(sys.argv) > 2:  # 같은 빌드를 내부 함수로 바로 연 실행 (사람과 같은 길과 견주는 참고값)
        direct = []
        for i, line in enumerate(x for x in Path(sys.argv[2]).read_text().splitlines() if x.strip()):
            o = json.loads(json.loads(line))
            mv = o["moves"]
            direct.append({"run": o.get("run", i + 1), "scan_ok": bool(o.get("scanned")), "scan_t": o.get("scanT"), "fps": o.get("fps"),
                           "errors": len(o.get("errs") or []), "moves_total": len(mv),
                           **{k: sum(1 for m in mv if a <= m[0] < b) for k, (a, b) in SEGS.items()},
                           "drop_turn_found": any(12.5 <= m[0] <= 15.0 and m[1] == 0 and m[2] == 1 for m in mv), "moves": mv})
        truth["참고_같은_빌드_내부_함수로_연_실행"] = direct
    OUT.write_text(json.dumps(truth, ensure_ascii=False, indent=1))
    for x in runs:
        print(x["run"], x["scan_t"], x["moves_total"], [x[k] for k in SEGS], x["drop_turn_found"], x["inserted_turns"], x["state_check"][:60])


if __name__ == "__main__":
    main()
