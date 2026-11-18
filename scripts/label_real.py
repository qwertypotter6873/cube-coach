"""실제 촬영 영상의 평가용 정답 만들기: 프레임마다 '보이는 칸의 중심과 색'을 클릭으로 찍는다.

합성 데이터로만 학습한 모델이 실제 화면에서 얼마나 맞히는지 재려면 실제 정답이 필요하다.
칸 하나를 픽셀 단위로 칠하는 대신 '칸 중심 한 점 + 색'만 찍는다 (프레임당 1분 남짓).
프레임은 큐브 둘레를 정사각으로 잘라(큐브가 절반쯤 차게, 추적 중 관심영역과 비슷하게) 크게 보여 준다.

  1) 프레임 고르기:  .venv/bin/python scripts/label_real.py extract data/raw/cube_unsolved.mov --start 300
     학습 배경 조각에 쓰지 않은 뒷부분(300초 이후)에서 선명한 프레임을 고르게 뽑는다.
  2) 찍기:           .venv/bin/python scripts/label_real.py serve   → http://127.0.0.1:5181
     결과는 data/real/labels.json 에 바로바로 저장된다.

원본 영상과 프레임에는 촬영자가 나오므로 저장소에 올리지 않는다 (data/real/ 은 .gitignore).
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data" / "real" / "label_frames"
LABELS = ROOT / "data" / "real" / "labels.json"


def extract(video: str, start: float, end: float | None, n: int, gap: float) -> None:
    cap = cv2.VideoCapture(video)
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    end = min(end or dur, dur - 0.5)
    cands = []
    t = start
    while t < end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok:
            break
        small = cv2.resize(fr, (480, 320), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        cube = (hsv[..., 1] > 150) & (hsv[..., 2] > 110)  # 큐브 칸처럼 채도 높은 곳
        if cube.sum() > 300:
            ys, xs = np.nonzero(cube)
            y0, y1 = np.percentile(ys, [2, 98]).astype(int)
            x0, x1 = np.percentile(xs, [2, 98]).astype(int)
            box = small[max(y0 - 10, 0):y1 + 10, max(x0 - 10, 0):x1 + 10]
            sharp = cv2.Laplacian(cv2.cvtColor(box, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            # 큐브 둘레의 정사각 영역 (원본 너비 기준 비율): 큐브가 영역의 절반쯤 차도록
            k = fr.shape[1] / 480
            side = max(x1 - x0, y1 - y0) * 2.0 * k
            cands.append((sharp, t, fr, ((x0 + x1) / 2 * k, (y0 + y1) / 2 * k, side)))
        t += 0.25
    # 선명한 순서로, 서로 gap초 이상 떨어진 것만
    cands.sort(key=lambda c: -c[0])
    picked: list[tuple[float, np.ndarray, tuple[float, float, float]]] = []
    for _, t, fr, box in cands:
        if all(abs(t - p) >= gap for p, _, _ in picked):
            picked.append((t, fr, box))
        if len(picked) >= n:
            break
    picked.sort(key=lambda p: p[0])
    DIR.mkdir(parents=True, exist_ok=True)
    meta = []
    for i, (t, fr, (cx, cy, side)) in enumerate(picked):
        h, w = fr.shape[:2]
        # 화면 밖은 검게 채워 정사각으로 자른다
        pad = int(side)
        big = cv2.copyMakeBorder(fr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        x0, y0 = int(cx - side / 2) + pad, int(cy - side / 2) + pad
        crop = big[y0:y0 + int(side), x0:x0 + int(side)]
        img = cv2.resize(crop, (720, 720), interpolation=cv2.INTER_AREA)
        name = f"f{i:02d}_{t:06.2f}.jpg"
        cv2.imwrite(str(DIR / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        # roi = 추적기와 같은 표기 (중심 x, 중심 y, 한 변; 원본 너비 기준, 화면 중심이 0)
        roi = [(cx - w / 2) / w, (cy - h / 2) / w, side / w]
        meta.append({"file": name, "t": round(t, 2), "video": Path(video).name, "roi": [round(v, 5) for v in roi]})
    (DIR / "frames.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"후보 {len(cands)}장 중 {len(picked)}장 → {DIR}")


PAGE = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>칸 정답 찍기</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#1b1b1a;color:#eee}
header{display:flex;gap:14px;align-items:center;padding:10px 14px;flex-wrap:wrap}
#wrap{position:relative;display:inline-block;margin:0 14px;cursor:crosshair}
#wrap img{display:block;max-width:calc(100vw - 28px);max-height:calc(100vh - 150px)}
.pt{position:absolute;width:14px;height:14px;margin:-9px 0 0 -9px;border-radius:50%;border:2px solid #000;box-shadow:0 0 0 2px #fff;pointer-events:none}
.sw{display:inline-flex;align-items:center;gap:4px;padding:4px 8px;border-radius:6px;border:2px solid transparent;cursor:pointer}
.sw.on{border-color:#fff}.sw i{width:16px;height:16px;border-radius:4px;display:inline-block}
kbd{background:#333;border-radius:4px;padding:1px 5px}
.help{font-size:13px;line-height:1.6;padding:0 14px 14px;color:#bbb;max-width:980px}
</style></head><body>
<header><b id="title"></b><span id="cols"></span><span id="count"></span><span id="saved"></span></header>
<div class="help">
<b>찍는 법</b> — 화면에 보이는 <b>모든 칸(스티커) 하나하나</b>에 점을 찍습니다. 센터만이 아니라 모서리·꼭짓점 칸까지 전부요.
보통 2~3면이 보이므로 프레임당 10~20점이 됩니다. 각 칸의 <b>정중앙</b>을 클릭하고, 클릭하기 전에 색을 고릅니다:
<kbd>W</kbd>흰 <kbd>R</kbd>빨 <kbd>G</kbd>초 <kbd>Y</kbd>노 <kbd>O</kbd>주 <kbd>B</kbd>파 (또는 <kbd>1</kbd>~<kbd>6</kbd>).
흰 센터의 로고 칸은 흰색입니다. 정중앙이 손가락에 가려진 칸, 너무 비스듬해서 색이 애매한 칸, 심하게 번진 칸은 건너뜁니다.
이미 찍어 둔 점(센터)은 그대로 두고 나머지 칸을 더하면 됩니다.<br>
<kbd>Backspace</kbd> 마지막 점 지우기 · 점 위에서 <kbd>Shift</kbd>+클릭 그 점 지우기 · <kbd>→</kbd>/<kbd>N</kbd> 다음 · <kbd>←</kbd>/<kbd>P</kbd> 이전 ·
<kbd>X</kbd> 이 프레임 빼기(큐브가 거의 안 보이거나 다 번졌을 때). 바꿀 때마다 자동 저장됩니다.
</div>
<div id="wrap"><img id="img"></div>
<script>
const NAMES=["흰","빨","초","노","주","파"],HEX=["#f4f4f0","#d6233a","#04a846","#e9e02c","#fa6a2c","#0464bc"],KEYS="wrgyob";
let frames=[],labels={},i=0,cur=0;
const $=id=>document.getElementById(id);
$("cols").innerHTML=NAMES.map((n,c)=>`<span class="sw" data-c="${c}"><i style="background:${HEX[c]}"></i>${n}</span>`).join("");
document.querySelectorAll(".sw").forEach(el=>el.onclick=()=>{cur=+el.dataset.c;draw()});
async function save(){const r=await fetch("/labels",{method:"POST",body:JSON.stringify(labels)});$("saved").textContent=r.ok?"저장됨":"저장 실패";}
function rec(){const f=frames[i].file;return labels[f]??={points:[],skip:false};}
function draw(){
  const f=frames[i],L=rec();
  $("title").textContent=`${i+1}/${frames.length}  ${f.file}${L.skip?"  (빼기)":""}`;
  document.querySelectorAll(".sw").forEach(el=>el.classList.toggle("on",+el.dataset.c===cur));
  document.querySelectorAll(".pt").forEach(e=>e.remove());
  const img=$("img");
  for(const p of L.points){const d=document.createElement("div");d.className="pt";d.style.left=(p.x*img.clientWidth)+"px";d.style.top=(p.y*img.clientHeight)+"px";d.style.background=HEX[p.c];$("wrap").appendChild(d);}
  const done=Object.values(labels).filter(l=>l.skip||l.points.length).length;
  $("count").textContent=`점 ${L.points.length}개 · 끝낸 프레임 ${done}/${frames.length}`;
}
function go(k){i=Math.max(0,Math.min(frames.length-1,i+k));$("img").src="/frame/"+frames[i].file;}
$("img").onload=draw;
$("wrap").onclick=e=>{const r=$("img").getBoundingClientRect();const x=(e.clientX-r.left)/r.width,y=(e.clientY-r.top)/r.height;const L=rec();
  if(e.shiftKey){let b=-1,bd=1e9;L.points.forEach((p,j)=>{const d=(p.x-x)**2+(p.y-y)**2;if(d<bd){bd=d;b=j}});if(b>=0&&bd<0.0004)L.points.splice(b,1);}
  else L.points.push({x:+x.toFixed(4),y:+y.toFixed(4),c:cur});L.skip=false;draw();save();};
addEventListener("keydown",e=>{const k=e.key.toLowerCase();
  if(KEYS.includes(k)&&k.length===1){cur=KEYS.indexOf(k);draw();}
  else if("123456".includes(k)&&k.length===1){cur=+k-1;draw();}
  else if(k==="backspace"){rec().points.pop();draw();save();e.preventDefault();}
  else if(k==="arrowright"||k==="n")go(1);else if(k==="arrowleft"||k==="p")go(-1);
  else if(k==="x"){const L=rec();L.skip=!L.skip;draw();save();}});
(async()=>{frames=await (await fetch("/frames")).json();labels=await (await fetch("/labels")).json();go(0);})();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/frames":
            self._send(200, (DIR / "frames.json").read_bytes(), "application/json")
        elif self.path == "/labels":
            self._send(200, LABELS.read_bytes() if LABELS.exists() else b"{}", "application/json")
        elif self.path.startswith("/frame/"):
            p = (DIR / Path(self.path[len("/frame/"):]).name)
            if p.suffix == ".jpg" and p.exists():
                self._send(200, p.read_bytes(), "image/jpeg")
            else:
                self._send(404, b"", "text/plain")
        else:
            self._send(404, b"", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/labels":
            return self._send(404, b"", "text/plain")
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        LABELS.write_text(json.dumps(data, ensure_ascii=False))
        self._send(200, b"ok", "text/plain")

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("video")
    e.add_argument("--start", type=float, default=300.0)
    e.add_argument("--end", type=float, default=None)
    e.add_argument("--n", type=int, default=40)
    e.add_argument("--gap", type=float, default=2.0)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=5181)
    args = ap.parse_args()
    if args.cmd == "extract":
        extract(args.video, args.start, args.end, args.n, args.gap)
    else:
        print(f"http://127.0.0.1:{args.port}  (끝내려면 Ctrl+C)")
        ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
