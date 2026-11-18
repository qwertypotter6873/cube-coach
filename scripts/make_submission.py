"""제출 묶음 만들기: 포트폴리오 PDF(README와 같음) + 소스코드 + submission.json → submission/기말-학번-이름.zip

1. GitHub 가 그린 README(지금 커밋, 저장소에 올라가 있어야 한다)를 받아 A4 PDF 로 찍는다 (Chrome headless).
   그림은 이 저장소의 파일을 쓰고, 파일 링크는 같은 커밋의 GitHub 주소로 바꾼다. 표·코드가 잘리지 않게 줄을 바꾼다.
2. 같은 커밋을 git archive 로 묶는다. 미리 합성한 음성 조각(web/public/tts, 58MB)과 음성 비교 녹음(reports/tts, 43MB)은
   용량 때문에 빼고(안내: 100MB 넘는 자료는 링크로), 공개 저장소에 있다고 제출_안내.txt 에 적는다.
README 를 고치면 커밋·push 한 뒤 다시 돌린다.
실행: .venv/bin/python scripts/make_submission.py
"""

from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission"
REPO = "qwertypotter6873/cube-coach"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CSS_URL = "https://cdn.jsdelivr.net/npm/github-markdown-css@5.8.1/github-markdown-light.css"
EXCLUDE = ("web/public/tts/", "reports/tts/")


def sh(*args: str, **kw) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True, cwd=ROOT, **kw).stdout


def main() -> None:
    meta = json.loads((ROOT / "submission.json").read_text())
    sid, name = meta["student_id"], meta["name"]
    sha = sh("git", "rev-parse", "HEAD").strip()
    if sh("git", "status", "--porcelain", "README.md", "submission.json").strip():
        raise SystemExit("README.md 나 submission.json 에 커밋하지 않은 변경이 있다. 커밋·push 한 뒤 다시 돌린다.")
    remote = sh("git", "ls-remote", "origin", "refs/heads/main").split()[0]
    if remote != sha:
        raise SystemExit(f"origin/main({remote[:7]})이 지금 커밋({sha[:7]})과 다르다. push 한 뒤 다시 돌린다.")
    OUT.mkdir(exist_ok=True)

    # 1. GitHub 가 그린 README (이 커밋)
    body = sh("gh", "api", f"repos/{REPO}/readme?ref={sha}", "-H", "Accept: application/vnd.github.html")

    def fix_href(m: re.Match) -> str:
        href = html.unescape(m.group(1))
        if href.startswith(("http://", "https://", "#", "mailto:")):
            return m.group(0)
        return f'href="https://github.com/{REPO}/blob/{sha}/{href}"'

    body = re.sub(r'href="([^"]*)"', fix_href, body)
    css_path = OUT / "github-markdown-light.css"
    if not css_path.exists():
        css_path.write_bytes(urllib.request.urlopen(CSS_URL, timeout=30).read())
    page = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<base href="{ROOT.as_uri()}/">
<title>포트폴리오 {sid} {name}</title>
<style>{css_path.read_text()}</style>
<style>
@page {{ size: A4; margin: 14mm 12mm 14mm 12mm; }}
body {{ margin: 0; }}
.markdown-body {{ font-size: 10.5pt; max-width: none; padding: 0; }}
.markdown-body table {{ display: table; width: 100%; table-layout: auto; font-size: 8.5pt; }}
.markdown-body table th, .markdown-body table td {{ padding: 3px 5px; word-break: keep-all; overflow-wrap: anywhere; }}
.markdown-body pre, .markdown-body pre code {{ white-space: pre-wrap; word-break: break-all; font-size: 8.5pt; }}
.markdown-body img {{ max-width: 100%; page-break-inside: avoid; }}
.markdown-body h2 {{ page-break-before: auto; }}
.markdown-body tr, .markdown-body img {{ break-inside: avoid; }}
.markdown-body a {{ color: #0969da; }}
</style></head><body><article class="markdown-body">{body}</article></body></html>"""
    html_path = OUT / "readme_print.html"
    html_path.write_text(page)
    pdf = OUT / f"포트폴리오_{sid}_{name}.pdf"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--allow-file-access-from-files",
                    "--virtual-time-budget=20000", f"--print-to-pdf={pdf}", html_path.as_uri()],
                   check=True, capture_output=True, text=True)

    # 2. 소스코드 (같은 커밋) + PDF + 안내
    stage = OUT / f"기말-{sid}-{name}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    tar = OUT / "src.tar"
    sh("git", "archive", "--format=tar", "-o", str(tar), sha)
    subprocess.run(["tar", "-xf", str(tar), "-C", str(stage)], check=True)
    tar.unlink()
    for ex in EXCLUDE:
        shutil.rmtree(stage / ex, ignore_errors=True)
    shutil.copy(pdf, stage / pdf.name)
    (stage / "제출_안내.txt").write_text(
        f"기말 프로젝트 제출물 — {sid} {name}\n\n"
        f"- 포트폴리오: {pdf.name} (GitHub README 와 같은 내용, 커밋 {sha[:7]})\n"
        f"- 저장소: {meta['github_url']} (커밋 {sha})\n- 데모: {meta['demo_url']}\n"
        "- 소스코드: 같은 커밋의 저장소 파일 전부 (src/, scripts/, tests/, web/, models/ 학습한 가중치, reports/ 학습 기록·평가 결과)\n"
        "- 뺀 것(용량, 공개 저장소에 있음): web/public/tts (미리 합성한 음성 조각 58MB, 없으면 앱은 브라우저 음성으로 말한다),\n"
        "  reports/tts (음성 엔진 비교 녹음 43MB). 촬영 영상·배경 조각·실제 프레임 정답은 Release data-v1 에 있고\n"
        "  scripts/download_data.sh 가 받아 SHA-256 으로 확인한다 (README 8절).\n"
        "- 실행 방법: README 8절 (파이썬 requirements.txt, 웹 web/package.json)\n")
    zpath = OUT / f"기말-{sid}-{name}.zip"
    if zpath.exists():
        zpath.unlink()
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in sorted(stage.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(OUT).as_posix())
    n_files = sum(1 for f in stage.rglob("*") if f.is_file())
    print(f"커밋 {sha[:7]}\nPDF  {pdf} ({pdf.stat().st_size / 1e6:.1f}MB)\nZIP  {zpath} ({zpath.stat().st_size / 1e6:.1f}MB, 파일 {n_files}개)")


if __name__ == "__main__":
    main()
