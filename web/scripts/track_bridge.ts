/**
 * 파이썬 평가 스크립트(scripts/eval_tracking.py --ts)가 TS 추적기를 그대로 돌려 보기 위한 다리.
 * 표준입력 한 줄 = 요청 JSON, 표준출력 한 줄 = 응답 JSON.
 *   {"init": [54색], "f": 0.95, "seed": 0, "fast": false, "follow": ["predict", ...], "aspect": 0.5625}
 *   {"n": 48, "logp": base64(Float32 8*n*n), "heat": base64(Float32 n*n), "roi": [x, y, s]}
 */
import { createInterface } from "node:readline";
import { Tracker } from "../src/vision/tracker.ts";

let tr: Tracker | null = null;
let aspect = 1; // 화면 높이/너비 (놓쳤을 때 찾을 영역 계산용)
const f32 = (b64: string) => {
  const buf = Buffer.from(b64, "base64");
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
};
const reply = () => {
  const t = tr!;
  return {
    roi: t.roi(),
    lostRoi: t.lostRoi(aspect),
    moving: t.moving,
    events: t.events.map((e) => [e.center, e.quarterTurns, e.frame, e.info ?? null]),
    R: t.R ? Array.from(t.R) : null,
    T: t.Tv ? Array.from(t.Tv) : null,
    diag: t.diag,
    colors: Array.from(t.colors),
    quality: t.quality,
    layer: t.layer,
    phi: t.phi,
    coverage: t.coverage,
    view: t.view(),
    relocks: t.relocks,
    localRelocks: t.localRelocks,
    inits: t.inits,
    rejectedTurns: t.rejectedTurns,
    offRejected: t.offRejected,
    reverts: t.reverts,
  };
};
const rl = createInterface({ input: process.stdin });
rl.on("line", (line) => {
  const msg = JSON.parse(line);
  if (msg.init) {
    tr = new Tracker(Uint8Array.from(msg.init), msg.f ?? 0.95, msg.seed ?? 0, msg.bank ?? 2500);
    aspect = msg.aspect ?? 1;
    if (msg.cover !== undefined) tr.useCover = msg.cover;
    if (msg.settle !== undefined) tr.settleFrames = msg.settle;
    if (msg.fast !== undefined) tr.fastFollow = msg.fast;
    for (const k of msg.follow ?? []) (tr.follow as unknown as Record<string, boolean>)[k] = true;
    Object.assign(tr, msg.opts ?? {}); // 실험용 추적기 설정 (turnVsBody 등)
    process.stdout.write(JSON.stringify(reply()) + "\n");
    return;
  }
  const t0 = performance.now();
  tr!.update({ n: msg.n, logp: f32(msg.logp), heat: f32(msg.heat) }, msg.roi);
  process.stdout.write(JSON.stringify({ ...reply(), ms: performance.now() - t0 }) + "\n");
});
