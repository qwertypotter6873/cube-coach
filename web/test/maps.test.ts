import { test } from "node:test";
import assert from "node:assert/strict";
import fx from "./fixtures/maps_to.json" with { type: "json" };
import { mapsTo } from "../src/vision/maps.ts";

const f32 = (b64: string) => {
  const b = Buffer.from(b64, "base64");
  return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
};

test("지도 줄이기(64x64 → 48x48, 넓이 평균)가 파이썬 maps_to 와 같다", () => {
  for (const c of fx.cases) {
    const r = mapsTo(f32(c.logp), f32(c.heat), c.n, 48);
    const el = f32(c.out_logp), eh = f32(c.out_heat);
    assert.equal(r.logp.length, el.length);
    let dl = 0, dh = 0;
    for (let i = 0; i < el.length; i++) dl = Math.max(dl, Math.abs(r.logp[i] - el[i]));
    for (let i = 0; i < eh.length; i++) dh = Math.max(dh, Math.abs(r.heat[i] - eh[i]));
    assert.ok(dl < 1e-4 && dh < 1e-5, `n=${c.n}: 로그확률 최대 차이 ${dl}, 중심 지도 최대 차이 ${dh}`);
  }
});

test("48x48 지도는 그대로 둔다", () => {
  const c = fx.cases.find((x) => x.n === 48)!;
  const lp = f32(c.logp), ht = f32(c.heat);
  const r = mapsTo(lp, ht, 48, 48);
  assert.equal(r.logp, lp);
  assert.equal(r.heat, ht);
});
