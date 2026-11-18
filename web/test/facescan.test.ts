import { test } from "node:test";
import assert from "node:assert/strict";
import fx from "./fixtures/facescan.json" with { type: "json" };
import { FaceScanner, NEED, readFace } from "../src/vision/facescan.ts";
import { SOLVED, applySlotAlg } from "../src/coach/cube.ts";

const f32 = (b64: string) => {
  const b = Buffer.from(b64, "base64");
  return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
};

test("면 읽기가 파이썬과 같다 (실제 영상 프레임 7장 + 떨어진 덩어리를 붙인 1장)", () => {
  assert.equal(fx.cases.length, 8);
  for (const c of fx.cases) {
    const got = readFace({ n: 48, logp: f32(c.logp), heat: f32(c.heat) });
    assert.deepEqual(got, c.face, `${c.t}초${"blob" in c ? " + 덩어리" : ""}`);
  }
});

test("큐브와 떨어진 곳이 스티커로 읽혀도(뒤쪽 책장 같은 것) 붙이기 전과 같은 면을 읽는다", () => {
  const plain = fx.cases.find((c) => c.t === 7.7 && !("blob" in c))!;
  const blob = fx.cases.find((c) => "blob" in c)!;
  const a = readFace({ n: 48, logp: f32(plain.logp), heat: f32(plain.heat) });
  const b = readFace({ n: 48, logp: f32(blob.logp), heat: f32(blob.heat) });
  assert.ok(a);
  assert.deepEqual(b, a);
});

test("실제 영상(latest2.mov) 처음 33초: 파이썬 읽기 순서를 그대로 넣으면 맞추기 시도마다 파이썬과 결과가 같다", () => {
  const sc = new FaceScanner();
  const { reads, attempts, solved } = fx.replay;
  let i = 0;
  for (const [t, ok] of attempts as [number, boolean][]) {
    while (i < reads.length && (reads[i][0] as number) <= t) sc.add(reads[i++][1] as number[]);
    assert.ok(sc.ready(), `${t}초`);
    const r = sc.solve();
    assert.equal(!!r, ok, `${t}초`);
    if (r) {
      assert.equal(r.mirrored, false);
      assert.deepEqual(Array.from(r.state), solved.state);
    }
  }
});

test("가운데 색만 같은 틀린 읽기는 '다 읽음'으로 치지 않는다 — 같은 격자가 NEED번 나와야 한다", () => {
  const sc = new FaceScanner();
  const seen = Object.values(fx.seen).map((g) => [...(g as number[])]);
  seen.slice(1).forEach((g) => { for (let n = 0; n < NEED; n++) sc.add(g); });
  // 빨 면: 가운데만 맞고 나머지가 매번 다른 읽기 (두 면에 걸친 격자 같은 것) 5번
  for (let k = 0; k < 5; k++) sc.add(seen[0].map((c, i) => (i === 4 ? c : (c + k + i) % 6)));
  assert.ok(sc.frames[1] >= NEED);
  assert.equal(sc.ready(), false);
  assert.equal(sc.weakest(), 1);
  for (let n = 0; n < NEED; n++) sc.add(seen[0]);
  assert.ok(sc.ready());
  assert.deepEqual(Array.from(sc.solve()!.state), fx.truth);
});

test("더 자주 나온 틀린 격자가 있어도, NEED번 이상 나온 맞는 격자로 맞춘다 (흰 센터 로고를 파랑으로 읽는 경우)", () => {
  const sc = new FaceScanner();
  const seen = fx.seen as Record<string, number[]>;
  Object.values(seen).forEach((g) => { for (let n = 0; n < NEED; n++) sc.add(g); });
  const logo = [...seen.W];
  logo[4] = 5; // 흰 면인데 가운데 로고 때문에 파랑으로 읽힘
  for (let n = 0; n < 10; n++) sc.add(logo);
  assert.deepEqual(Array.from(sc.solve()!.state), fx.truth);
});

test("여섯 면(보이는 그대로)으로 처음 상태를 하나로 맞춘다 — 사람이 읽은 정답과 같다", () => {
  const sc = new FaceScanner();
  // 일부러 면마다 다른 방향으로 돌려서 넣는다 (사람이 어느 방향으로 보여 줘도 되는지)
  const ROT = [6, 3, 0, 7, 4, 1, 8, 5, 2];
  Object.values(fx.seen).forEach((g, k) => {
    let face = g as number[];
    for (let r = 0; r < k % 4; r++) face = ROT.map((i) => face[i]);
    for (let n = 0; n < 3; n++) sc.add(face);
  });
  assert.ok(sc.ready());
  const r = sc.solve()!;
  assert.equal(r.mirrored, false);
  assert.deepEqual(Array.from(r.state), fx.truth);
});

test("한 칸을 잘못 읽어도 둘째 후보로 바로잡는다", () => {
  const sc = new FaceScanner();
  const seen = Object.values(fx.seen).map((g) => [...(g as number[])]);
  seen.forEach((g, k) => {
    for (let n = 0; n < 3; n++) sc.add(g);
    if (k === 0) {
      // 빨 면의 0번 칸을 한 번 주황으로 잘못 읽고, 정답은 두 번
      const bad = [...g];
      bad[0] = 4;
      sc.add(bad); sc.add(bad); sc.add(bad); sc.add(bad);
    }
  });
  assert.deepEqual(Array.from(sc.solve()!.state), fx.truth);
});

test("좌우반전된 카메라: 거울상으로 읽은 여섯 면에서도 같은 상태를 찾고, 반전을 알아챈다", () => {
  const MIRROR = [2, 1, 0, 5, 4, 3, 8, 7, 6];
  const ROT = [6, 3, 0, 7, 4, 1, 8, 5, 2];
  const sc = new FaceScanner();
  Object.values(fx.seen).forEach((g, k) => {
    let face = MIRROR.map((i) => (g as number[])[i]); // 반전된 화면에서 보이는 모습
    for (let r = 0; r < k % 4; r++) face = ROT.map((i) => face[i]);
    for (let n = 0; n < 3; n++) sc.add(face);
  });
  const r = sc.solve()!;
  assert.equal(r.mirrored, true);
  assert.deepEqual(Array.from(r.state), fx.truth);
});

test("거울상 큐브는 실제로 존재할 수 없다 (반전 판단의 근거)", async () => {
  const { validState } = await import("../src/vision/scan.ts");
  // 정답 상태의 각 면을 거울상으로 뒤집으면, 어떤 방향 조합으로도 유효하지 않아야 한다
  const MIRROR = [2, 1, 0, 5, 4, 3, 8, 7, 6];
  const ROT = [6, 3, 0, 7, 4, 1, 8, 5, 2];
  const faces = [0, 1, 2, 3, 4, 5].map((f) => MIRROR.map((i) => fx.truth[f * 9 + i]));
  const rots = faces.map((g) => { const r = [g]; for (let k = 0; k < 3; k++) r.push(ROT.map((i) => r[r.length - 1][i])); return r; });
  let valid = 0;
  const s = new Uint8Array(54);
  for (let code = 0; code < 4096; code++) {
    for (let f = 0; f < 6; f++) rots[f][(code >> (2 * f)) & 3].forEach((c, i) => (s[f * 9 + i] = c));
    if (validState(s)) valid++;
  }
  assert.equal(valid, 0);
});

test("다 맞춘 큐브도 읽는다 — 돌려도 같은 면 때문에 여러 방향 조합이 같은 상태를 만들어도 하나로 센다", () => {
  for (const alg of ["", "R2 L2 U2 D2 F2 B2", "R U R' U'"]) {
    const st = alg ? applySlotAlg(SOLVED, alg) : SOLVED;
    const sc = new FaceScanner();
    for (let slot = 0; slot < 6; slot++) for (let n = 0; n < NEED; n++) sc.add(Array.from(st.slice(slot * 9, slot * 9 + 9)));
    const r = sc.solve();
    assert.ok(r, alg || "다 맞춘 큐브");
    assert.deepEqual(Array.from(r!.state), Array.from(st), alg);
    assert.equal(sc.ambiguous, false);
  }
});

test("면만으로는 가릴 수 없는 상태(R 한 번 = L' 한 번의 면을 돌린 모양)는 억지로 맞추지 않고 ambiguous 로 알린다", () => {
  for (const alg of ["R", "R U"]) {
    const st = applySlotAlg(SOLVED, alg);
    const sc = new FaceScanner();
    for (let slot = 0; slot < 6; slot++) for (let n = 0; n < NEED; n++) sc.add(Array.from(st.slice(slot * 9, slot * 9 + 9)));
    assert.ok(sc.ready());
    assert.equal(sc.solve(), null, alg);
    assert.equal(sc.ambiguous, true, alg);
  }
});
