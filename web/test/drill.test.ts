import { test } from "node:test";
import assert from "node:assert/strict";
import { Drill } from "../src/coach/drill.ts";
import { SOLVED, applyColorMoves, translate, mulberry32 } from "../src/coach/cube.ts";
import type { State } from "../src/coach/cube.ts";
import { PLL, OCLL, YELLOW, analyze, searchAlgsPublic } from "../src/coach/method.ts";
import { scrambled } from "./student.ts";

/** 연습하는 학생: 준비 동작은 불러 주는 대로, 풀 때는 정답 공식(보는 방향·윗면 돌리기 포함)을 찾아 실행 */
function solveCase(s: State, cases: typeof PLL, kind: "oll" | "pll") {
  const goal = (t: State) => (kind === "oll" ? analyze(t).oriented : analyze(t).auf >= 0);
  const f = searchAlgsPublic(s, cases, goal);
  assert.ok(f, "풀 수 있는 공식이 있어야 함");
  const pre = f!.auf ? [{ color: YELLOW, k: f!.auf as 1 | 2 | 3 }] : [];
  const s1 = applyColorMoves(s, pre);
  return [...pre, ...translate(s1, f!.alg.alg, f!.view).map((m) => ({ color: m.color, k: m.k }))];
}

for (const [kind, cases] of [["pll", PLL.slice(0, 6)], ["oll", OCLL]] as const) {
  test(`${kind} 연습: 준비 → 풀기 → 결과를 반복한다`, () => {
    const d = new Drill(SOLVED, cases, kind, mulberry32(3));
    let t = 0;
    const said = [...d.start(t)];
    for (let round = 0; round < cases.length; round++) {
      let guard = 0;
      while (d.phase === "setup" && guard++ < 60) said.push(...d.onMove(d.expected()!, (t += 300)));
      assert.equal(d.phase, "solve");
      t += 1200; // 알아보는 시간
      for (const m of solveCase(d.s, cases, kind)) said.push(...d.onMove(m, (t += 250)));
    }
    assert.equal(d.results.length, cases.length);
    assert.ok(d.results.every((r) => r.reactMs >= 1200 && r.clean));
    assert.ok(said.filter((u) => u.kind === "summary").length >= cases.length);
    assert.ok(!said.some((u) => /undefined|NaN/.test(u.text)));
  });
}

test("섞인 큐브로 시작하면 먼저 1·2층까지 맞춰 준 뒤 연습한다", () => {
  const d = new Drill(scrambled(4), PLL.slice(0, 3), "pll", mulberry32(1));
  let t = 0;
  d.start(t);
  let guard = 0;
  while (d.phase === "restore" && guard++ < 400) d.onMove(d.expected()!, (t += 300));
  assert.equal(d.phase, "setup");
});

test("알아보기 연습: 풀기 전에는 공식 이름을 말하지 않고, 다 푼 뒤 결과와 함께 알려 준다", () => {
  for (const [kind, cases] of [["oll", OCLL], ["pll", PLL.slice(6, 12)]] as const) {
    const d = new Drill(SOLVED, cases, kind, mulberry32(7));
    let t = 0;
    let before = [...d.start(t)];
    for (let round = 0; round < cases.length; round++) {
      const name = d.current!.name;
      while (d.phase === "setup") before.push(...d.onMove(d.expected()!, (t += 300)));
      assert.ok(!before.some((u) => u.text.includes(name)), `${name}: 풀기 전에 이름이 나왔다 — ${before.map((u) => u.text).join(" / ")}`);
      const after = [];
      for (const m of solveCase(d.s, cases, kind)) after.push(...d.onMove(m, (t += 250)));
      const summary = after.find((u) => u.kind === "summary");
      assert.ok(summary?.text.startsWith(`정답은 ${name}`), `다 푼 뒤 이름을 알려 준다: ${summary?.text}`);
      before = after.slice(after.indexOf(summary!) + 1); // 다음 연습 안내부터는 다시 '풀기 전'
    }
  }
});

test("공식 연습: 엉켜서 그만둘 때도 정답 이름을 알려 준다", () => {
  const d = new Drill(SOLVED, PLL.slice(0, 2), "pll", mulberry32(2));
  let t = 0;
  d.start(t);
  while (d.phase === "setup") d.onMove(d.expected()!, (t += 300));
  const name = d.current!.name;
  let out: { text: string }[] = [];
  for (let i = 0; i < 60 && d.phase === "solve"; i++) out = d.onMove({ color: i % 2 ? 0 : 1, k: 1 }, (t += 200)); // 아무렇게나 돌린다
  assert.notEqual(d.phase, "solve");
  assert.ok(out.some((u) => u.text.includes(`정답은 ${name}`)), out.map((u) => u.text).join(" / "));
});

test("공식 연습: 바로잡는 중이면 fixing", () => {
  const d = new Drill(SOLVED, PLL.slice(0, 2), "pll", mulberry32(4));
  d.start(0);
  const exp = d.expected()!;
  assert.equal(d.fixing, false);
  const other = [0, 1, 2, 3, 4, 5].find((c) => c !== exp.color)!;
  d.onMove({ color: other, k: 1 }, 100);
  assert.equal(d.fixing, true, "엉뚱한 면을 돌려 되돌리는 중");
  d.onMove({ color: other, k: 3 }, 200);
  assert.equal(d.fixing, false);
});
