import { test } from "node:test";
import assert from "node:assert/strict";
import { compareStages, caseStats, feedbackText, stageDurations } from "../src/coach/progress.ts";
import type { SolveRecord } from "../src/coach/progress.ts";

const rec = (at: number, cross: number, f2l: number, oll: number, pll: number, jbExec = 2000): SolveRecord => ({
  at, method: "cfop4", level: "observe", totalMs: pll, moves: 60, mistakes: 0,
  splits: { cross, f2l, oll, pll },
  steps: [
    { stage: "pll", caseName: "Jb", alg: "R U R'", reactMs: 1500, execMs: jbExec, moves: 13, mistakes: 0 },
    { stage: "oll", caseName: "수네", alg: "R U R'", reactMs: 500, execMs: 1200, moves: 7, mistakes: 0 },
  ],
});

test("단계별 시간은 끝난 시각의 차이", () => {
  assert.deepEqual(stageDurations(rec(1, 5000, 30000, 38000, 45000)), { cross: 5000, f2l: 25000, oll: 8000, pll: 7000 });
});

test("최고 기록이면 칭찬하고, 아니면 평균과 비교한다", () => {
  const hist = [rec(1, 8000, 40000, 50000, 60000), rec(2, 9000, 41000, 51000, 61000)];
  const fast = rec(3, 5000, 37000, 47000, 57000);
  assert.match(feedbackText(compareStages(fast, hist))!, /최고 기록/);
  const slow = rec(4, 12000, 46000, 56000, 66000);
  assert.match(feedbackText(compareStages(slow, hist))!, /느렸어요/);
});

test("느리고 자주 틀리는 공식이 연습 후보 맨 앞에 온다", () => {
  const hist = [rec(1, 1, 2, 3, 4, 9000), rec(2, 1, 2, 3, 4, 8000)];
  const st = caseStats(hist);
  assert.equal(st[0].name, "PLL Jb");
});
