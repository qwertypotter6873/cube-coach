/**
 * 놓친 회전을 앞에 끼워 넣을 때 코치·연습이 추적기 상태를 따라가는가 (coach/resync.ts, Coach.resync, Drill.resync).
 *
 * 추적기는 놓쳤던 회전을 뒤늦게 찾으면 back 수 앞에 끼워 넣고 상태를 고친 순서로 다시 계산한다 (tracker.ts decide).
 * 예전 main.ts 는 받은 회전을 지금 상태 위에 얹기만 해서, 회전 순서가 바뀐 만큼 코치가 아는 큐브가 추적기와 달라졌다.
 * 여기서는 main.ts applyMove 와 같은 길(followMove → onMove 또는 resync)로 이벤트를 흘려 본다.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { WHOLE_TURNS, alignCenters, followMove, sameState } from "../src/coach/resync.ts";
import { Coach } from "../src/coach/coach.ts";
import type { Level, Utterance } from "../src/coach/coach.ts";
import { Drill } from "../src/coach/drill.ts";
import { SOLVED, applyColorMove, applyColorMoves, applySlotAlg, inverse, isSolved, mulberry32, opposite, translate } from "../src/coach/cube.ts";
import type { ColorMove, State } from "../src/coach/cube.ts";
import { PLL, YELLOW, analyze, planNext, searchAlgsPublic } from "../src/coach/method.ts";
import { planClips } from "../src/coach/voice.ts";
import { CAND_EVENTS, CAND_PERMS } from "../src/vision/tracker.ts";
import { scrambled } from "./student.ts";

/** 추적기 흉내: decide 처럼 회전을 back 수 앞에 끼워 넣고, 상태는 처음 상태에 고친 순서대로 다시 돌려 얻는다 */
class TrackerLog {
  events: ColorMove[] = [];
  readonly init: State;
  constructor(init: State) {
    this.init = init;
  }
  add(m: ColorMove, back = 0): State {
    this.events.splice(this.events.length - back, 0, m);
    return this.state();
  }
  state(): State {
    return applyColorMoves(this.init, this.events);
  }
}

const diff = (a: State, b: State) => a.reduce((n, x, i) => n + (x !== b[i] ? 1 : 0), 0);
const bad = (us: Utterance[]) => us.filter((u) => !u.text || /undefined|NaN|\[object|\$\{/.test(u.text));
/** X와 순서를 바꾸면 결과가 달라지는 회전 (같은 면도, 마주 보는 면도 아닌 면) */
const nonCommuting = (x: ColorMove): ColorMove => ({ color: [0, 1, 2, 3, 4, 5].find((c) => c !== x.color && c !== opposite(x.color))!, k: 1 });

const permute = (s: State, p: Int16Array) => Uint8Array.from({ length: 54 }, (_, i) => s[p[i]]);

test("추적기의 회전 순열 36가지는 코치의 applyColorMove 와 큐브를 통째로 돌린 만큼만 다르다", () => {
  for (let seed = 1; seed <= 5; seed++) {
    const s = scrambled(seed);
    CAND_EVENTS.forEach(([face, k], j) => {
      const m = { color: s[face * 9 + 4], k: k as 1 | 2 | 3 };
      const tracked = permute(s, CAND_PERMS[j]);
      const coach = applyColorMove(s, m);
      // 짝수 번째: 층이 돈 회전 (배열까지 같다), 홀수 번째: 층을 잡고 몸통이 돈 회전 (가운데 칸 자리가 바뀐다)
      if (j % 2 === 0) assert.ok(sameState(tracked, coach), `면 ${face}, ${k}번`);
      assert.ok(sameState(alignCenters(tracked, coach), coach), `면 ${face}, ${k}번 (${j % 2 ? "잡고 돌림" : "층"})`);
      assert.deepEqual(followMove(s, m, tracked).resync, false, "방향만 다른 것은 다시 맞출 일이 아니다");
    });
  }
  // 통째 회전 24가지는 모두 다르고, 어느 방향으로 돌려 놓아도 되돌린다
  const s = scrambled(8);
  assert.equal(new Set(WHOLE_TURNS.map((r) => Array.from(r ? applySlotAlg(s, r) : s).join(""))).size, 24);
  for (const r of WHOLE_TURNS) assert.ok(sameState(alignCenters(r ? applySlotAlg(s, r) : s, s), s), r);
});

test("층을 잡고 몸통을 돌린 회전이 섞여 추적기 배열의 방향이 바뀌어도, 끼워 넣기 전에는 다시 맞추지 않는다", () => {
  const s0 = scrambled(4);
  let tr = s0, cube = s0;
  const rand = mulberry32(2);
  for (let i = 0; i < 40; i++) {
    const face = Math.floor(rand() * 6), k = 1 + Math.floor(rand() * 3), held = rand() < 0.5 ? 1 : 0;
    const m = { color: tr[face * 9 + 4], k: k as 1 | 2 | 3 }; // 추적기가 내는 이벤트: 그 자리 가운데 색
    tr = permute(tr, CAND_PERMS[(face * 3 + k - 1) * 2 + held]);
    const r = followMove(cube, m, tr);
    assert.equal(r.resync, false, `${i}번째`);
    cube = r.state;
  }
  assert.ok(sameState(cube, alignCenters(tr, cube)));
});

/**
 * 저장된 실제 영상 추적 결과 하나 (reports/track_real_latest2_strict_app_s0.json, 앱 설정, 정답 48수를 다 맞힌 run).
 * [가운데 색, 90도 단위, 확정 시각(초), back]. 추적기의 마지막 순서 그대로이고, 확정한 순서는 시각 순이다.
 * 57.63초에 확정한 파랑 한 번이 세 수 앞(54.2초 뒤)에 끼워졌다.
 */
const APP_INIT = Uint8Array.from("344402114502315235020025333421431051555441202034052311", Number);
const APP_EVENTS: [number, number, number, number][] = [[5,3,54.2,0],[5,1,57.63,3],[0,3,54.47,0],[1,3,55.4,0],[1,1,56.53,0],[0,3,58.67,0],[0,2,59.9,0],[1,1,65.5,0],[0,1,68.1,0],[1,3,70.06,0],[0,3,71.76,0],[1,1,74.0,0],[0,1,76.26,0],[1,3,77.63,0],[0,3,78.66,0],[1,1,79.63,0],[0,1,80.86,0],[1,3,82.2,0],[0,3,83.46,0],[1,1,84.43,0],[0,1,85.73,0],[1,3,86.7,0],[0,3,87.63,0],[1,1,88.56,0],[0,1,90.46,0],[1,3,91.5,0],[0,3,92.53,0],[1,1,93.66,0],[0,1,94.8,0],[1,3,95.9,0],[0,3,96.9,0],[4,3,107.13,0],[0,3,107.99,0],[4,1,108.76,0],[0,1,109.49,0],[4,3,110.33,0],[0,3,111.29,0],[4,1,113.36,0],[0,1,114.29,0],[4,3,115.19,0],[0,3,116.03,0],[4,1,117.19,0],[0,1,118.13,0],[4,3,119.06,0],[0,3,120.03,0],[4,1,121.16,0],[0,1,122.33,0],[4,3,123.29,0],[0,3,125.73,0],[4,1,126.79,0],[0,1,127.73,0],[4,3,128.46,0],[0,3,129.23,0],[4,1,130.03,0],[0,1,131.23,0]]; // prettier-ignore

test("실제 추적 결과: 위에 얹기만 하면 마지막 상태가 16칸 어긋나고, 추적기 상태를 따르면 매 수 같다", () => {
  const stream = [...APP_EVENTS].sort((a, b) => a[2] - b[2]); // 코치가 받는 순서
  const tr = new TrackerLog(APP_INIT);
  let old = APP_INIT, now = APP_INIT;
  let resyncs = 0;
  for (const [color, k, , back] of stream) {
    const m = { color, k: k as 1 | 2 | 3 };
    const st = tr.add(m, back);
    old = applyColorMove(old, m); // 예전 main.ts
    const r = followMove(now, m, st);
    now = r.state;
    if (r.resync) resyncs++;
    assert.ok(sameState(now, st), "추적기 상태와 같아야 한다");
  }
  // 복원한 순서가 저장된 추적기 순서와 같은지 (흉내가 decide 와 같은지)
  assert.deepEqual(tr.events.map((m) => [m.color, m.k]), APP_EVENTS.map((e) => [e[0], e[1]]));
  assert.equal(resyncs, 1, "끼워 넣은 한 번만 다시 맞춘다");
  assert.equal(diff(old, tr.state()), 16, "예전 방식은 16칸 어긋난다");
  assert.ok(sameState(now, tr.state()));
});

test("키보드(추적기 상태 없음)와 순서를 바꿔도 같은 회전은 예전처럼 위에 얹는다", () => {
  const s = scrambled(3);
  const m: ColorMove = { color: 2, k: 1 };
  assert.deepEqual(followMove(s, m), { state: applyColorMove(s, m), resync: false });
  // 마주 보는 면은 순서를 바꿔도 같다: back 1 이어도 다시 맞출 일이 없다
  const tr = new TrackerLog(s);
  const y: ColorMove = { color: opposite(2), k: 3 };
  const c1 = followMove(s, y, tr.add(y)).state;
  const r = followMove(c1, m, tr.add(m, 1));
  assert.equal(r.resync, false);
  assert.ok(sameState(r.state, tr.state()));
});

/** 이 파일의 시험에서 코치·연습이 한 말: 다시 맞출 때(resync) / 그 밖(start·onMove). 끝에서 클립으로 읽히는지 본다 */
const RESYNC_TEXTS = new Set<string>();
const OTHER_TEXTS = new Set<string>();

/** main.ts applyMove 와 같은 길: 추적기 상태와 위에 얹은 결과가 다르면 resync, 아니면 onMove */
function camera(init: State, target: { onMove(m: ColorMove, t: number): Utterance[]; resync(s: State, m: ColorMove, t: number): Utterance[] }) {
  const tr = new TrackerLog(init);
  let cube = init, t = 0;
  const said: Utterance[] = [];
  const feed = (m: ColorMove, back = 0) => {
    const r = followMove(cube, m, tr.add(m, back));
    cube = r.state;
    t += 700;
    const us = r.resync ? target.resync(cube, m, t) : target.onMove(m, t);
    said.push(...us);
    us.forEach((u) => (r.resync ? RESYNC_TEXTS : OTHER_TEXTS).add(u.text));
    return { us, resync: r.resync };
  };
  // start() 처럼 바깥에서 넣는 말은 '그 밖'으로 센다
  const push = said.push.bind(said);
  said.push = (...us: Utterance[]) => (us.forEach((u) => OTHER_TEXTS.add(u.text)), push(...us));
  return { tr, feed, said, cube: () => cube };
}

for (const level of ["guided", "hint"] as Level[]) {
  test(`${level}: 시킨 회전을 놓쳤다가 다음 수 뒤에 찾으면, 코치가 추적기 상태로 다시 계획하고 그대로 따라 하면 다 맞춰진다`, () => {
    for (const seed of [7, 11, 19]) {
      const s0 = scrambled(seed);
      const coach = new Coach(s0, { method: "beginner", level, taughtBasics: true });
      const cam = camera(s0, coach);
      cam.said.push(...coach.start(0));
      for (let i = 0; i < 4; i++) cam.feed(coach.expected()!);
      // 사람은 시킨 회전 X를 돌렸지만 추적기가 놓쳤다. 그다음 다른 면 Y를 돌렸다 (코치는 엉뚱한 면으로 본다).
      const X = coach.expected()!;
      const Y = nonCommuting(X);
      cam.feed(Y);
      const before = cam.cube();
      // 추적기가 X를 찾아 Y 앞에 끼워 넣는다
      const { us, resync } = cam.feed(X, 1);
      assert.ok(resync, "다시 맞춰야 한다");
      assert.ok(!sameState(applyColorMove(before, X), cam.tr.state()), "예전처럼 위에 얹으면 어긋난다");
      assert.ok(sameState(coach.s, cam.tr.state()), "코치 상태 = 추적기 상태");
      assert.ok(sameState(cam.cube(), cam.tr.state()), "화면 상태 = 추적기 상태");
      assert.equal(coach.fixing, false, "틀린 상태로 시킨 되돌리기는 버린다");
      assert.deepEqual(bad(us), []);
      if (level === "guided") assert.ok(us.some((u) => u.kind === "move" || u.kind === "hold"), "새 지시를 바로 한다");
      assert.equal(coach.moves.length, 6, "끼워 넣은 수도 한 번만 기록한다");
      for (let i = 0; i < 3000 && !isSolved(coach.s); i++) cam.feed(coach.expected()!);
      assert.ok(isSolved(coach.s) && isSolved(cam.tr.state()), `시드 ${seed}: 다시 계획한 대로 따라 하면 다 맞춰져야 한다`);
      assert.equal(cam.said.at(-1)!.kind, "summary");
      assert.deepEqual(bad(cam.said), []);
    }
  });
}

test("observe: 끼워 넣은 회전으로 다 맞춰지면 완성 정리를 하고, 시간 재기는 이어진다", () => {
  const X: ColorMove = { color: 1, k: 1 };
  const Y = nonCommuting(X);
  const c0 = applyColorMoves(SOLVED, [inverse(Y), inverse(X)]); // X, Y 순서로 돌리면 맞춰지는 큐브
  const coach = new Coach(c0, { method: "beginner", level: "observe", taughtBasics: true });
  coach.start(0);
  const cam = camera(c0, coach);
  cam.feed(Y); // X는 놓쳤다
  assert.ok(!isSolved(applyColorMove(cam.cube(), X)), "예전처럼 위에 얹으면 맞춰진 줄 모른다");
  const { us } = cam.feed(X, 1);
  assert.ok(isSolved(coach.s));
  assert.equal(coach.currentStage, "done");
  assert.ok(us.some((u) => u.kind === "summary"), "완성 정리");
  assert.equal(coach.moves.length, 2);
  assert.equal(coach.moves[0].t, 700, "첫 수 시각(시간 재기 시작)은 그대로");
  // 끝난 풀이: 또 다시 맞춰도 상태만 따라가고 기록·말은 없다
  const more = cam.feed({ color: 0, k: 1 });
  const again = cam.feed({ color: 2, k: 1 }, 1);
  assert.deepEqual([...more.us, ...again.us], []);
  assert.equal(coach.moves.length, 2);
  assert.ok(sameState(coach.s, cam.tr.state()));
});

/** 연습하는 학생 (test/drill.test.ts 와 같다): 정답 공식을 찾아 실행 */
function solveCase(s: State, cases: typeof PLL): ColorMove[] {
  const f = searchAlgsPublic(s, cases, (x: State) => analyze(x).auf >= 0)!;
  const pre = f.auf ? [{ color: YELLOW, k: f.auf as 1 | 2 | 3 }] : [];
  return [...pre, ...translate(applyColorMoves(s, pre), f.alg.alg, f.view).map((m) => ({ color: m.color, k: m.k }))];
}

test("공식 연습 준비 중: 놓친 준비 동작을 앞에 끼워 넣으면 지금 상태에서 다시 맞추고 다음 연습으로 이어 간다", () => {
  const cases = PLL.slice(0, 4);
  const d = new Drill(SOLVED, cases, "pll", mulberry32(5));
  const cam = camera(SOLVED, d);
  cam.said.push(...d.start(0));
  cam.feed(d.expected()!);
  const X = d.expected()!; // 불러 준 대로 돌렸지만 추적기가 놓쳤다
  cam.feed(nonCommuting(X)); // 다른 면을 돌렸다 (연습은 "되돌려 주세요")
  assert.ok(d.fixing);
  const { us, resync } = cam.feed(X, 1);
  assert.ok(resync);
  assert.ok(sameState(d.s, cam.tr.state()));
  assert.equal(d.fixing, false);
  assert.ok(d.expected(), "다음에 할 동작을 불러 준다");
  assert.deepEqual(bad(us), []);
  for (let i = 0; i < 200 && d.phase !== "solve"; i++) cam.feed(d.expected()!);
  assert.equal(d.phase, "solve", "다시 맞춘 뒤 준비가 끝나야 한다");
  assert.ok(sameState(d.s, cam.tr.state()));
  assert.deepEqual(bad(cam.said), []);
});

test("공식 연습 푸는 중: 놓친 한 수를 앞에 끼워 넣어도 다 푼 것을 알아본다", () => {
  const cases = PLL.slice(0, 4);
  const d = new Drill(SOLVED, cases, "pll", mulberry32(9));
  const cam = camera(SOLVED, d);
  d.start(0);
  for (let i = 0; i < 60 && d.phase === "setup"; i++) cam.feed(d.expected()!);
  assert.equal(d.phase, "solve");
  const sol = solveCase(d.s, cases);
  let i = 0;
  while (sol[i + 1] && (sol[i + 1].color === sol[i].color || sol[i + 1].color === opposite(sol[i].color))) i++; // 순서를 바꾸면 달라지는 두 수
  assert.ok(sol[i + 1], "순서를 바꾸면 달라지는 이웃한 두 수가 있어야 한다");
  for (let j = 0; j < i; j++) cam.feed(sol[j]);
  cam.feed(sol[i + 1]); // sol[i] 는 놓쳤다
  cam.feed(sol[i], 1);
  assert.ok(sameState(d.s, cam.tr.state()));
  for (let j = i + 2; j < sol.length; j++) cam.feed(sol[j]);
  assert.equal(d.results.length, 1, "다 푼 것을 알아봐야 한다");
  assert.ok(cam.said.some((u) => u.kind === "summary"));
});

test("무작위로 놓쳤다가 1~3수 뒤에 찾아도, 코치는 늘 추적기 상태를 따라가고 결국 다 맞춰진다 (방법·수준별)", () => {
  for (const method of ["beginner", "cfop4", "cfop"] as const)
    for (const level of ["guided", "hint", "observe"] as Level[])
      for (let seed = 1; seed <= 3; seed++) {
        const rand = mulberry32(seed * 31 + method.length * 7 + level.length);
        const s0 = scrambled(seed + 200);
        const coach = new Coach(s0, { method, level, taughtBasics: true });
        const cam = camera(s0, coach);
        cam.said.push(...coach.start(0));
        const done: ColorMove[] = []; // 사람이 실제로 돌린 순서
        let missed: { m: ColorMove; wait: number; seen: number } | null = null;
        let inserts = 0;
        // 코치를 따라 하되, 놓친 회전이 있는 동안(코치가 모르는 동안)과 혼자 풀기에서는 실제 큐브로 세운 계획을 끝까지 따라 한다
        let queue: ColorMove[] = [];
        const fromPlan = (truth: State): ColorMove => {
          if (!queue.length) queue = (planNext(truth, method)?.moves ?? []).map((x) => ({ color: x.color, k: x.k }));
          return queue.shift() ?? { color: 0, k: 1 };
        };
        for (let i = 0; i < 4000; i++) {
          const truth = applyColorMoves(s0, done);
          if (isSolved(truth) && !missed) break;
          let m: ColorMove;
          if (missed || level === "observe") m = fromPlan(truth);
          else {
            queue = [];
            m = coach.expected() ?? fromPlan(truth);
          }
          done.push(m);
          if (!missed && i > 3 && rand() < 0.08) {
            missed = { m, wait: 1 + Math.floor(rand() * 3), seen: 0 }; // 추적기가 놓쳤다
            continue;
          }
          cam.feed(m);
          if (missed && ++missed.seen === missed.wait) {
            cam.feed(missed.m, missed.seen); // 뒤늦게 찾아 그만큼 앞에 끼워 넣는다
            missed = null;
            inserts++;
          }
          assert.ok(sameState(coach.s, cam.tr.state()), `${method}/${level}/${seed}: 코치 상태 = 추적기 상태`);
        }
        const tag = `${method}/${level}/${seed}`;
        assert.ok(inserts > 0, `${tag}: 끼워 넣기가 한 번은 있어야 시험이 된다`);
        assert.ok(sameState(cam.tr.state(), applyColorMoves(s0, done)), `${tag}: 흉내 낸 추적기 = 실제 큐브`);
        assert.ok(isSolved(coach.s), `${tag}: 다 맞춰져야 한다`);
        assert.ok(cam.said.some((u) => u.kind === "summary"), `${tag}: 완성 정리`);
        assert.deepEqual(bad(cam.said), [], tag);
      }
});

test("다시 맞출 때와 혼자 풀기에서 하는 말은 모두 미리 만든 클립으로 읽힌다", () => {
  assert.ok(RESYNC_TEXTS.size > 20, `다시 맞출 때 한 말이 너무 적다 (${RESYNC_TEXTS.size})`);
  for (const voice of ["F2", "M1"]) {
    const path = new URL(`../public/tts/${voice}/manifest.json`, import.meta.url);
    if (!existsSync(path)) continue;
    const manifest = JSON.parse(readFileSync(path, "utf8")) as Record<string, string>;
    const unread = [...RESYNC_TEXTS].filter((x) => !planClips(x, (p) => p in manifest));
    const unreadAll = [...new Set([...RESYNC_TEXTS, ...OTHER_TEXTS])].filter((x) => !planClips(x, (p) => p in manifest));
    assert.deepEqual(unread.filter((x) => !OTHER_TEXTS.has(x)), [], `${voice}: 다시 맞출 때만 나오는, 클립으로 못 읽는 대사`);
    // 예전 길에서도 나오는 말까지 모두 클립으로 읽혀야 한다 (혼자 풀기의 "흰색 십자 완성."이 클립 없이 조용하던 빈틈을 고친 뒤 막아 둔다)
    assert.deepEqual(unread, [], `${voice}: 클립으로 못 읽는 대사`);
    assert.ok(OTHER_TEXTS.size > 20, `예전 길(onMove)에서 한 말이 너무 적다 (${OTHER_TEXTS.size})`);
    assert.deepEqual(unreadAll, [], `${voice}: 이 시험의 풀이(혼자 풀기 포함)에서 클립으로 못 읽는 대사`);
  }
});
