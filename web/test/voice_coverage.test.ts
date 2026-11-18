/**
 * 코치가 하는 말이 빠짐없이 미리 합성한 클립으로 해결되는지 본다.
 *
 * 말해지는 것은 voice.push 로 들어가는 Utterance 뿐이다 (화면 글자 camStatus 등은 말하지 않는다).
 * 클립 모드에서는 목록에 없는 문장을 브라우저 음성으로 읽지 않으므로, 하나라도 빠지면 그 대사는 소리가 안 난다.
 *
 * 대사가 바뀌면(코칭 담당이 문장을 고치면) 이 시험이 먼저 빨개진다. 그때는 아래 명령을 다시 돌린다.
 *   node test/inventory.ts 1 40 ../reports/tts_inventory_train.json
 *   ../.venv/bin/python ../scripts/tts_build_clips.py --voice F2   (그리고 --voice M1)
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { ClipEngine, FakeEngine, GAP, planClips, buildPieces, readAloud, ttsTextFor } from "../src/coach/voice.ts";
import type { ClipPlayer, SeqItem } from "../src/coach/voice.ts";
import { allSpoken, buildList, expandNumbers, mainLiterals, planStats, saySiteLiterals, simulate } from "./inventory.ts";

const BUILD_SEEDS = [1, 40]; // 합성 목록을 만들 때 쓴 시드
const TEST_SEEDS = [500, 12]; // 시험은 겹치지 않는 시드로 본다

/** 합성 목록 (시험용으로 다시 만든다) — 무거우니 한 번만 */
const spokenForBuild = allSpoken(BUILD_SEEDS[0], BUILD_SEEDS[1]);
const list = buildList(spokenForBuild);
const pieces = new Set(list.map((x) => x.text));
const has = (t: string) => pieces.has(t);
const spokenFresh = simulate(TEST_SEEDS[0], TEST_SEEDS[1]);

function report(name: string, missing: Map<string, string[]>): string {
  const lines = [...missing].slice(0, 12).map(([text, lost]) => `  ${JSON.stringify(text)}\n    빠진 조각: ${lost.map((x) => JSON.stringify(x)).join(", ")}`);
  return `${name}: 클립으로 못 읽는 대사 ${missing.size}가지\n${lines.join("\n")}\n` +
    `(더 있으면 줄임) 고치려면: cd web && node test/inventory.ts 1 40 ../reports/tts_inventory_train.json 뒤에 scripts/tts_build_clips.py 를 --voice F2 / --voice M1 로 다시 돌리세요.`;
}

/** 못 읽는 대사와, 그 안에서 빠진 조각 */
function check(texts: Iterable<string>): Map<string, string[]> {
  const bad = new Map<string, string[]>();
  for (const t of texts) {
    if (planClips(t, has)) continue;
    bad.set(t, buildPieces(t).filter((p) => !has(p)));
  }
  return bad;
}

test("가상 풀이·연습에서 나오는 대사가 모두 클립으로 해결된다 (다른 시드)", () => {
  assert.ok(spokenFresh.size > 300, `모은 대사가 너무 적습니다 (${spokenFresh.size}가지) — 수집기를 보세요.`);
  assert.equal(check(spokenFresh.keys()).size, 0, report("가상 풀이", check(spokenFresh.keys())));
});

test("main.ts 의 고정 대사와 기록 피드백이 모두 클립으로 해결된다", () => {
  const bad = check(spokenForBuild.keys());
  assert.equal(bad.size, 0, report("고정 대사·기록 피드백", bad));
  assert.ok(mainLiterals().length >= 2, "main.ts 고정 대사를 못 찾았습니다.");
});

test("합성 목록만으로 모든 대사가 풀리고, 쉼표로 잘게 쪼개 읽는 대사는 없다", () => {
  const st = planStats(spokenForBuild.keys(), has);
  assert.equal(st.none, 0);
  // 쉼표 나누기는 처음 보는 동작 묶음을 읽을 때만 쓰는 비상구다. 아는 대사는 문장째 들어 있어야 한다.
  assert.equal(st.comma, 0, `쉼표로 쪼개 읽는 대사 ${st.comma}가지`);
});

test("숫자만 바뀌는 문장은 본 적 없는 숫자까지 채운다", () => {
  const got = expandNumbers(["여섯 수예요.", "세 수예요.", "섹시무브 한 번으로 먼저 꺼낼게요.", "다시, 두 번째."]);
  assert.ok(got.includes("한 수예요.") && got.includes("열두 수예요."), "범위를 채운다");
  assert.ok(!got.some((x) => x.includes("꺼낼게요")), "숫자가 하나로만 나온 문장은 그대로 둔다");
  assert.ok(!got.some((x) => x.includes("번째")), "번째는 숫자 자리가 아니다");
  for (const t of ["한 수예요.", "열 수예요.", "섹시무브 두 번.", "흰색 십자 최고 기록이에요! 삼백십이 초."]) {
    assert.ok(planClips(t, has), `목록에 없음: ${t}`);
  }
});

test("합성기에 넣는 글: 영어 약자와 개수는 읽는 소리로, 이어지는 조각은 쉼표로 끝낸다", () => {
  assert.equal(readAloud("가장 오래 걸린 단계는 F2L이었어요."), "가장 오래 걸린 단계는 에프투엘이었어요.");
  assert.equal(readAloud("PLL Jb 공식이에요."), "피엘엘 제이비 공식이에요.");
  assert.equal(readAloud("지금은 노란 꼭짓점 2개."), "지금은 노란 꼭짓점 두 개.");
  assert.equal(readAloud("1층 꼭짓점 완성!"), "1층 꼭짓점 완성!");
  assert.equal(ttsTextFor("총 일 분 이십삼 초"), "총 일 분 이십삼 초,");
  assert.equal(ttsTextFor("삼 초 빨랐어요."), "삼 초 빨랐어요.");
  for (const x of list) assert.ok(!/[A-Za-z]/.test(x.tts ?? x.text), `영어가 남은 합성 글: ${x.tts ?? x.text}`);
});

test("coach.ts·drill.ts 의 say(\"...\") 대사가 모두 수집된다", () => {
  const never = saySiteLiterals().filter((t) => !spokenFresh.has(t) && !simulate(BUILD_SEEDS[0], 8).has(t));
  assert.deepEqual(never, [], `가상 풀이에서 한 번도 나오지 않은 대사가 있습니다 (수집기가 그 상황을 못 만듭니다): ${never.join(" / ")}`);
});

test("합성해 둔 목소리에 모든 대사의 조각 파일이 실제로 있다", () => {
  const texts = [...new Set([...spokenForBuild.keys(), ...spokenFresh.keys()])];
  for (const voice of ["F2", "M1"]) {
    const dir = new URL(`../public/tts/${voice}/`, import.meta.url);
    const mpath = new URL("manifest.json", dir);
    if (!existsSync(mpath)) {
      console.log(`  (${voice} 아직 합성 전 — 건너뜀)`);
      continue;
    }
    const manifest = JSON.parse(readFileSync(mpath, "utf8")) as Record<string, string>;
    const lack = texts.filter((t) => !planClips(t, (x) => x in manifest));
    assert.equal(
      lack.length,
      0,
      `${voice}: 아직 합성 안 된 대사 ${lack.length}가지 — 예: ${lack.slice(0, 3).map((x) => JSON.stringify(x)).join(", ")}\n` +
        `  cd web && node test/inventory.ts 1 40 ../reports/tts_inventory_train.json && ../.venv/bin/python ../scripts/tts_build_clips.py --voice ${voice}`,
    );
    const noFile = Object.values(manifest).filter((f) => !existsSync(new URL(f, dir)));
    assert.equal(noFile.length, 0, `${voice}: manifest 에 적혀 있지만 파일이 없는 클립 ${noFile.length}개 (${noFile[0]})`);
  }
});

// --- 조각 이어 붙이기 동작 --------------------------------------------------------

class FakePlayer implements ClipPlayer {
  played: string[] = [];
  stopped = 0;
  private done: (() => void) | null = null;
  play(url: string): Promise<void> {
    this.played.push(url);
    return new Promise((r) => {
      this.done = r;
      setTimeout(r, 5);
    });
  }
  stop(): void {
    this.stopped++;
    this.done?.();
    this.done = null;
  }
}

const manifestOf = (texts: string[]) => Object.fromEntries(texts.map((t, i) => [t, `${i}.m4a`]));

test("숫자가 든 정리 문장을 조각으로 이어 읽는다", async () => {
  const text = "완성! 총 일 분 이십삼 초, 여든일곱 수 걸렸어요. 가장 오래 걸린 단계는 F2L이었어요.";
  const parts = ["완성!", "총 일 분 이십삼 초", "여든일곱 수 걸렸어요.", "가장 오래 걸린 단계는 F2L이었어요."];
  const p = new FakePlayer();
  const e = new ClipEngine(manifestOf(parts), "tts/", new FakeEngine(), p);
  e.gapScale = 0;
  await e.speak(text);
  assert.deepEqual(p.played, ["tts/0.m4a", "tts/1.m4a", "tts/2.m4a", "tts/3.m4a"]);
  assert.equal(e.misses, 0);
});

test("쉼표로 늘어놓은 동작 목록도 조각으로 읽는다", async () => {
  const parts = ["알 오른쪽 면 위로", "유 윗면 왼쪽으로", "알 프라임 오른쪽 면 아래로."];
  const p = new FakePlayer();
  const e = new ClipEngine(manifestOf(parts), "tts/", new FakeEngine(), p);
  e.gapScale = 0;
  await e.speak(parts.join(", "));
  assert.equal(p.played.length, 3);
});

test("클립이 없으면 브라우저 음성을 섞지 않고 읽을 수 있는 문장만 읽는다", async () => {
  const web = new FakeEngine();
  const p = new FakePlayer();
  const e = new ClipEngine(manifestOf(["앞 문장이에요."]), "tts/", web, p);
  e.gapScale = 0;
  await e.speak("앞 문장이에요. 아직 안 만든 뒷 문장이에요.");
  assert.deepEqual(p.played, ["tts/0.m4a"], "있는 문장만 클립으로 읽는다");
  assert.deepEqual(web.spoken, [], "브라우저 음성은 쓰지 않는다");
  assert.deepEqual(e.unresolved, ["앞 문장이에요. 아직 안 만든 뒷 문장이에요."]);
});

/** 여러 조각을 한 번에 받는 가짜 (브라우저의 Web Audio 이어 붙이기 자리) */
class FakeSeqPlayer extends FakePlayer {
  seqs: SeqItem[][] = [];
  canSeq = true;
  playSeq(items: SeqItem[]): Promise<boolean> {
    if (!this.canSeq) return Promise.resolve(false);
    this.seqs.push(items);
    return new Promise((r) => setTimeout(() => r(true), 5));
  }
}

test("여러 조각은 정한 쉼과 함께 한 번에 넘겨 빈틈없이 잇는다", async () => {
  const text = "완성! 총 일 분 이십삼 초, 여든일곱 수 걸렸어요. 흰색 십자가 평균보다 삼 초 빨랐어요.";
  const parts = ["완성!", "총 일 분 이십삼 초", "여든일곱 수 걸렸어요.", "흰색 십자가 평균보다", "삼 초 빨랐어요."];
  const p = new FakeSeqPlayer();
  const e = new ClipEngine(manifestOf(parts), "tts/", new FakeEngine(), p);
  await e.speak(text);
  assert.deepEqual(p.played, [], "한 파일씩 틀지 않는다");
  assert.deepEqual(
    p.seqs[0].map((x) => x.gap),
    [GAP.excl, GAP.comma, GAP.end, GAP.phrase, 0],
  );
  await e.speak("완성!");
  assert.deepEqual(p.played, ["tts/0.m4a"], "한 조각은 그냥 튼다");
});

test("Web Audio를 못 쓰면 하나씩 틀고, 겹쳐 붙이기는 쉼 0으로 바꾼다", async () => {
  const parts = ["흰색 십자가 평균보다", "삼 초 빨랐어요."];
  const p = new FakeSeqPlayer();
  p.canSeq = false;
  const e = new ClipEngine(manifestOf(parts), "tts/", new FakeEngine(), p);
  const t0 = Date.now();
  await e.speak("흰색 십자가 평균보다 삼 초 빨랐어요.");
  assert.deepEqual(p.played, ["tts/0.m4a", "tts/1.m4a"]);
  assert.ok(Date.now() - t0 < 200);
});

test("대사와 대사 사이는 한 박자 쉬고, 끊고 들어온 바로잡기는 바로 말한다", async () => {
  const p = new FakePlayer();
  const e = new ClipEngine(manifestOf(["좋아요.", "다음 수예요.", "잠깐, 반대 방향이에요!"]), "tts/", new FakeEngine(), p);
  await e.speak("좋아요.");
  let t = Date.now();
  await e.speak("다음 수예요.");
  assert.ok(Date.now() - t >= GAP.utter - 20, `쉼 없이 붙었습니다 (${Date.now() - t}ms)`);
  e.cancel();
  t = Date.now();
  await e.speak("잠깐, 반대 방향이에요!");
  assert.ok(Date.now() - t < 100, "끊은 뒤에는 기다리지 않는다");
});

test("쉬는 도중에 끊기면 바로 멈춘다", async () => {
  const p = new FakePlayer();
  const e = new ClipEngine(manifestOf(["하나.", "둘."]), "tts/", new FakeEngine(), p);
  await e.speak("하나.");
  const t = Date.now();
  const done = e.speak("둘.");
  setTimeout(() => e.cancel(), 20);
  await done;
  assert.ok(Date.now() - t < 120, `끊었는데 ${Date.now() - t}ms 뒤에 끝났습니다`);
  assert.deepEqual(p.played, ["tts/0.m4a"], "끊긴 대사는 읽지 않는다");
});

test("클립 목록이 비어 있으면 브라우저 음성으로 다 읽는다 (섞지 않고 통째로)", async () => {
  const web = new FakeEngine();
  const e = new ClipEngine({}, "tts/", web, new FakePlayer());
  await e.speak("아무 문장이에요.");
  assert.deepEqual(web.spoken, ["아무 문장이에요."]);
});

test("말하는 도중 끊기면 남은 조각은 읽지 않는다", async () => {
  const parts = ["하나.", "둘.", "셋."];
  const p = new FakePlayer();
  const e = new ClipEngine(manifestOf(parts), "tts/", new FakeEngine(), p);
  e.gapScale = 0;
  const done = e.speak("하나. 둘. 셋.");
  await new Promise((r) => setTimeout(r, 7));
  e.cancel();
  await done;
  assert.ok(p.played.length < 3, `끊었는데 ${p.played.length}조각을 다 읽었습니다`);
  assert.equal(p.stopped, 1);
});
