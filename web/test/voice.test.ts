import { test } from "node:test";
import assert from "node:assert/strict";
import { VoiceQueue, FakeEngine } from "../src/coach/voice.ts";
import { Coach } from "../src/coach/coach.ts";
import type { Utterance } from "../src/coach/coach.ts";
import { scrambled } from "./student.ts";

const u = (text: string, kind: Utterance["kind"], priority = 5): Utterance => ({ text, kind, priority });
const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));

test("새 지시가 오면 읽던 지시를 끊고 안 읽은 지시는 버린다", async () => {
  const e = new FakeEngine();
  e.msPerChar = 20;
  const q = new VoiceQueue(e);
  q.push([u("오른쪽 면을 위로 올려 주세요.", "move")]);
  await wait(30);
  q.push([u("윗면을 왼쪽으로 돌려 주세요.", "move")]);
  await wait(800);
  assert.deepEqual(e.spoken, ["오른쪽 면을 위로 올려 주세요.", "윗면을 왼쪽으로 돌려 주세요."]);
  assert.equal(e.cancelled, 1);
});

test("바로잡기는 맨 앞에서, 말하던 설명을 끊고 말한다", async () => {
  const e = new FakeEngine();
  e.msPerChar = 20;
  const q = new VoiceQueue(e);
  q.push([u("긴 단계 소개 문장입니다 아주 깁니다.", "intro"), u("목표.", "goal"), u("오른쪽 위로.", "move")]);
  await wait(30);
  q.push([u("잠깐, 반대 방향이에요!", "fix", 8)]);
  await wait(1500);
  assert.equal(e.spoken[1], "잠깐, 반대 방향이에요!");
  assert.ok(!e.spoken.includes("오른쪽 위로."), "바로잡기 전에 쌓인 지시는 버린다");
  assert.ok(e.spoken.includes("목표."), "설명은 남긴다");
});

test("칭찬 뒤에 새 말이 밀려 들어오면 칭찬은 건너뛴다", async () => {
  const e = new FakeEngine();
  e.msPerChar = 20;
  const q = new VoiceQueue(e);
  q.push([u("긴 단계 소개 문장입니다.", "intro")]);
  q.push([u("좋아요.", "praise", 1)]);
  q.push([u("다음 목표.", "goal", 3)]);
  await wait(900);
  assert.deepEqual(e.spoken, ["긴 단계 소개 문장입니다.", "다음 목표."]);
});

test("같이 들어온 칭찬과 다음 지시는 둘 다 말한다 (잡기 확인 \"좋아요.\")", async () => {
  const e = new FakeEngine();
  const q = new VoiceQueue(e);
  q.push([u("좋아요.", "praise", 1), u("알, 오른쪽 면을 위로 올려 주세요.", "move")]);
  await wait(200);
  assert.deepEqual(e.spoken, ["좋아요.", "알, 오른쪽 면을 위로 올려 주세요."]);
});

test("같이 들어온 잡기 지시는 뒤따르는 동작 지시에 지워지지 않는다", async () => {
  const e = new FakeEngine();
  e.msPerChar = 10;
  const q = new VoiceQueue(e);
  q.push([u("설명.", "intro"), u("노랑이 위, 초록이 앞에 오게 잡아 주세요.", "hold"), u("알, 오른쪽 면을 위로 올려 주세요.", "move")]);
  await wait(600);
  assert.deepEqual(e.spoken, ["설명.", "노랑이 위, 초록이 앞에 오게 잡아 주세요.", "알, 오른쪽 면을 위로 올려 주세요."]);
  // 앞서 받아 두고 못 읽은 잡기 지시는 새 동작 지시가 오면 지운다
  q.push([u("긴 단계 소개 문장입니다 아주 깁니다.", "intro"), u("빨강이 위, 파랑이 앞에 오게 잡아 주세요.", "hold")]);
  await wait(30);
  q.push([u("유, 윗면을 왼쪽으로 돌려 주세요.", "move")]);
  await wait(900);
  assert.ok(!e.spoken.includes("빨강이 위, 파랑이 앞에 오게 잡아 주세요."));
  assert.equal(e.spoken.at(-1), "유, 윗면을 왼쪽으로 돌려 주세요.");
});

test("대기열에 들어온 문장은 차례가 오기 전에 미리 준비시킨다 (파일 받기)", async () => {
  const e = new FakeEngine() as FakeEngine & { prepared: string[]; prepare(t: string): void };
  e.prepared = [];
  e.prepare = (t: string) => void e.prepared.push(t);
  e.msPerChar = 5;
  const q = new VoiceQueue(e);
  q.push([u("단계 소개예요.", "intro"), u("목표예요.", "goal")]);
  assert.ok(e.prepared.includes("목표예요."), "뒤에 기다리는 문장도 준비한다");
  await wait(200);
});

test("한 번에 들어온 바로잡기 두 개는 들어온 차례대로 말한다", async () => {
  const e = new FakeEngine();
  e.msPerChar = 10;
  const q = new VoiceQueue(e);
  q.push([u("긴 단계 소개 문장입니다.", "intro"), u("다음 목표.", "goal")]);
  await wait(20);
  q.push([u("괜찮아요. 지금 상태에서 다시 안내할게요.", "fix", 8), u("흰색 십자가 흐트러졌어요. 먼저 다시 맞춰요.", "fix", 8), u("오른쪽 위로.", "move")]);
  await wait(1200);
  assert.deepEqual(e.spoken.slice(1, 3), ["괜찮아요. 지금 상태에서 다시 안내할게요.", "흰색 십자가 흐트러졌어요. 먼저 다시 맞춰요."]);
  assert.equal(e.spoken.at(-1), "오른쪽 위로.");
});

// --- 낡은 바로잡기 ("되돌리세요"를 이미 따른 뒤) ------------------------------------------

test("바로잡는 중에 새 지시가 오면(\"한 번 더요.\") 앞선 바로잡기는 낡았다: 읽던 것은 끊고 안 읽은 것은 버린다", async () => {
  const e = new FakeEngine();
  e.msPerChar = 20;
  const q = new VoiceQueue(e);
  q.setFixing(true); // 반대로 돌려서, 같은 면을 두 번 더 돌려야 한다
  q.push([u("반대 방향이에요. 알 투, 오른쪽 면을 두 번 돌려 주세요.", "fix", 8), u("두 번째 바로잡기 문장이에요.", "fix", 8)]);
  await wait(30);
  q.setFixing(true); // 한 번 돌렸다: 아직 바로잡는 중
  q.push([u("한 번 더요.", "move")]);
  await wait(900);
  assert.deepEqual(e.spoken, ["반대 방향이에요. 알 투, 오른쪽 면을 두 번 돌려 주세요.", "한 번 더요."]);
  assert.equal(e.cancelled, 1, "읽던 바로잡기를 끊었다");
});

test("바로잡기가 아닐 때 들어온 바로잡기 안내(공식 연습의 \"엉켰네요. …\")는 뒤따르는 지시가 끊지 않는다", async () => {
  const e = new FakeEngine();
  e.msPerChar = 10;
  const q = new VoiceQueue(e);
  q.setFixing(false); // 공식 연습: 코치가 없어 늘 false
  q.push([u("엉켰네요. 정답은 T였어요. 맞춰 드릴 테니 다시 해 봐요.", "fix", 8), u("알, 오른쪽 면을 위로 올려 주세요.", "move")]);
  await wait(30);
  q.setFixing(false);
  q.push([u("유, 윗면을 왼쪽으로 돌려 주세요.", "move")]); // 카드를 보고 먼저 따라 했다
  await wait(1000);
  assert.deepEqual(e.spoken, ["엉켰네요. 정답은 T였어요. 맞춰 드릴 테니 다시 해 봐요.", "유, 윗면을 왼쪽으로 돌려 주세요."]);
  assert.equal(e.cancelled, 0, "안내는 끝까지 읽는다 (앞서 받아 둔 지시만 버린다)");
});

test("바로잡기가 끝나면(setFixing(false)) 새 말이 없어도 읽던 바로잡기를 끊고 기다리던 바로잡기는 버린다", async () => {
  const e = new FakeEngine();
  e.msPerChar = 20;
  const q = new VoiceQueue(e);
  q.setFixing(true);
  q.push([u("앗, 방금은 앞면을 돌렸어요. 먼저 되돌릴게요.", "fix", 8), u("에프 프라임, 앞면을 반시계 방향으로 돌려 주세요.", "fix", 8)]);
  await wait(30);
  q.setFixing(true); // 아직 되돌리는 중: 그대로 둔다
  assert.equal(e.cancelled, 0);
  q.setFixing(false); // 되돌렸다 (코치는 아무 말도 하지 않았다)
  assert.equal(e.cancelled, 1, "읽던 바로잡기를 끊었다");
  await wait(900);
  assert.deepEqual(e.spoken, ["앗, 방금은 앞면을 돌렸어요. 먼저 되돌릴게요."], "기다리던 바로잡기는 읽지 않는다");
});

test("바로잡기가 끝난 뒤에 들어온 바로잡기 말('괜찮아요…')은 시간이 흘러도 지우지 않는다", async () => {
  const e = new FakeEngine();
  e.msPerChar = 10;
  const q = new VoiceQueue(e);
  q.setFixing(true);
  q.push([u("엘 프라임, 왼쪽 면을 위로 올려 주세요.", "fix", 8)]);
  await wait(20);
  // 코치가 바로잡기를 그만두고 지금 상태에서 다시 안내한다: 끝났다고 먼저 알리고, 같이 온 새 말을 넣는다
  q.setFixing(false);
  q.push([u("괜찮아요. 지금 상태에서 다시 안내할게요.", "fix", 8), u("다음 목표.", "goal", 3)]);
  await wait(10);
  const cut = e.cancelled; // 여기까지 끊은 것은 낡은 바로잡기뿐이다
  assert.equal(e.spoken.at(-1), "괜찮아요. 지금 상태에서 다시 안내할게요.");
  for (let i = 0; i < 5; i++) {
    await wait(40);
    q.setFixing(false); // 시간이 흐름: 말이 없어도 계속 알린다 (main.ts 의 emit)
  }
  await wait(400);
  assert.deepEqual(e.spoken, ["엘 프라임, 왼쪽 면을 위로 올려 주세요.", "괜찮아요. 지금 상태에서 다시 안내할게요.", "다음 목표."]);
  assert.equal(e.cancelled, cut, "'괜찮아요…'는 끊지 않는다");
});

/** main.ts 의 emit 과 같은 순서: 바로잡기 중인지 먼저 알리고(받은 말이 없어도), 받은 말을 넣는다 */
const wire = (coach: Coach, q: VoiceQueue) => (us: Utterance[]) => {
  q.setFixing(coach.fixing);
  if (us.length) q.push(us);
};
const otherFace = (c: number) => [0, 1, 2, 3, 4, 5].find((x) => x !== c && x !== [3, 4, 5, 0, 1, 2][c])!;

test("코치: 엉뚱한 면을 돌렸다가 '되돌리세요'를 다 듣기 전에 되돌리면, 그 말을 끊고 다음 지시로 넘어간다", async () => {
  const coach = new Coach(scrambled(3), { method: "beginner", level: "guided", taughtBasics: true });
  coach.pose = { up: 3, front: 2 };
  coach.start(0); // 시작 안내는 이미 들은 셈 친다
  const e = new FakeEngine();
  e.msPerChar = 10;
  const q = new VoiceQueue(e);
  const emit = wire(coach, q);
  const other = otherFace(coach.expected()!.color);
  const fix = coach.onMove({ color: other, k: 1 }, 1000);
  assert.equal(fix[0].kind, "fix");
  emit(fix);
  await wait(50); // "… 되돌릴게요. …"를 말하는 도중에 되돌린다
  const next = coach.onMove({ color: other, k: 3 }, 1100);
  assert.ok(next.some((x) => x.kind === "move"), "되돌리면 다음 지시를 한다");
  emit(next);
  assert.ok(e.cancelled >= 1, "'되돌리세요'를 끝까지 읽지 않는다");
  await wait(20);
  assert.equal(e.spoken.at(-1), next[0].text, "다음 지시를 바로 말한다 (되돌리세요가 끝나기를 기다리지 않는다)");
  await wait(900);
  assert.deepEqual(e.spoken, [fix[0].text, ...next.map((x) => x.text)]);
});

test("코치: 되돌린 뒤 코치가 아무 말도 하지 않아도(익숙한 섹시무브 도중) 남은 '되돌리세요'를 끊는다", async () => {
  // 섹시무브를 두 번 깨끗하게 한 뒤에는 묶음 이름("섹시무브.")만 부르고, 묶음 도중에는 말하지 않는다
  const coach = new Coach(scrambled(1), { method: "beginner", level: "guided", taughtBasics: true });
  coach.start(0);
  let t = 0;
  let found = false;
  for (let i = 0; i < 400 && !found; i++) {
    t += 500;
    found = coach.onMove(coach.expected()!, t).some((x) => x.kind === "move" && /섹시무브\.$/.test(x.text));
  }
  assert.ok(found, "묶음 이름만 부르는 장면을 못 찾음");
  assert.deepEqual(coach.onMove(coach.expected()!, (t += 500)), [], "묶음 도중에는 말하지 않는다");
  const e = new FakeEngine();
  e.msPerChar = 10;
  const q = new VoiceQueue(e);
  const emit = wire(coach, q);
  const other = otherFace(coach.expected()!.color);
  const fix = coach.onMove({ color: other, k: 1 }, (t += 500));
  assert.equal(fix[0].kind, "fix");
  emit(fix);
  await wait(50);
  const after = coach.onMove({ color: other, k: 3 }, (t += 100));
  assert.deepEqual(after, [], "되돌린 뒤 코치는 말이 없다 (그래서 새 지시가 바로잡기를 밀어내지 못한다)");
  emit(after);
  assert.equal(e.cancelled, 1, "그래도 '되돌리세요'는 끊는다");
  await wait(700);
  assert.deepEqual(e.spoken, [fix[0].text]);
});
