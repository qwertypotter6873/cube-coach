/**
 * 코치가 하는 말 전체를 모아 "미리 합성할 클립 목록"을 만든다.
 *   node test/inventory.ts <시드 시작> <시드 개수> <출력 json>
 *   (보통: node test/inventory.ts 1 40 ../reports/tts_inventory_train.json)
 *
 * 말해지는 것은 main.ts 가 voice.push 로 넣는 Utterance 뿐이다. 모으는 곳
 *   1. 가상 풀이 (coach.ts / drill.ts) — 방법·수준·실수율을 바꿔 가며 돌린다
 *   2. main.ts 의 고정 대사 (카메라로 다 읽음, 목소리 바꾸기) — 소스에서 글자 그대로 뽑는다
 *      (카메라 흐름의 나머지 안내는 화면 글자(camStatus)라 말하지 않는다)
 *   3. progress.ts 의 기록 피드백 — 실제 함수를 불러 만든다
 *   4. 숫자가 들어가 끝없이 바뀌는 자리 — 쓸 만한 범위를 통째로 미리 만들어 둔다
 *      가상 풀이에서 숫자만 바뀌며 두 번 넘게 나온 문장은 그 숫자 범위를 다 채운다 (expandNumbers)
 *
 * 목록은 "문장 단위 조각"으로 적는다 (voice.ts 의 buildPieces). 코치가 문장을 새로 엮어도
 * 조각을 이어 붙여 읽을 수 있고, 같은 문장을 여러 번 합성하지 않아 양도 준다.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { Coach } from "../src/coach/coach.ts";
import type { Utterance } from "../src/coach/coach.ts";
import { durationWord } from "../src/coach/coach.ts";
import { isSolved, mulberry32, applySlotAlg, SOLVED, randomScramble, ALL_VIEWS, applyColorMoves, translate } from "../src/coach/cube.ts";
import type { AlgMove, ColorMove, State, View } from "../src/coach/cube.ts";
import { Drill } from "../src/coach/drill.ts";
import { BEGINNER_ALGS, OCLL, OLL_EDGES, PLL, STAGES, YELLOW, analyze, searchAlgsPublic } from "../src/coach/method.ts";
import type { Method, StageId } from "../src/coach/method.ts";
import { STAGE_NAME, countWord, holdText, josa, moveSay, moveText, orientationLines, reframeText, secondsWord, sinoWord, symbolSay } from "../src/coach/speech.ts";
import { feedbackText } from "../src/coach/progress.ts";
import type { StageCompare } from "../src/coach/progress.ts";
import { buildPieces, planWithKind, ttsTextFor } from "../src/coach/voice.ts";
import type { PlanKind } from "../src/coach/voice.ts";

const SRC = new URL("../src/", import.meta.url);

// 미리 만들어 둘 숫자 범위 (넘어가면 그 문장만 못 읽고 콘솔에 알린다)
export const RANGE = {
  /** 풀이 전체 시간: 0초 ~ 11분 59초 */
  totalSec: 12 * 60,
  /** 단계 하나의 기록 ("최고 기록이에요! 삼백십이 초."): 0 ~ 599초. 처음 배우는 사람은 한 단계에 몇 분 걸린다 */
  stageSec: 600,
  /** 평균과의 차이, 공식 연습 한 번: 0 ~ 299초 */
  sec: 300,
  /** 한 판에 둔 수: coach.ts 가 아흔아홉까지만 센다 */
  moves: 100,
  /** 아주 오래 걸렸을 때 분만 따로 붙이기 */
  min: 60,
  /** 숫자만 바뀌는 문장의 "몇 수 / 몇 번": 본 것 가운데 가장 큰 수의 두 배까지 (적어도 열) */
  countMin: 10,
};

// --- 1. 가상 풀이 ------------------------------------------------------------------

/** 여러 방법·수준·실수율로 가상 풀이를 돌려 나온 문장을 센다 */
export function simulate(from: number, count: number, into = new Map<string, number>()): Map<string, number> {
  const add = (us: Utterance[]) => us.forEach((u) => into.set(u.text, (into.get(u.text) ?? 0) + 1));

  for (let seed = from; seed < from + count; seed++) {
    for (const method of ["beginner", "cfop4", "cfop"] as const) {
      for (const level of ["guided", "hint", "observe"] as const) {
        const rand = mulberry32(seed * 131 + method.length * 7 + level.length);
        const s = applySlotAlg(SOLVED, randomScramble(25, rand));
        const coach = new Coach(s, { method, level, hintAfterMs: 4000, repeatAfterMs: 4000 });
        // 사람이 큐브를 여러 방향으로 들고 있다고 가정 (지시 문장이 방향마다 달라진다)
        coach.pose = rand() < 0.5 ? null : ALL_VIEWS[Math.floor(rand() * 24)];
        add(coach.start(0));
        let t = 0;
        for (let i = 0; i < 1500 && !isSolved(coach.s); i++) {
          t += 600;
          if (rand() < 0.03) add(coach.tick(t + 10000)), (t += 10000);
          let m = coach.expected();
          if (!m) break;
          if (rand() < 0.06) {
            const wrong = { color: Math.floor(rand() * 6), k: (1 + Math.floor(rand() * 3)) as 1 | 2 | 3 };
            add(coach.onTurning(wrong.color, (wrong.k === 3 ? -1 : 1) * 0.8, t));
            m = wrong;
          }
          // 엉뚱한 면을 잇달아 돌려 "괜찮아요. 지금 상태에서 다시 안내할게요." 까지 가 본다
          if (rand() < 0.02) {
            for (let j = 0; j < 4; j++) add(coach.onMove({ color: Math.floor(rand() * 6), k: 1 }, (t += 300)));
          }
          if (rand() < 0.05 && coach.pose) {
            const v = ALL_VIEWS[Math.floor(rand() * 24)];
            add(coach.onPose(v, t));
          }
          add(coach.onMove(m, t));
        }
      }
    }
    // 카메라로 읽은 큐브에서 바로 시작하는 흐름: 이미 어느 정도 맞춰진 상태, 다 맞춰진 상태
    for (const method of ["beginner", "cfop"] as const) {
      const rand = mulberry32(seed * 7717 + method.length);
      for (const n of [0, 4, 8, 14]) {
        const st = applySlotAlg(SOLVED, randomScramble(n, rand));
        for (const level of ["guided", "hint", "observe"] as const) {
          const c = new Coach(st, { method, level });
          c.pose = ALL_VIEWS[Math.floor(rand() * 24)];
          add(c.start(0));
        }
      }
    }
  }

  // 공식 연습 모드에서 엉망으로 돌려 보기 (맞춰 드릴게요 / 엉켰네요 / 다른 상황이 됐네요)
  for (let seed = from; seed < from + Math.max(2, Math.ceil(count / 8)); seed++) {
    for (const [kind, pool] of [["pll", PLL], ["oll", OCLL]] as const) {
      const rand = mulberry32(seed * 523 + kind.length);
      const d = new Drill(SOLVED, pool, kind, rand);
      let t = 0;
      add(d.start(t));
      for (let i = 0; i < 400; i++) {
        const m = rand() < 0.45 || !d.expected() ? { color: Math.floor(rand() * 6), k: (1 + Math.floor(rand() * 3)) as 1 | 2 | 3 } : d.expected()!;
        add(d.onMove(m, (t += 300)));
      }
      // 1·2층은 안 깨뜨리면서 목표도 아닌 수를 길게 (수네를 거듭) → "다른 상황이 됐네요."
      const e = new Drill(SOLVED, pool, kind, mulberry32(seed * 11 + 3));
      add(e.start((t = 0)));
      for (let g = 0; g < 300 && (e.phase === "setup" || e.phase === "restore"); g++) add(e.onMove(e.expected()!, (t += 250)));
      for (let r = 0; r < 4 && e.phase === "solve"; r++) {
        for (const m of translate(e.s, "R U R' U R U2 R'", { up: YELLOW, front: 2 })) add(e.onMove({ color: m.color, k: m.k }, (t += 250)));
      }
    }
  }

  // 공식 연습 모드
  for (let seed = from; seed < from + Math.max(1, Math.ceil(count / 4)); seed++) {
    for (const [kind, pool] of [["pll", PLL], ["oll", OCLL]] as const) {
      const rand = mulberry32(seed * 977 + kind.length);
      const d = new Drill(seed % 3 === 0 ? applySlotAlg(SOLVED, randomScramble(20, rand)) : SOLVED, pool, kind, rand);
      let t = 0;
      add(d.start(t));
      for (let round = 0; round < pool.length; round++) {
        let guard = 0;
        while ((d.phase === "setup" || d.phase === "restore") && guard++ < 300) {
          let m = d.expected()!;
          if (rand() < 0.05) m = { color: Math.floor(rand() * 6), k: 1 };
          else if (m.k === 2 && rand() < 0.3) add(d.onMove({ color: m.color, k: 1 }, (t += 300))), (m = { color: m.color, k: 1 });
          add(d.onMove(m, (t += 300)));
        }
        if (d.phase !== "solve") break;
        t += 1000 + rand() * 4000;
        const goal = (x: State) => (kind === "oll" ? analyze(x).oriented : analyze(x).auf >= 0);
        const f = searchAlgsPublic(d.s, pool, goal)!;
        const pre = f.auf ? [{ color: YELLOW, k: f.auf as 1 | 2 | 3 }] : [];
        const ms = [...pre, ...translate(applyColorMoves(d.s, pre), f.alg.alg, f.view).map((m) => ({ color: m.color, k: m.k }))];
        for (const m of ms) add(d.onMove(m, (t += 250 + rand() * 300)));
      }
    }
  }
  return into;
}

// --- 2. 소스에서 글자 그대로 뽑기 -------------------------------------------------------

const read = (name: string) => readFileSync(new URL(name, SRC), "utf8");
const STR = String.raw`"((?:[^"\\]|\\.)*)"`;

/** main.ts 가 voice.push 로 바로 넣는 고정 대사. 글자가 아닌 자리는 아래 목록에 적어 둔다. */
const MAIN_COMPUTED: Record<string, string> = {
  fb: "progress.feedbackText — 아래 feedbackSentences() 로 따로 모은다",
};
const KINDS = "status|intro|goal|hold|move|praise|fix|stage|hint|summary";

/**
 * main.ts 에서 { text: ..., kind: "..." } 꼴(순서는 상관없음)의 대사 객체를 다 찾는다.
 * 글자 그대로면 모으고, 이름(변수)이면 MAIN_COMPUTED 에 적혀 있어야 한다 — 없으면 멈춰서 알린다.
 */
export function mainLiterals(): string[] {
  const src = read("main.ts");
  const out: string[] = [];
  const isUtter = new RegExp(String.raw`\bkind:\s*["'](?:${KINDS})["']`);
  for (const [body] of src.matchAll(/\{[^{}]*\}/g)) {
    if (!isUtter.test(body)) continue;
    const m = /\btext:\s*(?:"((?:[^"\\]|\\.)*)"|'((?:[^'\\]|\\.)*)'|`([^`$]*)`|([A-Za-z_$][\w$.]*))/.exec(body);
    const what = m ? null : body.replace(/\s+/g, " ").slice(0, 80);
    if (!m) throw new Error(`main.ts 에 읽을 수 없는 대사가 있습니다: ${what} — test/inventory.ts 의 mainLiterals 를 고쳐 주세요.`);
    if (m[1] !== undefined) out.push(JSON.parse(`"${m[1]}"`) as string);
    else if (m[2] !== undefined) out.push(m[2].replace(/\\'/g, "'"));
    else if (m[3] !== undefined) out.push(m[3]);
    else if (!(m[4] in MAIN_COMPUTED)) {
      throw new Error(`main.ts 에 글자가 아닌 대사가 생겼습니다: ${m[4]} — test/inventory.ts 의 MAIN_COMPUTED 에 적고 그 문장들을 모아 주세요.`);
    }
  }
  if (!out.length) throw new Error("main.ts 에서 고정 대사를 하나도 못 찾았습니다 (모으는 방식을 고쳐야 합니다).");
  return out;
}

/** coach.ts·drill.ts 의 say("...") 자리에 적힌 글 (빠진 대사가 없는지 맞춰 보는 용도) */
export function saySiteLiterals(): string[] {
  const re = new RegExp(String.raw`say\(\s*${STR}`, "g");
  return ["coach/coach.ts", "coach/drill.ts"].flatMap((f) => [...read(f).matchAll(re)].map((m) => JSON.parse(`"${m[1]}"`) as string));
}

// --- 3. 기록 피드백 (progress.ts) ------------------------------------------------------

const cmp = (stage: StageId, now: number, avg: number | null, best: number | null): StageCompare[] => [{ stage, now, avg, best }];
const ALL_STAGES = [...new Set(Object.values(STAGES).flat())] as StageId[];

/** feedbackText 가 낼 수 있는 문장 전부 (단계 × 가지 × 초) */
export function feedbackSentences(secMax = RANGE.sec, stageMax = RANGE.stageSec): string[] {
  const out = new Set<string>();
  for (const st of ALL_STAGES) {
    for (let sec = 0; sec < stageMax; sec++) {
      out.add(feedbackText(cmp(st, sec * 1000, sec * 1000 + 9e4, sec * 1000 + 5000))!); // 최고 기록
      if (sec >= 1 && sec < secMax) {
        out.add(feedbackText(cmp(st, 1000, 1000 + sec * 1000 + 1, 500))!); // 평균보다 빠름
        out.add(feedbackText(cmp(st, 1000 + sec * 1000 + 1, 1000, 500))!); // 평균보다 느림
      }
    }
    out.add(feedbackText(cmp(st, 1000, 1200, 500))!); // 평소와 비슷
  }
  return [...out].filter((x) => x);
}

// --- 4. 숫자가 들어가 끝없이 바뀌는 자리 ------------------------------------------------

/**
 * 여러 수를 쉼표로 늘어놓을 때 쓰는 동작 말 (speech.ts 의 chunkText 와 같은 꼴로 만든다).
 * 공식은 수를 고르는 가짓수가 끝이 없어 통째로는 못 만든다 — 한 수씩 조각으로 두고 이어 붙인다.
 */
export function movePieces(): string[] {
  const view: View = { up: 3, front: 2 };
  const out = new Set<string>();
  for (let color = 0; color < 6; color++) {
    for (const k of [1, 2, 3] as const) {
      const ms = [{ color, k } as ColorMove, { color, k, token: "r", viewBefore: view } as AlgMove];
      for (const m of ms) {
        const say = moveSay(SOLVED, view, m, "short", " "); // chunkText 가 쓰는 꼴
        for (const p of [say, `${say}.`, `처음은 ${say}`, `처음은 ${say}.`]) out.add(p);
        out.add(symbolSay(SOLVED, view, m));
        out.add(moveText(SOLVED, view, m, "short"));
      }
    }
  }
  return [...out];
}

/**
 * 가짓수가 뻔한데 가상 풀이에서는 어쩌다 한 번만 나오는 말을, 실제 함수를 불러 빠짐없이 만든다.
 * (들고 있는 방향, 공식 이름 — 수집기가 운에 기대지 않게)
 */
export function enumeratedLines(): string[] {
  const out = new Set<string>();
  for (const now of ALL_VIEWS) {
    out.add(holdText(now));
    out.add(reframeText(null, now));
    for (const prev of ALL_VIEWS) out.add(reframeText(prev, now));
  }
  for (const line of [true, false].flatMap((t) => (["beginner", "cfop4", "cfop"] as Method[]).flatMap((m) => orientationLines(m, t)))) out.add(line);
  // 공식 이름이 들어가는 말 (speech.ts llText / coach.escalateHint / drill.next 와 같은 꼴)
  const names = [...PLL, ...OCLL, ...OLL_EDGES, ...Object.values(BEGINNER_ALGS)].map((a) => a.name);
  for (const n of [...new Set([...names, "섹시무브", "공식"])]) {
    out.add(`${n}.`);
    out.add(`${n} 공식을 할게요.`);
    out.add(`PLL ${n} 공식이에요.`);
    out.add(`${josa.eulReul(n)} 써 보세요.`);
    out.add(`정답은 ${josa.ieosseoyo(n)}.`); // 공식 연습: 풀고 나서야 이름을 알려 준다 (drill.ts)
  }
  return [...out];
}

/** 숫자만 바뀌는 조각을 범위대로 미리 만들어 둔다 */
export function numberPieces(): string[] {
  const out = new Set<string>();
  // 완성 정리: "완성! 총 일 분 이십삼 초, 여든일곱 수 걸렸어요."
  out.add("완성!");
  // durationWord 는 초를 반올림한다 — 오십구 점 몇 초는 "육십 초"가 되므로 그 자리도 만들어 둔다
  for (let m = 0; m <= RANGE.totalSec / 60; m++) {
    for (let s = 0; s < 60; s++) out.add(`총 ${durationWord(m * 60 + s)}`);
    out.add(`총 ${durationWord(m * 60 + 59.6)}`);
  }
  for (let m = 1; m <= RANGE.min; m++) out.add(`총 ${sinoWord(m)} 분`); // 아주 오래 걸렸을 때 분·초를 나눠 붙인다
  for (let s = 0; s <= 60; s++) out.add(`${sinoWord(s)} 초`);
  for (let s = 0; s < RANGE.stageSec; s++) out.add(`${secondsWord(s)}.`); // "최고 기록이에요! 삼백십이 초."
  for (let n = 0; n < RANGE.moves; n++) out.add(`${countWord(n)} 수 걸렸어요.`);
  out.add(`${countWord(RANGE.moves - 1)} 수 넘게 걸렸어요.`);
  for (const st of ALL_STAGES) out.add(`가장 오래 걸린 단계는 ${josa.ieosseoyo(STAGE_NAME[st])}.`);
  // 공식 연습 정리: "십이 초. 알아보는 데 삼 초 걸렸어요."
  for (let s = 0; s < RANGE.sec; s++) {
    out.add(`${secondsWord(s)}.`);
    out.add(`알아보는 데 ${secondsWord(s)} 걸렸어요.`);
  }
  for (const mv of movePieces()) out.add(mv);
  // 자주 앞에 붙는 토막
  out.add("다시");
  for (const o of ["첫 번째", "두 번째", "세 번째", "네 번째", "다섯 번째", "여섯 번째"]) out.add(`${o}.`);
  return [...out];
}

// --- 숫자만 바뀌는 문장 넓히기 ------------------------------------------------------

const COUNT_WORDS = Array.from({ length: 99 }, (_, i) => countWord(i + 1)).sort((x, y) => y.length - x.length);
/** "여덟 수", "세 번"(번째·수네는 뺀다), "삼백십이 초", "십삼 분"(초록 같은 낱말은 뺀다) */
const NUM_SLOT = new RegExp(
  String.raw`(?<=^|\s)(?:(${COUNT_WORDS.join("|")}) (수(?!네)|번(?!째))|([영일이삼사오육칠팔구십백]+) (초|분)(?=[\s.,!?]|$))`,
  "g",
);

/**
 * 가상 풀이에서 숫자만 다르게 두 번 넘게 나온 문장은, 운에 기대지 않게 그 숫자 범위를 다 만든다.
 * (예: "여섯 수예요." "세 수예요." → 한 수예요 ~ 열여섯 수예요). 숫자가 한 군데만 있는 문장만 넓힌다.
 * 코치 담당이 문장 꼴을 바꿔도 새 꼴을 그대로 따라간다.
 */
export function expandNumbers(pieces: Iterable<string>, explicit: ReadonlySet<string> = new Set()): string[] {
  const groups = new Map<string, { unit: string; values: Set<number>; make: (n: number) => string }>();
  const countOf = new Map(Array.from({ length: 99 }, (_, i) => [countWord(i + 1), i + 1]));
  const sinoOf = new Map(Array.from({ length: 1000 }, (_, i) => [sinoWord(i), i]));
  for (const p of pieces) {
    const hits = [...p.matchAll(NUM_SLOT)];
    if (hits.length !== 1) continue;
    const m = hits[0];
    const unit = m[2] ?? m[4];
    const value = m[1] !== undefined ? countOf.get(m[1]) : sinoOf.get(m[3]);
    if (value === undefined) continue;
    const head = p.slice(0, m.index);
    const tail = p.slice(m.index! + m[0].length);
    const key = `${head}\u0000${unit}\u0000${tail}`;
    const word = m[1] !== undefined ? countWord : sinoWord;
    const g = groups.get(key) ?? { unit, values: new Set<number>(), make: (n: number) => `${head}${word(n)} ${unit}${tail}` };
    g.values.add(value);
    groups.set(key, g);
  }
  const out = new Set<string>();
  for (const g of groups.values()) {
    if (g.values.size < 2) continue;
    const top = Math.max(...g.values);
    const [lo, hi] =
      g.unit === "초" ? [0, RANGE.sec - 1] : g.unit === "분" ? [1, RANGE.min] : [1, Math.min(99, Math.max(RANGE.countMin, top * 2))];
    const all = Array.from({ length: hi - lo + 1 }, (_, i) => g.make(lo + i));
    // numberPieces 가 이미 따로 범위를 정해 둔 집안은 건드리지 않는다 (예: "총 … 초"는 분까지 붙는다)
    if (all.filter((x) => explicit.has(x)).length >= 2) continue;
    for (const x of all) out.add(x);
  }
  return [...out];
}

// --- 목록 만들기 ---------------------------------------------------------------------

export interface ClipEntry {
  text: string;
  /** 모델에 넣을 글 (이어지는 조각은 끝을 내려 읽지 않게 쉼표를 붙인다). text 와 같으면 없다. */
  tts?: string;
  n?: number;
}

/** 말해지는 문장 전부 (조각으로 나누기 전) */
export function allSpoken(from = 1, count = 40): Map<string, number> {
  const m = simulate(from, count);
  // 소스에 글자 그대로 적힌 대사는 가상 풀이가 그 상황을 못 만들어도 반드시 넣는다
  for (const t of [...mainLiterals(), ...saySiteLiterals(), ...feedbackSentences(), ...enumeratedLines()]) m.set(t, (m.get(t) ?? 0) + 1);
  return m;
}

/** 미리 합성할 클립 목록 (자주 나오는 조각부터: 합성 중에도 앱이 먼저 쓰는 말부터 채워진다) */
export function buildList(spoken: Map<string, number>): ClipEntry[] {
  const n = new Map<string, number>();
  const bump = (t: string, c: number) => n.set(t, (n.get(t) ?? 0) + c);
  for (const [text, c] of spoken) for (const piece of buildPieces(text)) bump(piece, c);
  const fixed = numberPieces();
  for (const piece of expandNumbers([...n.keys()], new Set(fixed))) bump(piece, 0);
  for (const piece of fixed) bump(piece, 0);
  return [...n.entries()]
    .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))
    .map(([text, c]) => {
      const tts = ttsTextFor(text);
      return tts === text ? { text, n: c } : { text, tts, n: c };
    });
}

/** 말하는 문장이 어떤 방법으로 풀리는지 센다 (통째 / 문장 잇기 / 숫자 본 / 쉼표 나누기 / 못 품) */
export function planStats(spoken: Iterable<string>, has: (t: string) => boolean): Record<PlanKind, number> {
  const out: Record<PlanKind, number> = { whole: 0, sentences: 0, template: 0, comma: 0, none: 0 };
  for (const t of spoken) out[planWithKind(t, has).kind]++;
  return out;
}

if (import.meta.filename === process.argv[1]) {
  const [from = "1", count = "40", out = ""] = process.argv.slice(2);
  const spoken = allSpoken(Number(from), Number(count));
  const list = buildList(spoken);
  if (out) writeFileSync(out, JSON.stringify(list, null, 1));
  const keys = new Set(list.map((x) => x.text));
  const st = planStats(spoken.keys(), (t) => keys.has(t));
  const seen = list.filter((x) => (x.n ?? 0) > 0).length;
  console.log(
    `말하는 문장 ${spoken.size}가지 → 합성할 조각 ${list.length}개 (대사에서 바로 나온 조각 ${seen}개, 숫자 범위로 채운 조각 ${list.length - seen}개)\n` +
      `풀리는 방법: 통째 ${st.whole}, 문장 잇기 ${st.sentences}, 숫자 본 ${st.template}, 쉼표 나누기 ${st.comma}, 못 품 ${st.none}`,
  );
}
