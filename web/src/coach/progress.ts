/**
 * 실력 기록: 풀 때마다 단계별 시간과 공식별 반응·실행 시간을 남기고, 지난 기록과 비교한다.
 *
 * - 단계별 시간(splits): 어느 단계가 느린지
 * - 공식별 반응 시간: 안내를 듣거나 상황을 보고 첫 수를 두기까지 (= 상황 인식이 얼마나 빠른가)
 * - 공식별 실행 시간과 실수: 손이 공식을 얼마나 외웠는가
 * 이것으로 "다음에 연습할 공식"을 고른다.
 *
 * 저장은 기기의 브라우저 저장소(localStorage)에 한다. 쓸 수 없는 환경에서는 메모리에만 둔다.
 */

import type { StepRecord } from "./coach.ts";
import type { Method, StageId } from "./method.ts";
import { STAGES } from "./method.ts";
import { STAGE_NAME, josa, secondsWord } from "./speech.ts";

export interface SolveRecord {
  at: number;
  method: Method;
  level: string;
  totalMs: number;
  moves: number;
  mistakes: number;
  splits: Partial<Record<StageId, number>>; // 단계가 끝난 시각 (시작 기준)
  steps: StepRecord[];
}

const KEY = "cubecoach.history.v1";
const BASICS_KEY = "cubecoach.basics.v1";
let memory: SolveRecord[] = [];
let basicsMemory = 0;

/** 기준 면·기호 설명을 들어 본 적이 있는가 (두 번째부터는 한 줄만 말한다) */
export function basicsTaught(): boolean {
  try {
    const raw = localStorage.getItem(BASICS_KEY);
    if (raw !== null) return Number(raw) > 0;
  } catch {
    /* 저장소를 못 쓰면 메모리 기록만 */
  }
  return basicsMemory > 0;
}

export function noteBasicsTaught(): void {
  basicsMemory++;
  try {
    localStorage.setItem(BASICS_KEY, String(Number(localStorage.getItem(BASICS_KEY) ?? 0) + 1));
  } catch {
    /* 무시 */
  }
}

export function loadHistory(): SolveRecord[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) return JSON.parse(raw) as SolveRecord[];
  } catch {
    /* 저장소를 못 쓰면 메모리 기록만 */
  }
  return memory;
}

export function saveSolve(r: SolveRecord): void {
  const all = [...loadHistory(), r].slice(-200);
  memory = all;
  try {
    localStorage.setItem(KEY, JSON.stringify(all));
  } catch {
    /* 무시 */
  }
}

/** 단계별로 걸린 시간 (끝난 시각의 차이) */
export function stageDurations(r: SolveRecord): Partial<Record<StageId, number>> {
  const out: Partial<Record<StageId, number>> = {};
  let prev = 0;
  for (const st of STAGES[r.method]) {
    const end = r.splits[st];
    if (end === undefined) continue;
    out[st] = end - prev;
    prev = end;
  }
  return out;
}

export interface StageCompare {
  stage: StageId;
  now: number;
  avg: number | null;
  best: number | null;
}

export function compareStages(cur: SolveRecord, history: SolveRecord[]): StageCompare[] {
  const past = history.filter((h) => h.method === cur.method && h.at !== cur.at);
  const durs = past.map(stageDurations);
  const now = stageDurations(cur);
  return STAGES[cur.method]
    .filter((st) => now[st] !== undefined)
    .map((st) => {
      const xs = durs.map((d) => d[st]).filter((x): x is number => x !== undefined);
      return { stage: st, now: now[st]!, avg: xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null, best: xs.length ? Math.min(...xs) : null };
    });
}

export interface CaseStat {
  name: string; // 공식 이름 (예: "PLL Jb", "수네")
  n: number;
  reactMs: number;
  execMs: number;
  mistakeRate: number;
  score: number; // 클수록 연습이 필요
}

/** 공식(이름이 있는 덩어리)별 평균. 최근 30번까지만 본다. */
export function caseStats(history: SolveRecord[]): CaseStat[] {
  const acc = new Map<string, { n: number; react: number; exec: number; mistakes: number }>();
  for (const r of history.slice(-30)) {
    for (const s of r.steps) {
      if (!s.alg) continue;
      const name = s.caseName ? `${STAGE_NAME[s.stage]} ${s.caseName}` : STAGE_NAME[s.stage];
      const a = acc.get(name) ?? { n: 0, react: 0, exec: 0, mistakes: 0 };
      a.n++;
      a.react += s.reactMs;
      a.exec += s.execMs / Math.max(s.moves, 1); // 한 수당 시간으로 맞춰 비교
      a.mistakes += s.mistakes > 0 ? 1 : 0;
      acc.set(name, a);
    }
  }
  return [...acc.entries()]
    .map(([name, a]) => {
      const reactMs = a.react / a.n;
      const execMs = a.exec / a.n;
      const mistakeRate = a.mistakes / a.n;
      return { name, n: a.n, reactMs, execMs, mistakeRate, score: reactMs / 1000 + (execMs / 1000) * 4 + mistakeRate * 3 };
    })
    .sort((x, y) => y.score - x.score);
}

/** 끝난 직후 말해 줄 한두 문장: 가장 크게 좋아진 단계, 또는 가장 느린 단계 */
export function feedbackText(cmp: StageCompare[]): string | null {
  const withAvg = cmp.filter((c) => c.avg !== null);
  if (!withAvg.length) return null;
  const best = withAvg.find((c) => c.best !== null && c.now < c.best!);
  if (best) return `${STAGE_NAME[best.stage]} 최고 기록이에요! ${secondsWord(best.now / 1000)}.`;
  const gains = withAvg.map((c) => ({ c, d: c.avg! - c.now })).sort((a, b) => b.d - a.d);
  if (gains[0].d > 1000) return `${josa.iGa(STAGE_NAME[gains[0].c.stage])} 평균보다 ${secondsWord(gains[0].d / 1000)} 빨랐어요.`;
  const worst = gains[gains.length - 1];
  if (worst.d < -1000) return `${josa.iGa(STAGE_NAME[worst.c.stage])} 평균보다 ${secondsWord(-worst.d / 1000)} 느렸어요. 다음엔 여기를 연습해 봐요.`;
  return "평소와 비슷한 속도예요.";
}
