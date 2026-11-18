/**
 * 대본 뽑기: node test/transcript.ts [beginner|cfop4|cfop] [guided|hint] [시드] [실수확률] [몇 수마다 돌려 잡는지] [camera|keyboard] [처음|다시]
 *
 * main.ts 처럼 움직이는 가상 학생(test/student.ts 의 drive)이 푼다: 처음에 노랑 위·초록 앞으로 들고,
 * 잡아 달라고 하면 그렇게 잡고, 500ms마다 시간이 흐른다. 다섯 번째 값을 주면 중간중간 큐브를 통째로 돌려 잡는다
 * (기준 면이 바뀌는 장면). 마지막 값이 "처음"이면 기준 면 설명을 처음 듣는 사람으로 시작한다.
 */
import { drive } from "./student.ts";
import type { Method } from "../src/coach/method.ts";

const [method = "beginner", level = "guided", seed = "4", err = "0", regrip = "0", mode = "camera", first = "처음"] = process.argv.slice(2);
const lines: string[] = [];
const d = drive({
  seed: Number(seed),
  method: method as Method,
  level: level as "guided",
  errRate: Number(err),
  regripEvery: Number(regrip),
  mode: mode as "camera",
  taughtBasics: first !== "처음",
  log: (t, u) => lines.push(`${(t / 1000).toFixed(1).padStart(6)}s [${u.kind}] ${u.text}`),
});
console.log(lines.join("\n"));
if (d.problems.length) console.log(`\n문제 ${d.problems.length}개:\n${d.problems.join("\n")}`);
