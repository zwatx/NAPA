// Records how-napa-works.html#video frame by frame through Chrome's DevTools protocol.
// Usage: node record.mjs <page.html> <out-dir>
import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const [page, outDir] = process.argv.slice(2);
mkdirSync(outDir, { recursive: true });
const port = 9333;
const chrome = spawn("google-chrome", [
  "--headless=new", "--disable-gpu", "--no-sandbox", `--remote-debugging-port=${port}`,
  "--window-size=832,900", "--force-device-scale-factor=1.5", "--hide-scrollbars",
  "--force-prefers-reduced-motion=false", "--user-data-dir=/tmp/napa-viz2/profile", "about:blank",
], { stdio: "ignore" });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let target;
for (let i = 0; i < 50 && !target; i++) {
  await sleep(200);
  try {
    target = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).find((t) => t.type === "page");
  } catch {}
}
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r) => (ws.onopen = r));
let id = 0;
const pending = new Map();
ws.onmessage = (m) => {
  const msg = JSON.parse(m.data);
  if (msg.id && pending.has(msg.id)) pending.get(msg.id)(msg), pending.delete(msg.id);
};
const send = (method, params = {}) =>
  new Promise((r) => {
    pending.set(++id, r);
    ws.send(JSON.stringify({ id, method, params }));
  });
const evaluate = async (expression) => {
  const r = await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
  if (r.result.exceptionDetails) throw new Error(JSON.stringify(r.result.exceptionDetails));
  return r.result.result.value;
};

await send("Page.navigate", { url: "file://" + resolve(page) + "#video" });
for (let i = 0; i < 100; i++) {
  await sleep(200);
  if (await evaluate("!!(window.__napa && document.getElementById('view').width)")) break;
}
const info = await evaluate(
  "(() => { const n = window.__napa; n.pause(); const c = document.getElementById('view');" +
    " return { starts: n.starts, durs: n.STAGES.map((s) => s.dur), w: c.width, h: c.height }; })()"
);
const total = info.starts.at(-1) + info.durs.at(-1);
const fps = 15, frames = Math.round(total * fps);
console.log(JSON.stringify({ ...info, total, frames }));
for (let f = 0; f < frames; f++) {
  const url = await evaluate(
    `(() => { window.__napa.seek(${(f / fps).toFixed(5)} + 0.0001); return document.getElementById('view').toDataURL('image/png'); })()`
  );
  writeFileSync(`${outDir}/${String(f).padStart(4, "0")}.png`, Buffer.from(url.split(",")[1], "base64"));
}
ws.close();
chrome.kill();
console.log("frames", frames);
