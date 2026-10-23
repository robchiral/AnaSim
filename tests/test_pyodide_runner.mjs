import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

// Exercise process and cache coordination without loading several WASM heaps.
function run(t, args, { total = 3, failure = -1, startupFailure = false } = {}) {
  const root = mkdtempSync(join(tmpdir(), "anasim-pyodide-runner-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  mkdirSync(join(root, "scripts"));
  mkdirSync(join(root, "node_modules/pyodide"), { recursive: true });
  copyFileSync(new URL("../scripts/pyodide_tests.mjs", import.meta.url), join(root, "scripts/runner.mjs"));
  writeFileSync(join(root, "node_modules/pyodide/package.json"), JSON.stringify({
    type: "module", exports: "./index.mjs",
  }));
  const log = join(root, "runtimes.jsonl");
  writeFileSync(join(root, "node_modules/pyodide/index.mjs"), `
    import { appendFileSync, existsSync, writeFileSync } from "node:fs";
    export async function loadPyodide() {
      const shard = process.env.ANASIM_PYODIDE_SHARD || "0/1";
      appendFileSync(${JSON.stringify(log)}, JSON.stringify({ shard }) + "\\n");
      if (${startupFailure}) throw new Error("runtime startup failed");
      const [index] = shard.split("/").map(Number);
      const cache = ${JSON.stringify(join(root, "cache-ready"))};
      if (index && !existsSync(cache)) throw new Error("package cache is not ready");
      const globals = new Map();
      return {
        async loadPackage() { writeFileSync(cache, "ready"); },
        FS: { mkdirTree() {}, mount() {}, filesystems: { NODEFS: {} } },
        globals, toPy(value) { return value; },
        runPython() {
          const count = globals.get("worker_count")(${total});
          const selected = Array.from({ length: ${total} }, (_, i) => i).filter(i => i % count === index);
          console.log("selected tests:", selected.join(","));
          return selected.includes(${failure}) ? 1 : selected.length ? 0 : 5;
        },
      };
    }
  `);
  const result = spawnSync(process.execPath, [join(root, "scripts/runner.mjs"), ...args], {
    encoding: "utf8", timeout: 15000,
    env: { ...process.env, ANASIM_PYODIDE_SHARD: "" },
  });
  assert.ifError(result.error);
  const runtimes = existsSync(log) ? readFileSync(log, "utf8").trim().split("\n").map(JSON.parse) : [];
  return { ...result, runtimes };
}

test("workers reuse the first runtime and wait for its package cache", (t) => {
  const result = run(t, ["--workers", "2", "-q"]);
  assert.equal(result.status, 0, result.stderr);
  assert.equal(result.runtimes.length, 2);
  assert.match(result.stdout, /selected tests: 0,2/);
  assert.match(result.stdout, /selected tests: 1\n/);
});

test("focused and empty selections do not launch extra runtimes", (t) => {
  for (const total of [0, 1]) {
    const result = run(t, ["--workers=4", "-q"], { total });
    assert.equal(result.status, total ? 0 : 5, result.stderr);
    assert.equal(result.runtimes.length, 1);
  }
});

test("single worker mode uses the current process", (t) => {
  const result = run(t, ["--workers", "1", "-q"]);
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(result.runtimes, [{ shard: "0/1" }]);
  assert.match(result.stdout, /selected tests: 0,1,2/);
  assert.doesNotMatch(result.stdout, /Pyodide worker/);
});

test("worker failures and startup failures fail the run", (t) => {
  const failed = run(t, ["--workers", "2"], { failure: 1 });
  assert.equal(failed.status, 1);
  const startup = run(t, ["--workers", "2"], { startupFailure: true });
  assert.equal(startup.status, 1);
  assert.equal(startup.runtimes.length, 1);
});

test("invalid worker counts fail before loading Pyodide", (t) => {
  for (const args of [["--workers"], ["--workers=0"], ["--workers", "1.5"], ["--workers", "auto"]]) {
    const result = run(t, args);
    assert.equal(result.status, 2);
    assert.match(result.stderr, /positive integer/);
    assert.equal(result.runtimes.length, 0);
  }
});
