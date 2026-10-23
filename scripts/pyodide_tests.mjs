// Run the test suite inside Pyodide, the runtime the web app uses.
//
//   npm install --no-save pyodide@$(python scripts/build_web.py --pyodide-version)
//   node scripts/pyodide_tests.mjs [--workers N] [pytest arguments]
//
// Each worker has its own Pyodide. Default to four workers and keep every n-th
// collected test. The first worker warms the package cache before others start.
//
// Local HTTP server tests run in native Python; browsers use the worker transport.
import { spawn } from "node:child_process";
import { availableParallelism } from "node:os";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const NO_TESTS_COLLECTED = 5;

const script = fileURLToPath(import.meta.url);
const root = resolve(dirname(script), "..");
const pytestArgs = [];
let workers = Math.min(availableParallelism(), 4);
const args = process.argv.slice(2);
for (let i = 0; i < args.length; i++) {
  const arg = args[i];
  if (arg === "--") {
    pytestArgs.push(...args.slice(i + 1));
    break;
  }
  if (arg === "--workers" || arg.startsWith("--workers=")) {
    const value = arg === "--workers" ? args[++i] : arg.slice("--workers=".length);
    workers = Number(value);
    if (!Number.isSafeInteger(workers) || workers < 1) {
      process.stderr.write("--workers requires a positive integer\n");
      process.exit(2);
    }
  } else {
    pytestArgs.push(arg);
  }
}
const shard = process.env.ANASIM_PYODIDE_SHARD;

process.exit(shard ? await runShard(shard.split("/").map(Number))
  : workers === 1 ? await runShard([0, 1]) : await runWorkers());

async function runWorkers() {
  const started = Date.now();
  let collectionReady;
  const ready = new Promise((done) => { collectionReady = done; });
  function startWorker(index, count) {
    return new Promise((done) => {
      const child = spawn(process.execPath, [script, ...pytestArgs], {
        env: { ...process.env, ANASIM_PYODIDE_SHARD: `${index}/${count}` },
        stdio: ["ignore", "pipe", "pipe", "ipc"],
      });
      child.on("message", ({ count }) => { if (index === 0) collectionReady(count); });
      let output = "";
      child.stdout.on("data", (chunk) => { output += chunk; });
      child.stderr.on("data", (chunk) => { output += chunk; });
      child.on("error", (error) => { output += `${error}\n`; });
      child.on("close", (code) => {
        // A startup or collection failure can exit before reporting a test count.
        if (index === 0) collectionReady(1);
        // Print each worker's report whole so parallel output does not interleave.
        process.stdout.write(`\n--- Pyodide worker ${index + 1} ---\n${output}`);
        done(code ?? 1);
      });
    });
  }
  const first = startWorker(0, workers);
  // Reuse the first runtime for tests and avoid empty workers for focused runs.
  const count = await ready;
  const codes = await Promise.all([first,
    ...Array.from({ length: count - 1 }, (_, index) => startWorker(index + 1, count)),
  ]);
  const seconds = ((Date.now() - started) / 1000).toFixed(0);
  // Preserve pytest's exit code when no tests match.
  const ran = codes.filter((code) => code !== NO_TESTS_COLLECTED);
  const code = ran.length ? Math.max(...ran) : NO_TESTS_COLLECTED;
  process.stdout.write(`\n${count} Pyodide workers finished in ${seconds} s; ${code === 0 ? "all passed" : `exit code ${code}`}\n`);
  return code;
}

async function loadTestRuntime() {
  const { loadPyodide } = await import("pyodide");
  const pyodide = await loadPyodide();
  await pyodide.loadPackage(["numpy", "scipy", "pytest"], { messageCallback: () => {} });
  return pyodide;
}

async function runShard([index, count]) {
  const pyodide = await loadTestRuntime();
  pyodide.FS.mkdirTree("/repo");
  pyodide.FS.mount(pyodide.FS.filesystems.NODEFS, { root }, "/repo");

  pyodide.globals.set("pytest_args", pyodide.toPy([
    "-p", "no:cacheprovider",
    "--ignore=tests/test_local.py",
    ...pytestArgs,
  ]));
  pyodide.globals.set("shard", pyodide.toPy([index, count]));
  pyodide.globals.set("worker_count", (total) => {
    const actual = Math.min(count, Math.max(total, 1));
    if (index === 0 && process.send) process.send({ count: actual });
    return actual;
  });
  return pyodide.runPython(`
import os, sys
sys.dont_write_bytecode = True
os.chdir("/repo")
sys.path.insert(0, "/repo")
print(f"Python {sys.version.split()[0]} on {sys.platform}")
import pytest


class Shard:
    """Keep every n-th collected test, so slow parametrized tests spread across workers."""

    def __init__(self, index, count):
        self.index, self.count = index, count

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, config, items):
        self.count = worker_count(len(items))
        deselected = [item for i, item in enumerate(items) if i % self.count != self.index]
        items[:] = items[self.index::self.count]
        config.hook.pytest_deselected(items=deselected)


int(pytest.main(list(pytest_args), plugins=[Shard(*shard)]))
`);
}
