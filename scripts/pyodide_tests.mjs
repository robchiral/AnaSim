// Run the test suite inside Pyodide, the runtime the web app uses.
//
//   npm install --no-save pyodide@$(python scripts/build_web.py --pyodide-version)
//   node scripts/pyodide_tests.mjs [pytest arguments]
//
// Pyodide is single-threaded, so this starts one Node process per core, up to
// MAX_WORKERS, each with its own Pyodide and every n-th collected test.
//
// Local HTTP server tests run in native Python; browsers use the worker transport.
import { spawn } from "node:child_process";
import { availableParallelism } from "node:os";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const MAX_WORKERS = 8; // Each worker holds its own Pyodide heap.
const NO_TESTS_COLLECTED = 5;

const script = fileURLToPath(import.meta.url);
const root = resolve(dirname(script), "..");
const pytestArgs = process.argv.slice(2);
const shard = process.env.ANASIM_PYODIDE_SHARD;

process.exit(shard ? await runShard(shard.split("/").map(Number)) : await runWorkers());

async function runWorkers() {
  const count = Math.min(availableParallelism(), MAX_WORKERS);
  const started = Date.now();
  const codes = await Promise.all(Array.from({ length: count }, (_, index) => new Promise((done) => {
    const child = spawn(process.execPath, [script, ...pytestArgs], {
      env: { ...process.env, ANASIM_PYODIDE_SHARD: `${index}/${count}` },
    });
    let output = "";
    child.stdout.on("data", (chunk) => { output += chunk; });
    child.stderr.on("data", (chunk) => { output += chunk; });
    child.on("close", (code) => {
      // Print each worker's report whole so parallel output does not interleave.
      process.stdout.write(`\n--- Pyodide worker ${index + 1}/${count} ---\n${output}`);
      done(code ?? 1);
    });
  })));
  const seconds = ((Date.now() - started) / 1000).toFixed(0);
  // A worker can be left without tests when arguments select only a few.
  const ran = codes.filter((code) => code !== NO_TESTS_COLLECTED);
  const code = ran.length ? Math.max(...ran) : NO_TESTS_COLLECTED;
  process.stdout.write(`\n${count} Pyodide workers finished in ${seconds} s; ${code === 0 ? "all passed" : `exit code ${code}`}\n`);
  return code;
}

async function runShard([index, count]) {
  const { loadPyodide } = await import("pyodide");
  const pyodide = await loadPyodide();
  await pyodide.loadPackage(["numpy", "scipy", "pytest"], { messageCallback: () => {} });
  pyodide.FS.mkdirTree("/repo");
  pyodide.FS.mount(pyodide.FS.filesystems.NODEFS, { root }, "/repo");

  pyodide.globals.set("pytest_args", pyodide.toPy([
    "-p", "no:cacheprovider",
    "--ignore=tests/test_local.py",
    ...pytestArgs,
  ]));
  pyodide.globals.set("shard", pyodide.toPy([index, count]));
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

    def pytest_collection_modifyitems(self, config, items):
        deselected = [item for i, item in enumerate(items) if i % self.count != self.index]
        items[:] = items[self.index::self.count]
        config.hook.pytest_deselected(items=deselected)


int(pytest.main(list(pytest_args), plugins=[Shard(*shard)]))
`);
}
