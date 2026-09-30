// Run the test suite inside Pyodide, the runtime the web app uses.
//
//   npm install --no-save pyodide@$(python scripts/build_web.py --pyodide-version)
//   node scripts/pyodide_tests.mjs [pytest arguments]
//
// Local HTTP server tests run in native Python; browsers use the worker transport.
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { loadPyodide } from "pyodide";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);

const pyodide = await loadPyodide();
await pyodide.loadPackage(["numpy", "scipy", "pytest"], { messageCallback: () => {} });
pyodide.FS.mkdirTree("/repo");
pyodide.FS.mount(pyodide.FS.filesystems.NODEFS, { root }, "/repo");

pyodide.globals.set("pytest_args", pyodide.toPy([
  "-p", "no:cacheprovider",
  "--ignore=tests/test_local.py",
  ...args,
]));
const code = pyodide.runPython(`
import os, sys
sys.dont_write_bytecode = True
os.chdir("/repo")
sys.path.insert(0, "/repo")
print(f"Python {sys.version.split()[0]} on {sys.platform}")
import pytest
int(pytest.main(list(pytest_args)))
`);
process.exit(code);
