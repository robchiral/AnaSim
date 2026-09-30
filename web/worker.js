// Runs Pyodide and one anasim.web.WebSession off the main thread. Pyodide
// requires a module worker.
// In: create, cmd, close.
// Out: ready, load_error, created, create_error, tick, result, error.

const TICK_MS = 50;

let pyodide = null;
let webModule = null;
let session = null;
let timer = null;
let lastTick = 0;

const post = (message) => self.postMessage(message);

// Keep the last line of a Python traceback, without the exception type.
function describe(error) {
  const lines = String(error && error.message ? error.message : error).trim().split("\n");
  return lines[lines.length - 1].replace(/^\w+(Error|Exception): /, "");
}

async function load() {
  const build = await (await fetch("build.json", { cache: "no-cache" })).json();
  const { loadPyodide } = await import(`${build.pyodide}pyodide.mjs`);
  pyodide = await loadPyodide({ indexURL: build.pyodide });
  await pyodide.loadPackage(["numpy", "scipy"], { messageCallback: () => {} });
  const archive = await (await fetch(build.package)).arrayBuffer();
  await pyodide.unpackArchive(archive, "zip");
  webModule = pyodide.pyimport("anasim.web");
  post({ type: "ready", catalog: JSON.parse(webModule.catalog()) });
}

function closeSession() {
  clearTimeout(timer);
  timer = null;
  if (session) {
    session.destroy();
    session = null;
  }
}

// Post a snapshot; only running sessions need another timer tick.
function snapshot(realDt) {
  try {
    const snap = session.advance(realDt);
    post({ type: "tick", snap });
    return JSON.parse(snap).running;
  } catch (error) {
    closeSession();
    post({ type: "error", message: describe(error) });
    return false;
  }
}

function tick() {
  timer = null;
  const now = performance.now();
  const realDt = (now - lastTick) / 1000;
  lastTick = now;
  if (snapshot(realDt)) timer = setTimeout(tick, TICK_MS);
}

function create(params) {
  const pyParams = pyodide.toPy(params);
  let next;
  let info;
  try {
    next = webModule.WebSession(pyParams);
    info = JSON.parse(next.info());
  } catch (error) {
    next?.destroy();
    throw error;
  } finally {
    pyParams.destroy();
  }
  closeSession();
  session = next;
  post({ type: "created", info });
  lastTick = performance.now();
  tick();
}

self.onmessage = ({ data }) => {
  if (data.type === "create") {
    try {
      create(data.params);
    } catch (error) {
      post({ type: "create_error", message: describe(error) });
    }
  } else if (data.type === "cmd") {
    if (!session) {
      post({ type: "result", id: data.id, error: "The simulation has stopped. Start a new session to continue." });
      return;
    }
    try {
      const value = JSON.parse(session.command(data.name, JSON.stringify(data.args || {})));
      post({ type: "result", id: data.id, value });
    } catch (error) {
      post({ type: "result", id: data.id, error: describe(error) });
      return;
    }
    const running = snapshot(0);
    if (!running) {
      clearTimeout(timer);
      timer = null;
    } else if (timer === null) {
      lastTick = performance.now();
      timer = setTimeout(tick, TICK_MS);
    }
  } else if (data.type === "close") {
    closeSession();
  }
};

load().catch((error) => post({ type: "load_error", message: describe(error) }));
