// Both runtimes use the same command/result and snapshot messages.
export async function connect(onmessage) {
  const response = await fetch("runtime.json", { cache: "no-store" });
  if (!response.ok) throw new Error("Could not load the simulation runtime.");
  const { backend } = await response.json();
  if (backend === "pyodide") {
    const worker = new Worker("worker.js", { type: "module" });
    worker.onmessage = onmessage;
    worker.onerror = () => onmessage({ data: { type: "error", message: "The simulation worker stopped. Reload to restart." } });
    return worker;
  }
  if (backend !== "native") throw new Error(`Unknown runtime: ${backend}`);
  const transport = new NativeTransport(onmessage);
  // Let the caller store the transport before delivering any recovered session.
  queueMicrotask(() => transport.postMessage({ type: "init" }));
  return transport;
}

class NativeTransport {
  constructor(onmessage) {
    this.onmessage = onmessage;
    this.client = crypto.randomUUID();
    this.queue = Promise.resolve();
    this.timer = null;
    this.failed = false;
    window.addEventListener("pagehide", () => {
      clearTimeout(this.timer);
      fetch("api", {
        method: "POST", headers: { "Content-Type": "application/json" }, keepalive: true,
        body: JSON.stringify({ type: "detach", client: this.client }),
      }).catch(() => {});
    });
    window.addEventListener("pageshow", (event) => {
      if (event.persisted) location.reload();
    });
  }

  postMessage(message) {
    // Commands and polls stay ordered, with at most one request in flight.
    this.queue = this.queue.then(() => this.request(message));
  }

  async request(message) {
    if (this.failed) {
      if (message.type === "cmd") this.onmessage({ data: {
        type: "result", id: message.id, error: "Connection closed. Reload to reconnect.",
      } });
      return;
    }
    clearTimeout(this.timer);
    try {
      const response = await fetch("api", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...message, client: this.client }),
      });
      const messages = await response.json();
      if (!response.ok) throw new Error(messages.error);
      let stopped = false;
      for (const data of messages) {
        this.onmessage({ data });
        if (data.type === "error") stopped = true;
      }
      if (!stopped) this.timer = setTimeout(() => this.postMessage({ type: "poll" }), 50);
    } catch (error) {
      this.failed = true;
      const text = `${error.message}. Reload to reconnect. Local recordings remain on disk.`;
      if (message.type === "cmd") this.onmessage({ data: { type: "result", id: message.id, error: text } });
      this.onmessage({ data: { type: message.type === "init" ? "load_error" : "error", message: text } });
    }
  }
}
