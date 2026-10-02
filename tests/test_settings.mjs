// Exercise the editor state machine without launching a browser.
import assert from "node:assert/strict";
import test from "node:test";
import { SettingsEditor } from "../anasim/web_assets/settings.js";

class Field extends EventTarget {
  type = "number";
  value = "";
  disabled = false;
  reportValidity() { return this.value !== "" && Number(this.value) >= 0; }
  edit(value) {
    this.value = value;
    this.dispatchEvent(new Event("input"));
  }
  key(key) {
    const event = new Event("keydown", { cancelable: true });
    event.key = key;
    this.dispatchEvent(event);
  }
}

// Only the toolbar requires DOM construction; input events use native EventTarget.
globalThis.document = {
  createElement: () => ({ children: [{}, {}], remove() {} }),
};
const container = { append() {} };

test("drafts survive snapshots and blur; Escape restores the latest accepted settings", async () => {
  const rate = new Field();
  const peep = new Field();
  let commands = 0;
  const editor = new SettingsEditor(container, [rate, peep], async () => { commands++; return true; });
  editor.sync(rate, 12);
  editor.sync(peep, 5);
  rate.edit("16");
  rate.dispatchEvent(new Event("change"));
  rate.dispatchEvent(new Event("blur"));
  editor.sync(rate, 12);
  editor.sync(peep, 6);
  assert.equal(commands, 0);
  assert.equal(rate.value, "16");
  assert.equal(peep.value, "5");
  rate.key("Escape");
  assert.equal(rate.value, "12");
  assert.equal(peep.value, "6");
  assert.equal(editor.toolbar.hidden, true);
  rate.edit("");
  await editor.submit();
  assert.equal(commands, 0);
  rate.edit("18");
  rate.key("Enter");
  await Promise.resolve();
  assert.equal(commands, 1);
  assert.equal(editor.dirty, false);
  editor.destroy();
  rate.edit("20");
  rate.key("Enter");
  assert.equal(commands, 1, "replaced sessions must not retain input listeners");
});

test("pending Apply is single-flight and a rejected command retains the draft", async () => {
  const target = new Field();
  let resolve;
  let commands = 0;
  const editor = new SettingsEditor(container, [target], () => {
    commands++;
    return new Promise(done => { resolve = done; });
  });
  editor.sync(target, 3);
  target.edit("4");
  const pending = editor.submit();
  await editor.submit();
  editor.sync(target, 3);
  editor.cancel();
  assert.equal(commands, 1);
  assert.equal(target.value, "4");
  assert.equal(target.disabled, true);
  resolve(false);
  await pending;
  assert.equal(editor.dirty, true);
  assert.equal(target.disabled, false);
  const retry = editor.submit();
  editor.sync(target, 4);
  resolve(true);
  await retry;
  assert.equal(editor.toolbar.hidden, true);
  assert.equal(commands, 2);
  editor.destroy();
});
