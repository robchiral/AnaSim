// Explicit edits survive snapshots and focus changes until applied or cancelled.
export class SettingsEditor {
  constructor(container, fields, apply, onChange = () => {}) {
    this.fields = fields;
    this.apply = apply;
    this.onChange = onChange;
    this.values = new Map();
    this.dirty = false;
    this.pending = false;
    this.listeners = new AbortController();
    this.toolbar = document.createElement("div");
    this.toolbar.className = "setting-actions";
    this.toolbar.hidden = true;
    this.toolbar.innerHTML = `<button type="button" class="btn small">Apply</button>
      <button type="button" class="btn small outlined">Cancel</button>`;
    [this.applyButton, this.cancelButton] = this.toolbar.children;
    container.append(this.toolbar);
    this.applyButton.onclick = () => this.submit();
    this.cancelButton.onclick = () => this.cancel();
    for (const field of fields) {
      if (field.type === "number") {
        field.required = true;
        // Model-derived settings need not lie on the spinner's original step grid.
        field.step = "any";
      }
      field.addEventListener("input", () => {
        this.dirty = true;
        this.refresh();
      }, { signal: this.listeners.signal });
      field.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === "Escape") {
          event.preventDefault();
          if (event.key === "Enter") this.submit();
          else this.cancel();
        }
      }, { signal: this.listeners.signal });
    }
  }

  destroy() {
    this.listeners.abort();
    this.toolbar.remove();
  }

  sync(field, value) {
    const text = String(value);
    this.values.set(field, text);
    if (!this.dirty && !this.pending && field.value !== text) field.value = text;
  }

  refresh() {
    this.toolbar.hidden = !this.dirty && !this.pending;
    this.applyButton.disabled = this.pending;
    this.cancelButton.disabled = this.pending;
    this.onChange();
  }

  cancel() {
    if (this.pending) return;
    this.dirty = false;
    for (const [field, value] of this.values) field.value = value;
    this.refresh();
  }

  async submit() {
    if (!this.dirty || this.pending) return;
    for (const field of this.fields) {
      if (!field.disabled && !field.reportValidity()) return;
    }
    this.pending = true;
    const disabled = this.fields.map((field) => field.disabled);
    for (const field of this.fields) field.disabled = true;
    this.refresh();
    try {
      if (await this.apply()) this.dirty = false;
    } finally {
      this.pending = false;
      this.fields.forEach((field, index) => { field.disabled = disabled[index]; });
      this.refresh();
    }
  }
}
