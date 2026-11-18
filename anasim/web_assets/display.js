export const $ = (id) => document.getElementById(id);

// Replacing a button's text node mid-press makes Safari drop the click.
export function setText(element, text) {
  if (element.textContent !== text) element.textContent = text;
}

export function minutesSeconds(seconds) {
  const total = Math.max(0, Math.floor(seconds));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}
