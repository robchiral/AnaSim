"""Record the guided TIVA induction in the browser interface as the README GIF.

    pip install playwright pillow
    playwright install chromium
    python scripts/capture_demo.py
"""

import argparse
import io
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anasim.local import LocalServer  # noqa: E402

VIEWPORT = {"width": 1440, "height": 810}
SPEED = 30
# Pauses, in real seconds, so each click is visible before the next.
SETTLE_S = 0.4
NAVIGATE_S = 0.5
CONTINUE_S = 0.5
END_HOLD_S = 2.0

# A fading ring at each click, since screenshots do not show the pointer.
CLICK_MARKER = """
addEventListener("pointerdown", (event) => {
  const ring = document.createElement("div");
  Object.assign(ring.style, {
    position: "fixed", left: `${event.clientX - 18}px`, top: `${event.clientY - 18}px`,
    width: "36px", height: "36px", borderRadius: "50%", border: "3px solid #FFD54A",
    pointerEvents: "none", zIndex: 1000, transition: "opacity 0.6s", opacity: "1",
  });
  document.body.append(ring);
  setTimeout(() => { ring.style.opacity = "0"; }, 300);
  setTimeout(() => ring.remove(), 1000);
}, true);
"""


def enter(field, value):
    field.click()
    field.fill(str(value))
    field.blur()


def drug_card(page, name):
    summary = page.locator("summary").filter(has_text=re.compile(rf"^{re.escape(name)}(?:\s|$)"))
    card = page.locator(".drug-card").filter(has=summary)
    if card.get_attribute("open") is None:
        card.locator("summary").click()
    return card


def start_tci(page, name, target):
    card = drug_card(page, name)
    card.locator(".infusion-mode").select_option("tci")
    enter(card.locator(".target"), target)
    card.get_by_role("button", name="Apply", exact=True).click()


def give_bolus(page, name, amount):
    card = drug_card(page, name)
    enter(card.locator(".bolus-amount"), amount)
    card.locator(".bolus .btn").click()


def induce(page):
    start_tci(page, "Propofol", 4)
    give_bolus(page, "Propofol", 175)


def reduce_fresh_gas(page):
    # The objective opens Medications to check the infusions; the flow is on Machine.
    page.click('.tabs [data-tab="Machine"]')
    enter(page.locator("#c-o2"), 2)
    page.locator(".fresh-gas-settings").get_by_role("button", name="Apply", exact=True).click()


def preoxygenate(page):
    enter(page.locator("#c-o2"), 10)
    page.locator(".fresh-gas-settings").get_by_role("button", name="Apply", exact=True).click()


# Objectives not listed here complete by waiting.
ACTIONS = {
    "APPLY_MASK": lambda page: page.click('#c-airway [data-value="Mask"]'),
    "SET_FGF_PREOX": preoxygenate,
    "START_ANALGESIA": lambda page: start_tci(page, "Remifentanil", 4),
    "INDUCE": induce,
    "MASK_VENTILATE": lambda page: page.click("#c-bag"),
    "GIVE_NMB": lambda page: give_bolus(page, "Rocuronium", 50),
    "INTUBATE": lambda page: page.click('#c-airway [data-value="ETT"]'),
    "CONFIRM_ETT": lambda page: page.click("#c-vent-power"),
    "MAINTENANCE": reduce_fresh_gas,
}


def current_step(server):
    with server.lock:
        session = server.session
        if session.step_index >= len(session.scenario):
            return None
        return session.scenario[session.step_index]


def start_session(page, url):
    page.goto(url)
    page.check('input[name="session"][value="guided"]')
    page.select_option('select[name="scenario_id"]', "induction_tiva")
    page.click("#setup-start")
    page.wait_for_selector("#app", state="visible")
    enter(page.locator("#speed"), SPEED)
    page.click("#run")
    # Fill the waveform sweep so the first frame, shown while the GIF loads, is complete.
    page.wait_for_timeout(1000)


def record(server, page, fps, max_duration):
    """Complete each objective through the page, capturing frames on a fixed clock."""
    frames = []
    interval = 1.0 / fps
    started = next_frame = time.monotonic()
    step_id, phase, phase_start = None, "settle", started
    done_at = None
    while True:
        now = time.monotonic()
        if now >= next_frame:
            frames.append(Image.open(io.BytesIO(page.screenshot())).convert("RGB"))
            next_frame = max(next_frame + interval, now)
        if now - started > max_duration:
            step = current_step(server)
            status = page.text_content("#step-status")
            raise RuntimeError(f"Capture timed out at {step.id if step else 'the end'}: {status}")

        step = current_step(server)
        if step is None:
            done_at = done_at or now
            if now - done_at >= END_HOLD_S:
                return frames
        elif step.id != step_id:
            step_id, phase, phase_start = step.id, "settle", now
        elif phase == "settle" and now - phase_start >= SETTLE_S:
            if step.target_tab:
                page.click("#step-target")
            phase, phase_start = "navigate", now
        elif phase == "navigate" and now - phase_start >= NAVIGATE_S:
            if step.id in ACTIONS:
                ACTIONS[step.id](page)
            phase, phase_start = "wait", now
        elif phase == "wait" and page.is_enabled("#step-next"):
            phase, phase_start = "continue", now
        elif phase == "continue" and now - phase_start >= CONTINUE_S:
            page.click("#step-next")
            phase = "advanced"
        time.sleep(0.01)


def save_gif(frames, path, fps):
    width, height = frames[0].size
    # One palette for every frame avoids color flicker between frames.
    samples = frames[:: max(1, len(frames) // 12)][:12]
    sheet = Image.new("RGB", (width, height * len(samples)))
    for index, frame in enumerate(samples):
        sheet.paste(frame, (0, height * index))
    palette = sheet.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    indexed = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    indexed[0].save(
        path, save_all=True, append_images=indexed[1:], duration=round(1000 / fps), loop=0, optimize=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "images" / "anasim_demo.gif")
    parser.add_argument("--width", type=int, default=1200, help="GIF width in pixels")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--max-duration", type=float, default=60.0, help="Capture time limit in seconds")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as recordings, LocalServer(recordings_dir=recordings) as server:
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            # Render at the GIF size rather than resampling screenshots.
            page = browser.new_page(viewport=VIEWPORT, device_scale_factor=args.width / VIEWPORT["width"])
            page.add_init_script(CLICK_MARKER)
            start_session(page, server.url)
            frames = record(server, page, args.fps, args.max_duration)
            browser.close()
        server.shutdown()

    save_gif(frames, args.output, args.fps)
    size_mb = args.output.stat().st_size / 1e6
    print(f"Saved {args.output} ({len(frames)} frames, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
