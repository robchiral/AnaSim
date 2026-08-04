"""Record a short, deterministic AnaSim walkthrough as an animated GIF."""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtCore import QObject, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QApplication, QScrollArea

from capture_screenshots import ScreenshotMainWindow


DEMO_PARAMS = {
    "age": 25,
    "weight": 70.0,
    "height": 175.0,
    "sex": "male",
    "baseline_hb": 14.0,
    "renal_function": 1.0,
    "hepatic_function": 1.0,
    "renal_status": "Normal",
    "hepatic_status": "Normal",
    "mode": "awake",
    "maint_type": "tiva",
    "tutorial_mode": True,
    "scenario_id": "induction_tiva",
    "pk_model_propofol": "Eleveld",
    "pk_model_nore": "Li",
    "pk_model_epi": "Clutter",
    "bis_model": "Bouillon",
    "loc_model": "Kern",
    "enable_death_detector": False,
    "arterial_line_enabled": True,
}


class GifRecorder(QObject):
    """Capture scaled window frames at a fixed rate."""

    def __init__(self, window, size: QSize, fps: int):
        super().__init__(window)
        self.window = window
        self.size = size
        self.fps = fps
        self.frames = []
        self.timer = QTimer(self)
        self.timer.setInterval(round(1000 / fps))
        self.timer.timeout.connect(self.capture_frame)

    def start(self):
        self.capture_frame()
        self.timer.start()

    def stop(self):
        self.timer.stop()
        self.capture_frame()

    def capture_frame(self):
        image = self.window.grab().toImage().scaled(
            self.size,
            Qt.IgnoreAspectRatio,
            Qt.SmoothTransformation,
        )
        image = image.convertToFormat(QImage.Format_RGB888)

        painter = QPainter(image)
        pen = QPen(QColor("#666666"))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRect(0, 0, image.width() - 1, image.height() - 1)
        painter.end()

        frame = Image.frombuffer(
            "RGB",
            (image.width(), image.height()),
            bytes(image.bits()),
            "raw",
            "RGB",
            image.bytesPerLine(),
            1,
        ).copy()
        self.frames.append(frame)


class DemoDirector(QObject):
    """Drive the induction tutorial at a readable visual pace."""

    ACTIONS = {
        "APPLY_MASK": "apply_mask",
        "SET_FGF_PREOX": "set_preoxygenation_flow",
        "START_ANALGESIA": "start_remifentanil",
        "INDUCE": "induce_anesthesia",
        "MASK_VENTILATE": "start_bag_mask_ventilation",
        "GIVE_NMB": "give_rocuronium",
        "INTUBATE": "intubate",
    }

    def __init__(self, window, recorder, max_duration: float):
        super().__init__(window)
        self.window = window
        self.recorder = recorder
        self.max_duration = max_duration
        self.started_at = None
        self.step_id = None
        self.phase = "settle"
        self.phase_started_at = None
        self.completion_started_at = None
        self.error = None
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.tick)

    @property
    def overlay(self):
        return self.window.overlay

    @property
    def controls(self):
        return self.window.controls

    def start(self):
        self.started_at = time.monotonic()
        self.phase_started_at = self.started_at
        self.window.sb_speed.setValue(30.0)
        self.window.toggle_simulation()
        self.recorder.start()
        self.timer.start()

    def elapsed(self):
        return time.monotonic() - self.started_at

    def phase_elapsed(self, now):
        return now - self.phase_started_at

    def tick(self):
        now = time.monotonic()
        if self.elapsed() >= self.max_duration:
            if self.overlay.current_step < len(self.overlay.scenario):
                step = self.overlay.scenario[self.overlay.current_step]
                print(f"Incomplete step {step.id}: {self.overlay.check_requirements()[1]}")
                state = self.window.engine.state
                print(
                    "Capture state: "
                    f"airway={state.airway_mode.value}, vent={self.window.engine.vent.is_on}, "
                    f"rr={state.rr:.1f}, etco2={state.etco2:.1f}, "
                    f"signal={state.etco2_signal_valid}"
                )
            self.finish("maximum duration reached", failed=True)
            return

        self.overlay.update_state()
        if self.overlay.current_step >= len(self.overlay.scenario):
            if self.completion_started_at is None:
                self.completion_started_at = now
            elif now - self.completion_started_at >= 1.5:
                self.finish("scenario complete")
            return

        current = self.overlay.scenario[self.overlay.current_step]
        if current.id != self.step_id:
            self.step_id = current.id
            print(f"Demo step: {current.id}")
            self.phase = "settle"
            self.phase_started_at = now

        if self.phase == "settle" and self.phase_elapsed(now) >= 0.35:
            if current.target_tab is not None:
                self.overlay.btn_target.click()
            self.phase = "navigate"
            self.phase_started_at = now
            return

        if self.phase == "navigate" and self.phase_elapsed(now) >= 0.35:
            action_name = self.ACTIONS.get(current.id)
            if action_name is not None:
                getattr(self, action_name)()
            self.phase = "wait"
            self.phase_started_at = now
            return

        if self.phase == "wait" and self.overlay.requirements_met:
            self.phase = "complete"
            self.phase_started_at = now
            return

        if self.phase == "complete" and self.phase_elapsed(now) >= 0.4:
            self.overlay.btn_next.click()

    def show_control(self, widget):
        scroll = self.controls.tabs.currentWidget().findChild(QScrollArea)
        if scroll is not None:
            scroll.ensureWidgetVisible(widget, 12, 12)
        widget.setFocus()

    def apply_mask(self):
        self.show_control(self.controls.rb_mask)
        self.controls.rb_mask.click()

    def set_preoxygenation_flow(self):
        self.show_control(self.controls.sb_o2)
        self.controls.sb_air.setValue(0.0)
        self.controls.sb_n2o.setValue(0.0)
        self.controls.sb_o2.setValue(10.0)

    def start_remifentanil(self):
        widgets = self.controls.drug_widgets["remi"]
        self.show_control(widgets["target"])
        widgets["rb_tci"].click()
        widgets["target"].setValue(4.0)

    def induce_anesthesia(self):
        widgets = self.controls.drug_widgets["propofol"]
        self.show_control(widgets["bolus"])
        widgets["rb_tci"].click()
        widgets["target"].setValue(4.0)
        widgets["bolus"].setValue(175.0)
        self.window.engine.give_drug_bolus("propofol", 175.0)

    def start_bag_mask_ventilation(self):
        self.show_control(self.controls.btn_bag_mask)
        self.controls.btn_bag_mask.click()

    def give_rocuronium(self):
        widgets = self.controls.drug_widgets["roc"]
        self.show_control(widgets["bolus"])
        widgets["bolus"].setValue(50.0)
        self.window.engine.give_drug_bolus("roc", 50.0)

    def intubate(self):
        self.show_control(self.controls.rb_ett)
        self.controls.rb_ett.click()
        self.controls.sb_rr.setValue(12)
        self.controls.sb_tv.setValue(500)
        self.controls.sb_peep.setValue(5)
        self.controls.btn_vent_power.click()

    def finish(self, reason, failed=False):
        print(f"Stopping capture: {reason}")
        if failed:
            self.error = reason
        self.timer.stop()
        if self.window.engine.running:
            self.window.toggle_simulation()
        self.recorder.stop()
        QApplication.instance().quit()


def build_palette(frames):
    """Build one palette from representative frames to avoid GIF flicker."""
    sample_count = min(12, len(frames))
    sample_indexes = [
        round(index * (len(frames) - 1) / max(1, sample_count - 1))
        for index in range(sample_count)
    ]
    sample_width = 240
    sample_height = round(sample_width * frames[0].height / frames[0].width)
    sheet = Image.new("RGB", (sample_width * 4, sample_height * 3))
    for position, frame_index in enumerate(sample_indexes):
        sample = frames[frame_index].resize(
            (sample_width, sample_height),
            Image.Resampling.BILINEAR,
        )
        sheet.paste(
            sample,
            ((position % 4) * sample_width, (position // 4) * sample_height),
        )
    return sheet.quantize(colors=192, method=Image.Quantize.MAXCOVERAGE)


def save_gif(frames, output_path: Path, fps: int):
    if not frames:
        raise RuntimeError("The recorder did not capture any frames")

    palette = build_palette(frames)
    indexed_frames = [
        frame.quantize(palette=palette, dither=Image.Dither.NONE)
        for frame in frames
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    indexed_frames[0].save(
        output_path,
        save_all=True,
        append_images=indexed_frames[1:],
        duration=round(1000 / fps),
        loop=0,
        optimize=True,
        disposal=1,
    )


def record_demo(output_path: Path, width: int, height: int, fps: int, max_duration: float):
    app = QApplication(sys.argv)
    window = ScreenshotMainWindow(sim_params=DEMO_PARAMS)
    window.resize(1800, 900)
    window.show()

    recorder = GifRecorder(window, QSize(width, height), fps)
    director = DemoDirector(window, recorder, max_duration)
    QTimer.singleShot(500, director.start)
    app.exec()

    if director.error is not None:
        raise RuntimeError(f"Demo capture failed: {director.error}")

    print(f"Encoding {len(recorder.frames)} frames...")
    save_gif(recorder.frames, output_path, fps)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved {output_path} ({size_mb:.1f} MB)")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/images/anasim_demo.gif"),
        help="Output GIF path",
    )
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument(
        "--max-duration",
        type=float,
        default=22.0,
        help="Maximum real-time capture duration in seconds",
    )
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0:
        parser.error("width and height must be positive")
    if args.fps <= 0:
        parser.error("fps must be positive")
    if args.max_duration <= 0:
        parser.error("max-duration must be positive")
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    record_demo(
        cli_args.output,
        cli_args.width,
        cli_args.height,
        cli_args.fps,
        cli_args.max_duration,
    )
