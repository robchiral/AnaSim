import pyqtgraph as pg
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QGridLayout, QFrame)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
import numpy as np

from .styles import (
    COLORS,
    get_base_widget_style,
    get_tinted_frame_style,
    get_rgba,
)


class NumericDisplay(QFrame):
    """
    A unified widget for displaying a single vital sign numeric value.
    Handles styling, layout, and alarm states internally.
    """
    def __init__(self, label, unit="", color=COLORS['text'], initial_value="--", 
                 tooltip="", size_variant="normal", embedded=False):
        super().__init__()
        self.base_color = color
        self.label_text = label
        self.current_alarm_state = None
        self.embedded = embedded
        self.setObjectName("numericDisplay")
        
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(1)
        
        if size_variant == "small":
            self.layout.setContentsMargins(7, 5, 7, 6)
            self._val_size = "22px"
            self._lbl_size = "10px"
        elif size_variant == "compact":
            self.layout.setContentsMargins(8, 6, 8, 7)
            self._val_size = "26px"
            self._lbl_size = "10px"
        else:
            self.layout.setContentsMargins(10, 7, 10, 8)
            self._val_size = "42px"
            self._lbl_size = "11px"

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        self.lbl_title = QLabel(label)
        self.lbl_title.setStyleSheet(
            f"color: {color}; font-size: {self._lbl_size}; font-weight: 700;"
        )
        header.addWidget(self.lbl_title)
        header.addStretch()

        self.lbl_unit = QLabel(unit)
        self.lbl_unit.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-size: 9px; font-weight: 500;"
        )
        self.lbl_unit.setVisible(bool(unit))
        header.addWidget(self.lbl_unit, alignment=Qt.AlignRight | Qt.AlignVCenter)
        self.layout.addLayout(header)
        
        self.lbl_val = QLabel(initial_value)
        self.lbl_val.setStyleSheet(
            f"color: {color}; font-size: {self._val_size}; font-weight: 700;"
        )
        self.lbl_val.setAlignment(Qt.AlignRight)
        self.layout.addWidget(self.lbl_val)
            
        self._apply_base_style()
        
        if tooltip:
            self.setToolTip(tooltip)

    def _apply_base_style(self):
        if self.embedded:
            self.setStyleSheet(
                "QFrame#numericDisplay { background: transparent; border: none; }"
            )
        else:
            self.setStyleSheet(f"""
                QFrame#numericDisplay {{
                    background-color: transparent;
                    border: none;
                    border-bottom: 1px solid {COLORS['border']};
                }}
            """)
        self.lbl_title.setText(self.label_text)
        self.lbl_title.setStyleSheet(f"color: {self.base_color}; font-size: {self._lbl_size}; font-weight: 600;")

    def _apply_alarm_style(self, is_low):
        color = COLORS['danger'] if is_low else COLORS['warning']
        indicator = "low" if is_low else "high"
        
        self.setStyleSheet(f"""
            QFrame#numericDisplay {{
                background-color: {get_rgba(color, 0.12)};
                border: 1px solid {color};
                border-left: 4px solid {color};
                border-radius: 4px;
            }}
        """)
        self.lbl_title.setText(f"{self.label_text} {indicator}")
        self.lbl_title.setStyleSheet(f"color: {color}; font-size: {self._lbl_size}; font-weight: 700;")

    def set_value(self, text):
        self.lbl_val.setText(text)

    def set_alarm(self, active: bool, is_low: bool = False):
        new_state = ('low' if is_low else 'high') if active else None
        if self.current_alarm_state != new_state:
            self.current_alarm_state = new_state
            if active:
                self._apply_alarm_style(is_low)
            else:
                self._apply_base_style()




class PatientMonitorWidget(QWidget):
    """Real-time patient monitor showing waveforms (ECG, SpO2, Art, CO2) and numerics."""
    def __init__(self, tutorial_mode=False, arterial_line_enabled=True):
        super().__init__()
        self.tutorial_mode = tutorial_mode
        self.arterial_line_enabled = arterial_line_enabled
        self.setStyleSheet(get_base_widget_style())
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        # Fixed-screen sweep buffers. A moving gap separates the newest sample
        # from the previous sweep, while already-drawn samples remain stationary.
        self.buffer_size = 1000  # 10s at 100Hz
        self.ecg_data = np.full(self.buffer_size, np.nan)
        self.spo2_data = np.full(self.buffer_size, np.nan)
        self.art_data = np.full(self.buffer_size, np.nan)
        self.capno_data = np.full(self.buffer_size, np.nan)
        self.wave_write_index = 0
        self.sweep_gap_samples = 14
        self.last_plot_time = 0.0
        
        self.setup_ui()
        
    def setup_ui(self):
        # --- Top Status Bar ---
        header = QFrame()
        header.setObjectName("monitorHeader")
        header.setStyleSheet(
            f"QFrame#monitorHeader {{ background-color: {COLORS['header']}; "
            f"border-bottom: 1px solid {COLORS['border']}; }}"
        )
        header.setFixedHeight(40)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(14, 0, 14, 0)
        
        self.lbl_patient = QLabel("")
        self.lbl_patient.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 11px; font-weight: 600;"
        )
        header_layout.addWidget(self.lbl_patient)
        header_layout.addStretch()
        self.layout.addWidget(header)
        
        # --- Main Content ---
        content = QFrame()
        content.setObjectName("monitorContent")
        content.setStyleSheet(
            f"QFrame#monitorContent {{ background-color: {COLORS['background_alt']}; }}"
        )
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(6, 6, 0, 6)
        content_layout.setSpacing(6)
        self.layout.addWidget(content, stretch=1)
        
        # Left Column: Waveforms
        wave_frame = QFrame()
        wave_frame.setObjectName("waveformColumn")
        wave_frame.setStyleSheet(
            "QFrame#waveformColumn { background: transparent; border: none; }"
        )
        wave_layout = QVBoxLayout(wave_frame)
        wave_layout.setContentsMargins(0, 0, 0, 0)
        wave_layout.setSpacing(3)
        content_layout.addWidget(wave_frame, stretch=72)
        
        # Right Column: Numerics
        num_frame = QFrame()
        num_frame.setObjectName("numericColumn")
        num_frame.setStyleSheet(
            f"QFrame#numericColumn {{ background-color: {COLORS['panel']}; "
            f"border-left: 1px solid {COLORS['border']}; }}"
        )
        num_layout = QVBoxLayout(num_frame)
        num_layout.setContentsMargins(8, 4, 8, 6)
        num_layout.setSpacing(2)
        content_layout.addWidget(num_frame, stretch=28)

        # --- Plot Initializers ---
        self.ecg_plot, self.ecg_curve = self.create_plot(
            COLORS['ecg'], "ECG · II", y_range=(-0.5, 1.5)
        )
        self.spo2_plot, self.spo2_curve = self.create_plot(
            COLORS['spo2'], "Pleth", y_range=(-0.1, 1.4)
        )
        self.art_plot, self.art_curve = self.create_plot(
            COLORS['abp'], "ART", y_range=(0, 200), show_y_axis=True, y_ticks=[0, 50, 100, 150, 200]
        )
        self.capno_plot, self.capno_curve = self.create_plot(
            COLORS['co2'], "CO₂ · mmHg", y_range=(0, 60), show_y_axis=True, y_ticks=[0, 20, 40, 60]
        )

        wave_layout.addWidget(self.ecg_plot)
        wave_layout.addWidget(self.spo2_plot)
        wave_layout.addWidget(self.art_plot)
        wave_layout.addWidget(self.capno_plot)

        # --- Numeric Widgets ---
        
        # HR
        self.num_hr = NumericDisplay(
            "HR", "bpm", COLORS['ecg'], "60"
        )
        num_layout.addWidget(self.num_hr)
        
        # SpO2
        self.num_spo2 = NumericDisplay(
            "SpO₂", "%", COLORS['spo2'], "100"
        )
        num_layout.addWidget(self.num_spo2)
        
        # BP (ABP or NIBP)
        self.num_map = NumericDisplay(
            "ART", "mmHg", COLORS['abp'], "120/80 (93)"
        )
        self.num_nibp = NumericDisplay(
            "NIBP", "mmHg", COLORS['abp'], "--/-- (--)"
        )
        num_layout.addWidget(self.num_map)
        num_layout.addWidget(self.num_nibp)
        
        # EtCO2 & RR container
        co2_frame = QFrame()
        co2_frame.setObjectName("co2Panel")
        co2_frame.setStyleSheet(
            f"QFrame#co2Panel {{ background-color: transparent; "
            f"border-bottom: 1px solid {COLORS['border']}; }}"
        )
        co2_layout = QHBoxLayout(co2_frame)
        co2_layout.setContentsMargins(0, 0, 0, 0)
        co2_layout.setSpacing(2)
        
        self.num_etco2 = NumericDisplay(
            "EtCO₂", "mmHg", COLORS['co2'], "38", size_variant="normal", embedded=True
        )
        self.num_rr = NumericDisplay(
            "RR", "/min", COLORS['co2'], "12", size_variant="compact", embedded=True
        )

        co2_layout.addWidget(self.num_etco2, stretch=2)
        co2_layout.addWidget(self.num_rr, stretch=1)
        num_layout.addWidget(co2_frame)

        # Secondary monitoring row
        row_secondary = QHBoxLayout()
        row_secondary.setSpacing(2)
        self.num_bis = NumericDisplay("BIS", "", COLORS['bis'], "--", size_variant="compact")
        self.num_tof = NumericDisplay("TOF", "", COLORS['tof'], "--%", size_variant="compact")
        self.num_temp = NumericDisplay("Temp", "°C", COLORS['temp'], "37.0", size_variant="compact")
        row_secondary.addWidget(self.num_bis)
        row_secondary.addWidget(self.num_tof)
        row_secondary.addWidget(self.num_temp)
        num_layout.addLayout(row_secondary)

        # I/O panel
        self.io_panel = self._create_io_panel()
        num_layout.addWidget(self.io_panel)
        
        # Gas Panel (Sevo)
        self.gas_panel = self._create_gas_panel()
        num_layout.addWidget(self.gas_panel)
        
        self._apply_bp_mode()

    def _create_io_panel(self):
        frame = QFrame()
        frame.setObjectName("tintedPanel")
        frame.setStyleSheet(get_tinted_frame_style(COLORS['info'], alpha=0.05, radius=6))
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        header_layout = QHBoxLayout()
        header_layout.setSpacing(6)
        header_layout.addStretch()
        self.lbl_io_title = QLabel("Fluid balance")
        self.lbl_io_title.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-weight: 700; font-size: 10px;"
        )
        self.lbl_io_title.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        header_layout.addWidget(self.lbl_io_title)

        self.lbl_io_net = QLabel("--")
        self.lbl_io_net.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_io_net.setStyleSheet(
            f"color: {COLORS['info']}; font-size: 18px; font-weight: 700;"
        )
        header_layout.addWidget(self.lbl_io_net)
        layout.addLayout(header_layout)

        self.lbl_io_detail = QLabel("IV --  Blood --  |  Urine --  Loss --")
        self.lbl_io_detail.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_io_detail.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 11px;"
        )
        layout.addWidget(self.lbl_io_detail)

        return frame

    def _create_gas_panel(self):
        frame = QFrame()
        frame.setObjectName("gasPanel")
        frame.setStyleSheet(f"""
            QFrame#gasPanel {{
                background-color: {get_rgba(COLORS['gas'], 0.05)};
                border: 1px solid {get_rgba(COLORS['border'], 0.3)};
                border-radius: 6px;
            }}
        """)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(4)
        
        # Header
        h_layout = QHBoxLayout()
        lbl_gas = QLabel("Sevo")
        lbl_gas.setStyleSheet(f"color: {COLORS['gas']}; font-weight: 700; font-size: 11px;")
        self.lbl_mac = QLabel("MAC: 0.0")
        self.lbl_mac.setStyleSheet(f"color: {COLORS['gas']}; font-weight: 600; font-size: 12px;")
        h_layout.addWidget(lbl_gas)
        h_layout.addStretch()
        h_layout.addWidget(self.lbl_mac)
        layout.addLayout(h_layout)
        
        # Values
        vals_layout = QHBoxLayout()
        
        def make_col(label, val_label_attr):
            vbox = QVBoxLayout()
            vbox.setSpacing(0)
            l = QLabel(label)
            l.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 10px;")
            v = QLabel("0.0")
            v.setStyleSheet(f"color: {COLORS['gas']}; font-size: 20px; font-weight: 700;")
            setattr(self, val_label_attr, v)
            vbox.addWidget(l, alignment=Qt.AlignCenter)
            vbox.addWidget(v, alignment=Qt.AlignCenter)
            return vbox

        vals_layout.addLayout(make_col("Fi", "lbl_fi_val"))
        vals_layout.addStretch()
        vals_layout.addLayout(make_col("Et", "lbl_et_val"))
        
        layout.addLayout(vals_layout)
        return frame

    def update_patient_info(self, name="Simulated patient", age=40, gender="M", weight=70,
                            renal_status=None, hepatic_status=None):
        """Update the patient information label."""
        info = [name, f"{age:.0f} y", str(gender).capitalize(), f"{weight:.1f} kg"]
        if renal_status and str(renal_status).lower() != "normal":
            info.append(f"Renal: {renal_status}")
        if hepatic_status and str(hepatic_status).lower() != "normal":
            info.append(f"Hepatic: {hepatic_status}")
        self.lbl_patient.setText("  ·  ".join(info))

    def create_plot(self, color, title, y_range=None, show_y_axis=False, y_ticks=None):
        plot = pg.PlotWidget()
        plot.setBackground(COLORS['background'])
        plot.showGrid(x=False, y=show_y_axis, alpha=0.08)
        plot.setMouseEnabled(x=False, y=False)
        plot.hideAxis('bottom')
        plot.setXRange(0, self.buffer_size, padding=0)
        
        # Consistent Axis Width
        axis = plot.getAxis('left')
        axis.setWidth(35)
        
        if y_range:
            plot.setYRange(y_range[0], y_range[1], padding=0.03)

        if show_y_axis and y_range:
            axis.setStyle(showValues=True, tickLength=-4)
            axis.setTextPen(pg.mkPen(color=COLORS['text_dim']))
            axis.setPen(pg.mkPen(color=COLORS['border'], width=0))
            if y_ticks:
                axis.setTicks([[(v, str(int(v))) for v in y_ticks]])
        else:
            axis.setStyle(showValues=False, tickLength=0)
            axis.setPen(pg.mkPen(color=COLORS['background_alt']))  # Hide

        plot.setMinimumHeight(80)
        
        # Title as Item
        text = pg.TextItem(text=title, color=color, anchor=(0, 1))
        text.setFont(QFont("Arial", 9, QFont.Weight.DemiBold))
        plot.addItem(text)
        
        # Keep the channel label stable while the waveform buffer fills.
        title_y = 0.05 if not y_range else y_range[1] - (y_range[1] - y_range[0]) * 0.10
        text.setPos(10, title_y)
        
        # Antialiasing a trace that updates at 20 FPS creates visible edge
        # shimmer. The clinical monitor palette remains crisp without it.
        plot.setAntialiasing(False)
        plot.setClipToView(True)
        
        pen = pg.mkPen(color=color, width=1.6)
        curve = plot.plot(pen=pen)
        return plot, curve
    
    def _apply_bp_mode(self):
        self.art_plot.setVisible(self.arterial_line_enabled)
        self.num_map.setVisible(self.arterial_line_enabled)
        self.num_nibp.setVisible(not self.arterial_line_enabled)

    def update_numerics(self, state):
        display_hr = state.display_value("hr")
        display_spo2 = state.display_value("spo2")
        display_map = state.display_value("map")
        display_sbp = state.display_value("sbp")
        display_dbp = state.display_value("dbp")
        display_etco2 = state.display_value("etco2")
        display_bis = state.display_value("bis")

        self.num_hr.set_value(f"{int(display_hr)}")
        self.num_spo2.set_value(
            f"{int(display_spo2)}" if state.spo2_signal_valid else "--"
        )
        
        if self.arterial_line_enabled:
            self.num_map.set_value(f"{int(display_sbp)}/{int(display_dbp)} ({int(display_map)})")
        else:
            ts = state.nibp_timestamp
            is_cycling = state.nibp_is_cycling
            cuff = state.nibp_cuff_pressure
            
            if is_cycling:
                self.num_nibp.set_value(f"Cuff: {int(cuff)}")
            elif ts <= 0.0:
                self.num_nibp.set_value("--/-- (--)")
            else:
                self.num_nibp.set_value(
                    f"{int(state.nibp_sys)}/{int(state.nibp_dia)} ({int(state.nibp_map)})"
                )

        self.num_etco2.set_value(
            f"{int(display_etco2)}" if state.etco2_signal_valid else "--"
        )
        self.num_rr.set_value(f"{int(state.rr)}")
        
        self.num_bis.set_value(f"{int(display_bis)}")
        self.num_tof.set_value(f"{int(state.tof)}%")
        self.num_temp.set_value(f"{state.temp_c:.1f}")

        fluid_in = state.fluid_in_ml
        blood_in = state.blood_in_ml
        urine_out = state.urine_out_ml
        blood_out = state.blood_out_ml
        net = state.net_fluid_ml
        self.lbl_io_detail.setText(
            f"IV {fluid_in:.0f}  Blood {blood_in:.0f}  |  "
            f"Urine {urine_out:.0f}  Loss {blood_out:.0f}"
        )
        self.lbl_io_net.setText(f"{net:+.0f}")
        net_color = COLORS['danger'] if net < 0 else COLORS['success']
        self.lbl_io_net.setStyleSheet(
            f"color: {net_color}; font-size: 18px; font-weight: 700;"
        )
        
        # Gas
        self.lbl_fi_val.setText(f"{state.fi_sevo:.1f}")
        self.lbl_et_val.setText(f"{state.et_sevo:.1f}")
        self.lbl_mac.setText(f"MAC: {state.mac:.2f}")

    def update_waveforms(self, engine):
        buffer = engine.output_buffer
        if not buffer:
            return

        # Check if there's new data by comparing the last state's time
        latest_time = buffer[-1].time
        if latest_time <= self.last_plot_time:
            return  # No new data

        # Estimate how many new states based on time difference
        # Typical frame has ~5-10 new states at 20 FPS with 100 steps/sec
        # Only search the tail of the buffer for efficiency
        time_diff = latest_time - self.last_plot_time
        estimated_new = max(5, min(int(time_diff * 100) + 5, len(buffer)))

        # Search backwards from end, but only through estimated_new entries
        new_states = []
        search_start = max(0, len(buffer) - estimated_new)
        for i in range(len(buffer) - 1, search_start - 1, -1):
            s = buffer[i]
            if s.time <= self.last_plot_time:
                break
            new_states.append(s)

        if not new_states:
            return

        new_states.reverse()
        new_states = new_states[-self.buffer_size:]
        self.last_plot_time = latest_time

        # Extract columns
        ecg_c = np.array([s.ecg_voltage for s in new_states])
        spo2_c = np.array([s.pleth_voltage for s in new_states])
        capno_c = np.array([s.capno_co2 for s in new_states])

        count = len(ecg_c)
        start = self.wave_write_index
        self._write_sweep_chunk(self.ecg_data, ecg_c, start)
        self._write_sweep_chunk(self.spo2_data, spo2_c, start)
        self._write_sweep_chunk(self.capno_data, capno_c, start)

        if self.arterial_line_enabled:
            map_c = np.array([s.display_value("map") for s in new_states])
            # Synthetic art line: Pleth * 40 + (MAP - 13)
            art_c = spo2_c * 40 + (map_c - 13)
            self._write_sweep_chunk(self.art_data, art_c, start)

        self.wave_write_index = (start + count) % self.buffer_size
        gap = (
            self.wave_write_index + np.arange(self.sweep_gap_samples)
        ) % self.buffer_size
        for data in (self.ecg_data, self.spo2_data, self.art_data, self.capno_data):
            data[gap] = np.nan

        # Most of each trace is now unchanged between frames. `finite` prevents
        # pyqtgraph from bridging the sweep gap.
        self.ecg_curve.setData(self.ecg_data, connect="finite")
        self.spo2_curve.setData(self.spo2_data, connect="finite")
        self.capno_curve.setData(self.capno_data, connect="finite")
        if self.arterial_line_enabled:
            self.art_curve.setData(self.art_data, connect="finite")

    @staticmethod
    def _write_sweep_chunk(target, values, start):
        """Write a sample chunk into a circular display buffer."""
        count = len(values)
        first_count = min(count, len(target) - start)
        target[start:start + first_count] = values[:first_count]
        remaining = count - first_count
        if remaining:
            target[:remaining] = values[first_count:]

    def update_alarms(self, state):
        alarms = state.alarms
        
        # Handle NIBP auto-alarm logic if Art line is off
        if not self.arterial_line_enabled:
            # Check for NIBP MAP alarm if not provided
            if state.nibp_timestamp > 0:
                nmap = state.nibp_map
                if nmap < 60: alarms['MAP'] = {'low': True}
                elif nmap > 110: alarms['MAP'] = {'high': True}
        
        # Map keys to widgets
        mapping = {
            'HR': self.num_hr,
            'SpO2': self.num_spo2,
            'MAP': self.num_map if self.arterial_line_enabled else self.num_nibp,
            'EtCO2': self.num_etco2,
            'BIS': self.num_bis
        }
        
        for name, widget in mapping.items():
            if name in alarms:
                a_data = alarms[name]
                is_low = a_data.get('low', False)
                is_high = a_data.get('high', False)
                if is_low or is_high:
                    widget.set_alarm(True, is_low)
                    continue
            
            widget.set_alarm(False)
