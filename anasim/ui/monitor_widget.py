import pyqtgraph as pg
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QGridLayout, QFrame)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor
import numpy as np

from .styles import (
    COLORS,
    get_bar_style,
    get_base_widget_style,
    get_frame_style,
    get_tinted_frame_style,
    hex_to_rgb,
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
        self.current_alarm_state = None  # None, 'low', 'high'
        self.embedded = embedded
        
        # Configure layout
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(0)
        
        # Style configuration based on variant
        if size_variant == "small":
            self.layout.setContentsMargins(6, 2, 6, 4)
            val_size = "18px"
            lbl_size = "11px"
        elif size_variant == "compact":
            self.layout.setContentsMargins(8, 4, 8, 6)
            val_size = "24px"
            lbl_size = "11px"
        else:  # normal
            self.layout.setContentsMargins(10, 6, 10, 8)
            val_size = "38px"
            lbl_size = "12px"

        # Label Component
        self.lbl_title = QLabel(label)
        self.lbl_title.setStyleSheet(f"color: {color}; font-size: {lbl_size}; font-weight: 600; font-family: Arial;")
        self.layout.addWidget(self.lbl_title, alignment=Qt.AlignRight)
        
        # Value Component
        self.lbl_val = QLabel(initial_value)
        self.lbl_val.setStyleSheet(f"color: {color}; font-size: {val_size}; font-weight: 700; font-family: Arial;")
        self.lbl_val.setAlignment(Qt.AlignRight)
        self.layout.addWidget(self.lbl_val)
        
        # Unit Component
        if unit:
            self.lbl_unit = QLabel(unit)
            self.lbl_unit.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px; font-family: Arial;")
            self.layout.addWidget(self.lbl_unit, alignment=Qt.AlignRight)
            
        # Initial Style
        self._apply_base_style()
        
        if tooltip:
            self.setToolTip(tooltip)

    def _apply_base_style(self):
        if self.embedded:
            self.setStyleSheet("background: transparent; border: none;")
        else:
            self.setStyleSheet(f"""
                QFrame {{
                    background-color: {self._get_rgba(self.base_color, 0.05)};
                    border: 1px solid {self._get_rgba(COLORS['border'], 0.5)};
                    border-radius: 6px;
                }}
            """)
        self.lbl_title.setText(self.label_text)
        self.lbl_title.setStyleSheet(f"color: {self.base_color}; font-size: {self._get_font_size(title=True)}; font-weight: 600; font-family: Arial;")

    def _apply_alarm_style(self, is_low):
        color = COLORS['danger'] if is_low else COLORS['warning']
        indicator = "LOW" if is_low else "HIGH"
        
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {self._get_rgba(color, 0.15)};
                border: 2px solid {color};
                border-radius: 6px;
            }}
        """)
        self.lbl_title.setText(f"{self.label_text} {indicator}")
        self.lbl_title.setStyleSheet(f"color: {color}; font-size: {self._get_font_size(title=True)}; font-weight: 700; font-family: Arial;")

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

    def _get_rgba(self, hex_color, alpha):
        hc = hex_color.lstrip('#')
        r, g, b = tuple(int(hc[i:i+2], 16) for i in (0, 2, 4))
        return f"rgba({r}, {g}, {b}, {alpha})"
    
    def _get_font_size(self, title=False):
        # Uses the init-time size variants embedded in the stylesheet.
        style = self.lbl_val.styleSheet()
        if "18px" in style: return "11px" if title else "18px"
        if "24px" in style: return "11px" if title else "24px"
        return "12px" if title else "38px"


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
        
        # Data Buffers
        self.buffer_size = 1000  # 10s at 100Hz
        self.ecg_data = np.zeros(self.buffer_size)
        self.spo2_data = np.zeros(self.buffer_size)
        self.art_data = np.zeros(self.buffer_size)
        self.capno_data = np.zeros(self.buffer_size)
        self.last_plot_time = 0.0
        
        self.setup_ui()
        
    def setup_ui(self):
        # --- Top Status Bar ---
        header = QFrame()
        header.setStyleSheet(f"background-color: {COLORS['header']}; border-bottom: 1px solid {COLORS['border']};")
        header.setFixedHeight(32)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 0, 12, 0)
        
        self.lbl_patient = QLabel("")
        self.lbl_patient.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px; font-weight: 500; font-family: Arial;")
        header_layout.addWidget(self.lbl_patient)
        
        header_layout.addStretch()
        
        lbl_brand = QLabel("AnaSim Monitor")
        lbl_brand.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px; font-family: Arial;")
        header_layout.addWidget(lbl_brand)
        self.layout.addWidget(header)
        
        # --- Main Content ---
        content = QFrame()
        content.setStyleSheet(f"background-color: {COLORS['background_alt']};")
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(4, 4, 4, 4)
        content_layout.setSpacing(4)
        self.layout.addWidget(content, stretch=1)
        
        # Left Column: Waveforms
        wave_frame = QFrame()
        wave_frame.setStyleSheet("background: transparent; border: none;")
        wave_layout = QVBoxLayout(wave_frame)
        wave_layout.setContentsMargins(0, 0, 0, 0)
        wave_layout.setSpacing(2)
        content_layout.addWidget(wave_frame, stretch=70)
        
        # Right Column: Numerics
        num_frame = QFrame()
        num_frame.setStyleSheet(f"background-color: {COLORS['panel']}; border-left: 1px solid {COLORS['border']};")
        num_layout = QVBoxLayout(num_frame)
        num_layout.setContentsMargins(6, 6, 6, 6)
        num_layout.setSpacing(6)
        content_layout.addWidget(num_frame, stretch=25)

        # --- Plot Initializers ---
        self.ecg_plot, self.ecg_curve = self.create_plot(COLORS['ecg'], "ECG  Lead II")
        self.spo2_plot, self.spo2_curve = self.create_plot(COLORS['spo2'], "SpO₂  Pleth")
        self.art_plot, self.art_curve = self.create_plot(
            COLORS['abp'], "ABP  Arterial", y_range=(0, 200), show_y_axis=True, y_ticks=[0, 50, 100, 150, 200]
        )
        self.capno_plot, self.capno_curve = self.create_plot(
            COLORS['co2'], "EtCO₂  mmHg", y_range=(0, 60), show_y_axis=True, y_ticks=[0, 20, 40, 60]
        )

        wave_layout.addWidget(self.ecg_plot)
        wave_layout.addWidget(self.spo2_plot)
        wave_layout.addWidget(self.art_plot)
        wave_layout.addWidget(self.capno_plot)

        # --- Numeric Widgets ---
        
        # HR
        self.num_hr = NumericDisplay(
            "Heart rate", "bpm", COLORS['ecg'], "60"
        )
        num_layout.addWidget(self.num_hr)
        
        # SpO2
        self.num_spo2 = NumericDisplay(
            "SpO₂", "%", COLORS['spo2'], "100"
        )
        num_layout.addWidget(self.num_spo2)
        
        # BP (ABP or NIBP)
        self.num_map = NumericDisplay(
            "ABP", "mmHg", COLORS['abp'], "120/80 (93)"
        )
        self.num_nibp = NumericDisplay(
            "NIBP", "mmHg", COLORS['abp'], "--/-- (--)"
        )
        num_layout.addWidget(self.num_map)
        num_layout.addWidget(self.num_nibp)
        
        # EtCO2 & RR container
        co2_frame = QFrame()
        co2_frame.setStyleSheet(f"background-color: {self._get_rgba(COLORS['co2'], 0.05)}; border-radius: 6px;")
        co2_layout = QVBoxLayout(co2_frame)
        co2_layout.setContentsMargins(2, 2, 2, 2)
        co2_layout.setSpacing(0)
        
        self.num_etco2 = NumericDisplay("EtCO₂", "mmHg", COLORS['co2'], "38", size_variant="normal", embedded=True)
        self.num_rr = NumericDisplay("Resp rate", "/min", COLORS['co2'], "12", size_variant="small", embedded=True)

        
        co2_layout.addWidget(self.num_etco2)
        co2_layout.addWidget(self.num_rr)
        num_layout.addWidget(co2_frame)

        # BIS / TOF Row
        row_bis_tof = QHBoxLayout()
        row_bis_tof.setSpacing(4)
        self.num_bis = NumericDisplay("BIS", "", COLORS['bis'], "--", size_variant="compact")
        self.num_tof = NumericDisplay("TOF", "", COLORS['tof'], "--%", size_variant="compact")
        row_bis_tof.addWidget(self.num_bis)
        row_bis_tof.addWidget(self.num_tof)
        num_layout.addLayout(row_bis_tof)
        
        # Temp
        self.num_temp = NumericDisplay("Core temp", "°C", COLORS['temp'], "37.0", size_variant="compact")
        num_layout.addWidget(self.num_temp)

        # I/O panel
        self.io_panel = self._create_io_panel()
        num_layout.addWidget(self.io_panel)
        
        # Gas Panel (Sevo)
        self.gas_panel = self._create_gas_panel()
        num_layout.addWidget(self.gas_panel)
        
        num_layout.addStretch()
        
        self._apply_bp_mode()

    def _create_io_panel(self):
        frame = QFrame()
        frame.setStyleSheet(get_tinted_frame_style(COLORS['info'], alpha=0.05, radius=6))
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        header_layout = QHBoxLayout()
        header_layout.setSpacing(6)
        header_layout.addStretch()
        self.lbl_io_title = QLabel("I/O (mL)")
        self.lbl_io_title.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-weight: 700; font-size: 10px; font-family: Arial;"
        )
        self.lbl_io_title.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        header_layout.addWidget(self.lbl_io_title)

        self.lbl_io_net = QLabel("--")
        self.lbl_io_net.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_io_net.setStyleSheet(
            f"color: {COLORS['info']}; font-size: 16px; font-weight: 700; font-family: Arial;"
        )
        header_layout.addWidget(self.lbl_io_net)
        layout.addLayout(header_layout)

        self.lbl_io_detail = QLabel("In F-- B-- | Out U-- B--")
        self.lbl_io_detail.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_io_detail.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 11px; font-family: Arial;"
        )
        layout.addWidget(self.lbl_io_detail)

        return frame

    def _create_gas_panel(self):
        frame = QFrame()
        frame.setStyleSheet(f"""
            QFrame {{
                background-color: {self._get_rgba(COLORS['gas'], 0.05)};
                border: 1px solid {self._get_rgba(COLORS['border'], 0.3)};
                border-radius: 6px;
            }}
        """)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(4)
        
        # Header
        h_layout = QHBoxLayout()
        lbl_gas = QLabel("SEVO")
        lbl_gas.setStyleSheet(f"color: {COLORS['gas']}; font-weight: 700; font-size: 11px; font-family: Arial;")
        self.lbl_mac = QLabel("MAC: 0.0")
        self.lbl_mac.setStyleSheet(f"color: {COLORS['gas']}; font-weight: 600; font-size: 12px; font-family: Arial;")
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
            l.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 10px; font-family: Arial;")
            v = QLabel("0.0")
            v.setStyleSheet(f"color: {COLORS['gas']}; font-size: 20px; font-weight: 700; font-family: Arial;")
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
        info = [f"Patient: {name}", f"{age}y {gender}", f"{weight}kg"]
        if renal_status and str(renal_status).lower() != "normal":
            info.append(f"Renal: {renal_status}")
        if hepatic_status and str(hepatic_status).lower() != "normal":
            info.append(f"Hepatic: {hepatic_status}")
        self.lbl_patient.setText("  •  ".join(info))

    def create_plot(self, color, title, y_range=None, show_y_axis=False, y_ticks=None):
        plot = pg.PlotWidget()
        plot.setBackground(COLORS['background_alt'])
        plot.showGrid(x=False, y=False)
        plot.setMouseEnabled(x=False, y=False)
        plot.hideAxis('bottom')
        
        # Consistent Axis Width
        axis = plot.getAxis('left')
        axis.setWidth(35)
        
        if show_y_axis and y_range:
            axis.setStyle(showValues=True, tickLength=-4)
            axis.setTextPen(pg.mkPen(color=COLORS['text_dim']))
            axis.setPen(pg.mkPen(color=COLORS['border'], width=0))
            if y_ticks:
                axis.setTicks([[(v, str(int(v))) for v in y_ticks]])
            plot.setYRange(y_range[0], y_range[1], padding=0.05)
        else:
            axis.setStyle(showValues=False, tickLength=0)
            axis.setPen(pg.mkPen(color=COLORS['background_alt']))  # Hide

        plot.setMinimumHeight(80)
        
        # Title as Item
        text = pg.TextItem(text=title, color=color, anchor=(0, 1))
        text.setFont(QFont('Arial', 9, QFont.Weight.Medium))
        plot.addItem(text)
        
        # Position title slightly inset
        text.setPos(5, 0.05 if not y_range else y_range[0] + (y_range[1]-y_range[0])*0.05)
        
        plot.setAntialiasing(True) 
        plot.setClipToView(True)
        
        pen = pg.mkPen(color=color, width=2.0)
        curve = plot.plot(pen=pen)
        return plot, curve
    
    def _apply_bp_mode(self):
        self.art_plot.setVisible(self.arterial_line_enabled)
        self.num_map.setVisible(self.arterial_line_enabled)
        self.num_nibp.setVisible(not self.arterial_line_enabled)

    def update_numerics(self, state):
        self.num_hr.set_value(f"{int(state.hr)}")
        self.num_spo2.set_value(f"{int(state.spo2)}")
        
        if self.arterial_line_enabled:
            sys = getattr(state, 'sbp', state.map + 27)
            dia = getattr(state, 'dbp', state.map - 13)
            self.num_map.set_value(f"{int(sys)}/{int(dia)} ({int(state.map)})")
        else:
            ts = getattr(state, 'nibp_timestamp', 0.0)
            is_cycling = getattr(state, 'nibp_is_cycling', False)
            cuff = getattr(state, 'nibp_cuff_pressure', 0.0)
            
            if is_cycling:
                self.num_nibp.set_value(f"Cuff: {int(cuff)}")
            elif ts <= 0.0:
                self.num_nibp.set_value("--/-- (--)")
            else:
                sys = getattr(state, 'nibp_sys', 0)
                dia = getattr(state, 'nibp_dia', 0)
                mean = getattr(state, 'nibp_map', 0)
                self.num_nibp.set_value(f"{int(sys)}/{int(dia)} ({int(mean)})")

        self.num_etco2.set_value(f"{int(state.etco2)}")
        self.num_rr.set_value(f"{int(state.rr)}")
        
        self.num_bis.set_value(f"{int(state.bis)}")
        self.num_tof.set_value(f"{int(state.tof)}%")
        self.num_temp.set_value(f"{state.temp_c:.1f}")

        fluid_in = getattr(state, 'fluid_in_ml', 0.0)
        blood_in = getattr(state, 'blood_in_ml', 0.0)
        urine_out = getattr(state, 'urine_out_ml', 0.0)
        blood_out = getattr(state, 'blood_out_ml', 0.0)
        net = getattr(state, 'net_fluid_ml', fluid_in + blood_in - urine_out - blood_out)
        self.lbl_io_detail.setText(
            f"In F{fluid_in:.0f} B{blood_in:.0f} | Out U{urine_out:.0f} B{blood_out:.0f}"
        )
        self.lbl_io_net.setText(f"{net:+.0f}")
        net_color = COLORS['danger'] if net < 0 else COLORS['success']
        self.lbl_io_net.setStyleSheet(
            f"color: {net_color}; font-size: 16px; font-weight: 700; font-family: Arial;"
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
        self.last_plot_time = latest_time

        count = len(new_states)
        
        # Extract columns
        ecg_c = np.array([s.ecg_voltage for s in new_states])
        spo2_c = np.array([s.pleth_voltage for s in new_states])
        capno_c = np.array([s.capno_co2 for s in new_states])
        
        # Shift & Append
        def shift_append(existing, new_chunk):
            c = len(new_chunk)
            existing[:-c] = existing[c:]
            existing[-c:] = new_chunk
            return existing

        self.ecg_data = shift_append(self.ecg_data, ecg_c)
        self.ecg_curve.setData(self.ecg_data)
        
        self.spo2_data = shift_append(self.spo2_data, spo2_c)
        self.spo2_curve.setData(self.spo2_data)
        
        self.capno_data = shift_append(self.capno_data, capno_c)
        self.capno_curve.setData(self.capno_data)
        
        if self.arterial_line_enabled:
            map_c = np.array([s.map for s in new_states])
            # Synthetic art line: Pleth * 40 + (MAP - 13)
            art_c = spo2_c * 40 + (map_c - 13)
            self.art_data = shift_append(self.art_data, art_c)
            self.art_curve.setData(self.art_data)

    def update_alarms(self, state):
        alarms = getattr(state, 'alarms', {}) or {}
        
        # Handle NIBP auto-alarm logic if Art line is off
        if not self.arterial_line_enabled:
            # Check for NIBP MAP alarm if not provided
            if getattr(state, 'nibp_timestamp', 0) > 0:
                nmap = getattr(state, 'nibp_map', 0)
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

    def _get_rgba(self, hex_color, alpha):
        hc = hex_color.lstrip('#')
        r, g, b = tuple(int(hc[i:i+2], 16) for i in (0, 2, 4))
        return f"rgba({r}, {g}, {b}, {alpha})"
