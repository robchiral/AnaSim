"""
Centralized styles module for AnaSim UI.

This module provides a unified theme, colors, fonts, and style builders
to ensure consistency across all UI components.
"""

COLORS = {
    # Core UI surfaces
    "background": "#080C11",
    "background_alt": "#0B1016",
    "panel": "#101720",
    "card": "#151E29",
    "header": "#0D131A",
    # Borders and dividers
    "border": "#26313E",
    "border_light": "#39495B",
    "border_focus": "#5B94F8",
    # Text
    "text": "#F0F4F8",
    "text_secondary": "#B8C3D0",
    "text_dim": "#748296",
    # Controls
    "control": "#182230",
    "control_hover": "#202D3D",
    "control_pressed": "#263548",
    # Accent colors
    "primary": "#5B94F8",
    "success": "#35C77A",
    "warning": "#E7AE4A",
    "danger": "#F06A66",
    "info": "#54B5D8",
    # Vital signs
    "ecg": "#37D786",
    "spo2": "#51BCE5",
    "abp": "#F06464",
    "co2": "#E9B44C",
    "bis": "#D7A16B",
    "tof": "#B7BFCC",
    "temp": "#6DA7EA",
    "gas": "#B58BDD",
}

FONTS = {
    "family": "Arial",
    "size_small": "11px",
    "size_normal": "12px",
    "size_medium": "13px",
    "size_large": "14px",
    "size_title": "16px",
    "size_display": "20px",
    "size_numeric": "34px",
    "size_numeric_compact": "24px",
    "size_numeric_small": "18px",
}


def get_base_widget_style():
    """Base style for all widgets."""
    return f"""
        QWidget {{
            background-color: {COLORS["background"]};
            color: {COLORS["text"]};
            font-family: {FONTS["family"]};
            font-size: {FONTS["size_normal"]};
        }}
        QLabel {{
            background-color: transparent;
            background: none;
            color: {COLORS["text"]};
        }}
        QToolTip {{
            background-color: {COLORS["card"]};
            color: {COLORS["text"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 6px;
            padding: 6px 8px;
        }}
    """


def get_groupbox_style(accent_color=None):
    """Style for QGroupBox with optional left accent border."""
    accent = f"border-left: 3px solid {accent_color};" if accent_color else ""
    return f"""
        QGroupBox {{
            font-weight: 600;
            font-size: {FONTS["size_normal"]};
            color: {COLORS["text"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 7px;
            margin-top: 14px;
            padding: 12px;
            background-color: {COLORS["card"]};
            {accent}
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            padding: 0 5px;
            background-color: {COLORS["card"]};
            color: {COLORS["text_secondary"]};
        }}
    """


def get_spinbox_style():
    """Style for QSpinBox and QDoubleSpinBox."""
    return f"""
        QSpinBox, QDoubleSpinBox {{
            background-color: {COLORS["control"]};
            color: {COLORS["text"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 5px;
            padding: 7px 10px;
            font-size: {FONTS["size_medium"]};
            min-width: 80px;
            selection-background-color: {COLORS["primary"]};
        }}
        QSpinBox:hover, QDoubleSpinBox:hover {{
            border-color: {COLORS["border_light"]};
        }}
        QSpinBox:focus, QDoubleSpinBox:focus {{
            border-color: {COLORS["primary"]};
        }}
        QSpinBox:disabled, QDoubleSpinBox:disabled {{
            color: {COLORS["text_dim"]};
            background-color: {COLORS["background_alt"]};
        }}
    """


def get_combobox_style():
    """Style for QComboBox."""
    return f"""
        QComboBox {{
            background-color: {COLORS["control"]};
            color: {COLORS["text"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 5px;
            padding: 7px 10px;
            font-size: {FONTS["size_medium"]};
            min-width: 100px;
        }}
        QComboBox:focus {{
            border-color: {COLORS["primary"]};
        }}
        QComboBox:hover {{
            border-color: {COLORS["border_light"]};
        }}
        QComboBox::drop-down {{
            border: none;
            width: 24px;
        }}
        QComboBox QAbstractItemView {{
            background-color: {COLORS["panel"]};
            color: {COLORS["text"]};
            selection-background-color: {COLORS["primary"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 5px;
            outline: none;
            padding: 4px;
        }}
        QComboBox:disabled {{
            color: {COLORS["text_dim"]};
            background-color: {COLORS["background_alt"]};
        }}
    """


def get_label_style():
    """Style for QLabel."""
    return f"""
        QLabel {{
            color: {COLORS["text"]};
            font-size: {FONTS["size_normal"]};
            background-color: transparent;
            background: none;
            padding: 0;
            margin: 0;
        }}
    """


def get_button_style(
    bg_color=None,
    text_color=None,
    outlined=False,
    variant="neutral",
    padding="8px 16px",
    radius=6,
    min_width=None,
    font_size=None,
    font_weight=600,
):
    """Style for QPushButton with various variants."""
    variant_map = {
        "primary": COLORS["primary"],
        "success": COLORS["success"],
        "warning": COLORS["warning"],
        "danger": COLORS["danger"],
        "info": COLORS["info"],
        "neutral": COLORS["control"],
    }
    base = bg_color or variant_map.get(variant, COLORS["control"])
    is_neutral = base == COLORS["control"]
    font_size = font_size or FONTS["size_medium"]
    if text_color:
        text = text_color
    elif outlined and not is_neutral:
        text = base
    elif outlined and is_neutral:
        text = COLORS["text"]
        base = COLORS["border_light"]  # Use lighter color for border visibility
    else:
        text = COLORS["text"] if is_neutral or outlined else "white"

    if outlined:
        hover_bg = get_rgba(base, 0.12)
        pressed_bg = get_rgba(base, 0.2)
        border = f"1px solid {base}"
        background = "transparent"
    else:
        hover_bg = COLORS["control_hover"] if is_neutral else get_rgba(base, 0.9)
        pressed_bg = COLORS["control_pressed"] if is_neutral else get_rgba(base, 0.8)
        border = "1px solid transparent"
        background = base

    min_width_rule = f"min-width: {min_width}px;" if min_width else ""

    return f"""
        QPushButton {{
            background-color: {background};
            color: {text};
            padding: {padding};
            border-radius: {radius}px;
            font-size: {font_size};
            font-weight: {font_weight};
            border: {border};
            {min_width_rule}
        }}
        QPushButton:hover {{
            background-color: {hover_bg};
        }}
        QPushButton:pressed {{
            background-color: {pressed_bg};
        }}
        QPushButton:disabled {{
            background-color: {COLORS["background_alt"]};
            color: {COLORS["text_dim"]};
            border-color: {COLORS["border"]};
        }}
    """


def get_toggle_button_style(active_color, text_color=None, inactive_bg=None):
    """Style for toggle/checkable buttons."""
    text = text_color or COLORS["text"]
    inactive = inactive_bg or COLORS["control"]
    return f"""
        QPushButton {{
            background-color: {inactive};
            color: {text};
            padding: 8px 16px;
            border-radius: 6px;
            font-size: {FONTS["size_medium"]};
            font-weight: 600;
            border: 1px solid {COLORS["border"]};
        }}
        QPushButton:hover {{
            background-color: {COLORS["control_hover"]};
            border-color: {active_color};
        }}
        QPushButton:checked {{
            background-color: {active_color};
            color: white;
            border-color: {get_rgba(active_color, 0.7)};
            border-width: 1px;
        }}
        QPushButton:checked:hover {{
            background-color: {get_rgba(active_color, 0.85)};
        }}
    """


def get_segment_button_style(active_color=COLORS["primary"], compact=False):
    """Style an exclusive checkable button as a clear equipment-state segment."""
    padding = "6px 10px" if compact else "8px 12px"
    font_size = FONTS["size_small"] if compact else FONTS["size_normal"]
    return f"""
        QPushButton {{
            background-color: {COLORS["control"]};
            color: {COLORS["text_secondary"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 5px;
            padding: {padding};
            font-size: {font_size};
            font-weight: 600;
        }}
        QPushButton:hover {{
            color: {COLORS["text"]};
            border-color: {COLORS["border_light"]};
            background-color: {COLORS["control_hover"]};
        }}
        QPushButton:checked {{
            color: {active_color};
            border-color: {active_color};
            background-color: {get_rgba(active_color, 0.14)};
        }}
        QPushButton:disabled {{
            color: {COLORS["text_dim"]};
            background-color: {COLORS["background_alt"]};
            border-color: {COLORS["border"]};
        }}
    """


def get_section_group_style():
    """Flat section treatment for dense control panels."""
    return f"""
        QGroupBox {{
            color: {COLORS["text_secondary"]};
            background-color: transparent;
            border: none;
            border-top: 1px solid {COLORS["border"]};
            border-radius: 0;
            margin-top: 18px;
            padding: 14px 2px 4px 2px;
            font-size: {FONTS["size_normal"]};
            font-weight: 650;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 0;
            padding: 0 8px 0 0;
            background-color: {COLORS["panel"]};
            color: {COLORS["text_secondary"]};
        }}
    """


def get_drug_card_style():
    """Compact medication card with neutral separation."""
    return f"""
        QGroupBox {{
            color: {COLORS["text"]};
            background-color: {COLORS["card"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 6px;
            margin-top: 12px;
            padding: 10px;
            font-size: {FONTS["size_normal"]};
            font-weight: 650;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            padding: 0 5px;
            background-color: {COLORS["card"]};
            color: {COLORS["text"]};
        }}
    """


def get_radiobutton_style(color=None, indicator_color=None):
    """Style for QRadioButton."""
    c = color or COLORS["text"]
    ic = indicator_color or c
    return f"""
        QRadioButton {{
            color: {c};
            font-size: {FONTS["size_normal"]};
            spacing: 8px;
            background-color: transparent;
            background: none;
        }}
        QRadioButton::indicator {{
            width: 16px;
            height: 16px;
            border-radius: 8px;
            border: 2px solid {COLORS["border"]};
            background-color: transparent;
        }}
        QRadioButton::indicator:checked {{
            background-color: {ic};
            border-color: {ic};
        }}
    """


def get_tab_widget_style():
    """Style for QTabWidget."""
    return f"""
        QTabWidget::pane {{
            border: none;
            border-top: 1px solid {COLORS["border"]};
            background-color: {COLORS["panel"]};
        }}
        QTabWidget {{
            background-color: {COLORS["panel"]};
        }}
        QTabBar::tab {{
            background-color: transparent;
            color: {COLORS["text_dim"]};
            padding: 12px 18px 10px 18px;
            margin-right: 4px;
            font-size: {FONTS["size_normal"]};
            font-weight: 600;
            min-width: 86px;
            border: none;
            border-bottom: 2px solid transparent;
        }}
        QTabBar::tab:selected {{
            color: {COLORS["text"]};
            border-bottom-color: {COLORS["primary"]};
        }}
        QTabBar::tab:hover:!selected {{
            color: {COLORS["text_secondary"]};
        }}
    """


def get_scrollarea_style():
    """Style for QScrollArea."""
    return f"""
        QScrollArea {{
            border: none;
            background-color: transparent;
        }}
        QScrollBar:vertical {{
            background-color: transparent;
            width: 8px;
        }}
        QScrollBar::handle:vertical {{
            background-color: {COLORS["border_light"]};
            border-radius: 4px;
            min-height: 30px;
        }}
        QScrollBar::handle:vertical:hover {{
            background-color: {COLORS["border_light"]};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        QScrollBar:horizontal {{
            background-color: transparent;
            height: 8px;
        }}
        QScrollBar::handle:horizontal {{
            background-color: {COLORS["border"]};
            border-radius: 5px;
            min-width: 30px;
        }}
        QScrollBar::handle:horizontal:hover {{
            background-color: {COLORS["border_light"]};
        }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0px;
        }}
    """


def get_frame_style(bg_color=None, border_color=None, radius=8, border_width=1):
    """Style for QFrame."""
    bg = bg_color or COLORS["panel"]
    bc = border_color or COLORS["border"]
    return f"""
        QFrame#styledSurface {{
            background-color: {bg};
            border: {border_width}px solid {bc};
            border-radius: {radius}px;
        }}
    """


def get_bar_style(border_edge="bottom"):
    """Style for top/bottom bars."""
    edge = "bottom" if border_edge == "bottom" else "top"
    return f"""
        QFrame#controlBar {{
            background-color: {COLORS["header"]};
            border-{edge}: 1px solid {COLORS["border"]};
        }}
    """


def get_tinted_frame_style(color, alpha=0.06, radius=8):
    """Subtle tinted frame for numeric panels."""
    return f"""
        QFrame#tintedPanel {{
            background-color: {get_rgba(color, alpha)};
            border: 1px solid {COLORS["border"]};
            border-radius: {radius}px;
        }}
    """


def get_checkbox_style():
    """Style for QCheckBox."""
    return f"""
        QCheckBox {{
            color: {COLORS["text"]};
            font-size: {FONTS["size_normal"]};
            spacing: 8px;
            background-color: transparent;
        }}
        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            border-radius: 4px;
            border: 1px solid {COLORS["border"]};
            background-color: {COLORS["control"]};
        }}
        QCheckBox::indicator:hover {{
            border-color: {COLORS["border_light"]};
        }}
        QCheckBox::indicator:checked {{
            background-color: {COLORS["primary"]};
            border-color: {COLORS["primary"]};
        }}
        QCheckBox::indicator:disabled {{
            background-color: {COLORS["background_alt"]};
            border-color: {COLORS["border"]};
        }}
    """


def get_status_label_style(color):
    """Style for simulation status indicator labels."""
    return f"color: {color}; font-size: {FONTS['size_small']}; font-weight: 600;"


def get_dialog_style():
    """Style for QDialog."""
    return f"""
        QDialog {{
            background-color: {COLORS["background"]};
            color: {COLORS["text"]};
            font-family: {FONTS["family"]};
        }}
        {get_groupbox_style()}
        {get_spinbox_style()}
        {get_combobox_style()}
        {get_label_style()}
        {get_radiobutton_style(COLORS["text"], indicator_color=COLORS["primary"])}
        {get_checkbox_style()}
        QPushButton {{
            background-color: {COLORS["control"]};
            color: {COLORS["text"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 6px;
            padding: 8px 18px;
            font-size: {FONTS["size_medium"]};
            font-weight: 600;
            min-width: 80px;
        }}
        QPushButton:hover {{
            background-color: {COLORS["control_hover"]};
            border-color: {COLORS["border_light"]};
        }}
        QPushButton:focus {{
            border-color: {COLORS["primary"]};
        }}
    """


def get_overlay_style():
    """Style for tutorial/scenario overlay."""
    return f"""
        QFrame#scenarioOverlay {{
            background-color: {COLORS["panel"]};
            border: none;
            border-bottom: 1px solid {COLORS["border"]};
        }}
        QLabel {{
            color: {COLORS["text"]};
            background: transparent;
        }}
    """


def get_progressbar_style():
    """Style for QProgressBar."""
    return f"""
        QProgressBar {{
            border: 1px solid {COLORS["border"]};
            border-radius: 4px;
            background-color: {COLORS["control"]};
            text-align: center;
            color: transparent;
        }}
        QProgressBar::chunk {{
            background-color: {COLORS["primary"]};
            border-radius: 3px;
        }}
    """


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def hex_to_rgb(hex_color):
    """Convert hex color to r, g, b string for rgba()."""
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"{r}, {g}, {b}"


def get_rgba(hex_color, alpha):
    """Get rgba string from hex color and alpha value (0-1)."""
    return f"rgba({hex_to_rgb(hex_color)}, {alpha})"


# =============================================================================
# PRE-BUILT STYLE CONSTANTS for common use
# =============================================================================

STYLE_GROUPBOX = get_groupbox_style()
STYLE_SPINBOX = get_spinbox_style()
STYLE_COMBOBOX = get_combobox_style()
STYLE_LABEL = get_label_style()
STYLE_TAB_WIDGET = get_tab_widget_style()
STYLE_SCROLLAREA = get_scrollarea_style()
STYLE_PROGRESSBAR = get_progressbar_style()
STYLE_CHECKBOX = get_checkbox_style()
