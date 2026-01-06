"""
Centralized styles module for AnaSim UI.

This module provides a unified theme, colors, fonts, and style builders
to ensure consistency across all UI components.
"""

# =============================================================================
# COLORS - Unified color palette
# =============================================================================

COLORS = {
    # Core UI surfaces
    'background': '#0B0F14',
    'background_alt': '#0F141C',
    'panel': '#151B24',
    'card': '#1C2431',
    'header': '#10151D',

    # Borders & Dividers
    'border': '#2A3341',
    'border_light': '#364355',
    'border_focus': '#4C86F7',

    # Text
    'text': '#E7ECF4',
    'text_secondary': '#C1CAD8',
    'text_dim': '#7E8A9C',

    # Controls
    'control': '#1A2230',
    'control_hover': '#222C3A',
    'control_pressed': '#283246',

    # Accent Colors
    'primary': '#4C86F7',
    'success': '#2FB36D',
    'warning': '#E1A644',
    'danger': '#E26D5C',
    'info': '#4BA3C7',

    # Vital Signs (monitor)
    'ecg': '#35C679',
    'spo2': '#4BA3C7',
    'abp': '#D05757',
    'co2': '#D6A34D',
    'bis': '#C48754',
    'tof': '#A7ADB9',
    'temp': '#5A8CC6',
    'gas': '#9E7BC9',

    # Drug categories
    'drug_hypnotic': '#4C86F7',
    'drug_opioid': '#F2A53B',
    'drug_nmb': '#B07AE6',
    'drug_pressor': '#E35B5B',
    'drug_inotrope': '#E85C88',
    'drug_vasoconstrictor': '#F07A3B',
}

# =============================================================================
# FONTS
# =============================================================================

FONTS = {
    'family': 'Arial',
    'size_small': '11px',
    'size_normal': '12px',
    'size_medium': '13px',
    'size_large': '14px',
    'size_title': '16px',
    'size_display': '20px',
    'size_numeric': '34px',
    'size_numeric_compact': '24px',
    'size_numeric_small': '18px',
}

# =============================================================================
# STYLE BUILDERS - Functions to generate stylesheet strings
# =============================================================================

def get_base_widget_style():
    """Base style for all widgets."""
    return f"""
        QWidget {{
            background-color: {COLORS['background']};
            color: {COLORS['text']};
            font-family: {FONTS['family']};
            font-size: {FONTS['size_normal']};
        }}
        QLabel {{
            background-color: transparent;
            background: none;
            color: {COLORS['text']};
        }}
        QToolTip {{
            background-color: {COLORS['card']};
            color: {COLORS['text']};
            border: 1px solid {COLORS['border']};
            border-radius: 6px;
            padding: 6px 8px;
        }}
    """

def get_groupbox_style(accent_color=None):
    """Style for QGroupBox with optional left accent border."""
    accent = f"border-left: 4px solid {accent_color}; padding-left: 10px;" if accent_color else ""
    return f"""
        QGroupBox {{
            font-weight: 600;
            font-size: {FONTS['size_medium']};
            color: {COLORS['text']};
            border: 1px solid {COLORS['border']};
            border-radius: 10px;
            margin-top: 14px;
            padding: 10px 12px 12px 12px;
            background-color: {COLORS['card']};
            {accent}
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 12px;
            padding: 0 6px;
            background-color: {COLORS['card']};
            color: {COLORS['text_secondary']};
        }}
    """

def get_spinbox_style():
    """Style for QSpinBox and QDoubleSpinBox."""
    return f"""
        QSpinBox, QDoubleSpinBox {{
            background-color: {COLORS['control']};
            color: {COLORS['text']};
            border: 1px solid {COLORS['border']};
            border-radius: 6px;
            padding: 6px 10px;
            font-size: {FONTS['size_medium']};
            min-width: 80px;
        }}
        QSpinBox:focus, QDoubleSpinBox:focus {{
            border-color: {COLORS['primary']};
        }}
        QSpinBox:disabled, QDoubleSpinBox:disabled {{
            color: {COLORS['text_dim']};
            background-color: {COLORS['background_alt']};
        }}
    """

def get_combobox_style():
    """Style for QComboBox."""
    return f"""
        QComboBox {{
            background-color: {COLORS['control']};
            color: {COLORS['text']};
            border: 1px solid {COLORS['border']};
            border-radius: 6px;
            padding: 6px 10px;
            font-size: {FONTS['size_medium']};
            min-width: 100px;
        }}
        QComboBox:focus {{
            border-color: {COLORS['primary']};
        }}
        QComboBox::drop-down {{
            border: none;
            width: 24px;
        }}
        QComboBox QAbstractItemView {{
            background-color: {COLORS['panel']};
            color: {COLORS['text']};
            selection-background-color: {COLORS['primary']};
            border: 1px solid {COLORS['border']};
        }}
        QComboBox:disabled {{
            color: {COLORS['text_dim']};
            background-color: {COLORS['background_alt']};
        }}
    """

def get_label_style():
    """Style for QLabel."""
    return f"""
        QLabel {{
            color: {COLORS['text']};
            font-size: {FONTS['size_normal']};
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
    radius=8,
    min_width=None,
    font_size=None,
    font_weight=600,
):
    """Style for QPushButton with various variants."""
    variant_map = {
        "primary": COLORS['primary'],
        "success": COLORS['success'],
        "warning": COLORS['warning'],
        "danger": COLORS['danger'],
        "info": COLORS['info'],
        "neutral": COLORS['control'],
    }
    base = bg_color or variant_map.get(variant, COLORS['control'])
    is_neutral = base == COLORS['control']
    font_size = font_size or FONTS['size_medium']
    if text_color:
        text = text_color
    elif outlined and not is_neutral:
        text = base
    elif outlined and is_neutral:
        text = COLORS['text']
        base = COLORS['border_light']  # Use lighter color for border visibility
    else:
        text = COLORS['text'] if is_neutral or outlined else "white"

    if outlined:
        hover_bg = get_rgba(base, 0.12)
        pressed_bg = get_rgba(base, 0.2)
        border = f"1px solid {base}"
        background = "transparent"
    else:
        hover_bg = COLORS['control_hover'] if is_neutral else get_rgba(base, 0.9)
        pressed_bg = COLORS['control_pressed'] if is_neutral else get_rgba(base, 0.8)
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
            background-color: {COLORS['background_alt']};
            color: {COLORS['text_dim']};
            border-color: {COLORS['border']};
        }}
    """

def get_toggle_button_style(active_color, text_color=None, inactive_bg=None):
    """Style for toggle/checkable buttons."""
    text = text_color or COLORS['text']
    inactive = inactive_bg or COLORS['control']
    return f"""
        QPushButton {{
            background-color: {inactive};
            color: {text};
            padding: 8px 16px;
            border-radius: 8px;
            font-size: {FONTS['size_medium']};
            font-weight: 600;
            border: 1px solid {COLORS['border']};
        }}
        QPushButton:hover {{
            background-color: {COLORS['control_hover']};
            border-color: {active_color};
        }}
        QPushButton:checked {{
            background-color: {active_color};
            color: white;
            border-color: {active_color};
        }}
    """

def get_radiobutton_style(color=None, indicator_color=None):
    """Style for QRadioButton."""
    c = color or COLORS['text']
    ic = indicator_color or c
    return f"""
        QRadioButton {{
            color: {c};
            font-size: {FONTS['size_normal']};
            spacing: 8px;
            background-color: transparent;
            background: none;
        }}
        QRadioButton::indicator {{
            width: 16px;
            height: 16px;
            border-radius: 8px;
            border: 2px solid {COLORS['border']};
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
            border: 1px solid {COLORS['border']};
            border-radius: 10px;
            background-color: {COLORS['panel']};
        }}
        QTabBar::tab {{
            background-color: transparent;
            color: {COLORS['text_dim']};
            padding: 8px 16px;
            margin-right: 2px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            font-size: {FONTS['size_normal']};
            font-weight: 600;
            min-width: 70px;
            border: 1px solid transparent;
        }}
        QTabBar::tab:selected {{
            background-color: {COLORS['panel']};
            color: {COLORS['text']};
            border-color: {COLORS['border']};
            border-bottom: 2px solid {COLORS['primary']};
        }}
        QTabBar::tab:hover:!selected {{
            background-color: {COLORS['control_hover']};
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
            background-color: {COLORS['panel']};
            width: 10px;
            border-radius: 5px;
        }}
        QScrollBar::handle:vertical {{
            background-color: {COLORS['border']};
            border-radius: 5px;
            min-height: 30px;
        }}
        QScrollBar::handle:vertical:hover {{
            background-color: {COLORS['border_light']};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
    """

def get_frame_style(bg_color=None, border_color=None, radius=8, border_width=1):
    """Style for QFrame."""
    bg = bg_color or COLORS['panel']
    bc = border_color or COLORS['border']
    return f"""
        QFrame {{
            background-color: {bg};
            border: {border_width}px solid {bc};
            border-radius: {radius}px;
        }}
    """

def get_bar_style(border_edge="bottom"):
    """Style for top/bottom bars."""
    edge = "bottom" if border_edge == "bottom" else "top"
    return f"""
        QFrame {{
            background-color: {COLORS['header']};
            border-{edge}: 1px solid {COLORS['border']};
        }}
    """

def get_tinted_frame_style(color, alpha=0.06, radius=8):
    """Subtle tinted frame for numeric panels."""
    return f"""
        QFrame {{
            background-color: {get_rgba(color, alpha)};
            border: 1px solid {COLORS['border']};
            border-radius: {radius}px;
        }}
    """

def get_dialog_style():
    """Style for QDialog."""
    return f"""
        QDialog {{
            background-color: {COLORS['background']};
            color: {COLORS['text']};
            font-family: {FONTS['family']};
        }}
        {get_groupbox_style()}
        {get_spinbox_style()}
        {get_combobox_style()}
        {get_label_style()}
        {get_radiobutton_style(COLORS['text'], indicator_color=COLORS['primary'])}
        QCheckBox {{
            color: {COLORS['text']};
            font-size: {FONTS['size_normal']};
            spacing: 8px;
            background-color: transparent;
        }}
        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            border-radius: 4px;
            border: 1px solid {COLORS['border']};
            background-color: {COLORS['control']};
        }}
        QCheckBox::indicator:checked {{
            background-color: {COLORS['primary']};
            border-color: {COLORS['primary']};
        }}
        QCheckBox::indicator:disabled {{
            background-color: {COLORS['background_alt']};
            border-color: {COLORS['border']};
        }}
        QPushButton {{
            background-color: {COLORS['control']};
            color: {COLORS['text']};
            border: 1px solid {COLORS['border']};
            border-radius: 8px;
            padding: 8px 18px;
            font-size: {FONTS['size_medium']};
            font-weight: 600;
            min-width: 80px;
        }}
        QPushButton:hover {{
            background-color: {COLORS['control_hover']};
            border-color: {COLORS['border_light']};
        }}
        QPushButton:focus {{
            border-color: {COLORS['primary']};
        }}
    """

def get_overlay_style():
    """Style for tutorial/scenario overlay."""
    return f"""
        QFrame {{
            background-color: {COLORS['card']};
            border: 1px solid {COLORS['border']};
            border-radius: 12px;
        }}
        QLabel {{
            color: {COLORS['text']};
            background: transparent;
        }}
    """

def get_progressbar_style():
    """Style for QProgressBar."""
    return f"""
        QProgressBar {{
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
            background-color: {COLORS['control']};
            text-align: center;
            color: transparent;
        }}
        QProgressBar::chunk {{
            background-color: {COLORS['primary']};
            border-radius: 3px;
        }}
    """

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def hex_to_rgb(hex_color):
    """Convert hex color to r, g, b string for rgba()."""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
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
