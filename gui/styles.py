"""
Application Styles and Themes - Professional Dual-Mode UI
"""

class StyleSheet:
    """Application stylesheets with dynamic theme generation."""

    # Professional Color Palettes
    THEMES = {
        'dark': {
            'bg_main':    '#1e1e2e',      # Deep Blue-Grey
            'bg_surface': '#2b2b3d',      # Lighter Panel
            'bg_input':   '#181825',      # Dark Input
            'primary':    '#3a86ff',      # Bright Blue
            'secondary':  '#00b4d8',      # Cyan
            'accent':     '#8338ec',      # Purple
            'success':    '#00e676',      # Green
            'warning':    '#ffbe0b',      # Amber
            'danger':     '#ff5252',      # Red
            'text_main':  '#ffffff',
            'text_dim':   '#a6a6c0',
            'border':     '#3e3e52',
            'hover':      '#323246'
        },
        'light': {
            'bg_main':    '#f0f2f5',      # Light Grey-Blue
            'bg_surface': '#ffffff',      # White Panel
            'bg_input':   '#f8f9fa',      # Light Input
            'primary':    '#2563eb',      # Solid Blue
            'secondary':  '#0891b2',      # Cyan
            'accent':     '#7c3aed',      # Purple
            'success':    '#059669',      # Green
            'warning':    '#d97706',      # Orange
            'danger':     '#dc2626',      # Red
            'text_main':  '#1f2937',      # Dark Grey
            'text_dim':   '#6b7280',      # Dim Grey
            'border':     '#e5e7eb',
            'hover':      '#f3f4f6'
        }
    }

    current_theme = 'dark'

    @classmethod
    def get_color(cls, name):
        """Get a specific color from the current theme."""
        return cls.THEMES.get(cls.current_theme, cls.THEMES['dark']).get(name, '#ffffff')

    @classmethod
    def get_stylesheet(cls, theme='dark'):
        """Generate the full QSS stylesheet for the selected theme."""
        cls.current_theme = theme
        c = cls.THEMES.get(theme, cls.THEMES['dark'])

        return f"""
        QMainWindow, QWidget {{
            background-color: {c['bg_main']};
            color: {c['text_main']};
            font-family: 'Segoe UI', 'Roboto', sans-serif;
            font-size: 14px;
        }}
        
        /* --- Cards & Panels --- */
        QGroupBox {{
            background-color: {c['bg_surface']};
            border: 1px solid {c['border']};
            border-radius: 10px;
            margin-top: 1.5em;
            padding-top: 15px;
            font-weight: bold;
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 5px 10px;
            background-color: {c['bg_surface']};
            color: {c['primary']};
            border-radius: 6px;
        }}
        
        QFrame#StatCard {{
            background-color: {c['bg_surface']};
            border: 1px solid {c['border']};
            border-radius: 12px;
        }}
        QFrame#StatCard:hover {{
            border: 1px solid {c['primary']};
            background-color: {c['hover']};
        }}

        /* --- Buttons --- */
        QPushButton {{
            background-color: {c['primary']};
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            font-weight: 600;
        }}
        
        QPushButton:hover {{
            background-color: {c['secondary']};
        }}
        
        QPushButton:pressed {{
            background-color: {c['accent']};
        }}
        
        QPushButton:disabled {{
            background-color: {c['border']};
            color: {c['text_dim']};
        }}

        QPushButton#ActionBtn {{
            background-color: {c['bg_surface']};
            color: {c['text_main']};
            border: 1px solid {c['border']};
            text-align: left;
            padding: 15px;
        }}
        QPushButton#ActionBtn:hover {{
            background-color: {c['hover']};
            border-left: 4px solid {c['primary']};
        }}

        /* --- Inputs --- */
        QLineEdit, QSpinBox, QDoubleSpinBox {{
            background-color: {c['bg_input']};
            border: 1px solid {c['border']};
            border-radius: 6px;
            padding: 8px;
            color: {c['text_main']};
        }}
        
        QLineEdit:focus, QSpinBox:focus {{
            border: 1px solid {c['primary']};
        }}

        /* --- Tabs --- */
        QTabWidget::pane {{
            border: 1px solid {c['border']};
            border-radius: 6px;
            top: -1px;
        }}
        
        QTabBar::tab {{
            background-color: {c['bg_surface']};
            color: {c['text_dim']};
            padding: 10px 20px;
            margin-right: 4px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            font-weight: 600;
        }}
        
        QTabBar::tab:selected {{
            background-color: {c['primary']};
            color: white;
        }}

        /* --- Combo Box (Fixed Visibility) --- */
        QComboBox {{
            background-color: {c['bg_input']};
            border: 1px solid {c['border']};
            border-radius: 6px;
            padding: 6px;
            color: {c['text_main']};
            min-width: 100px;
        }}
        
        QComboBox::drop-down {{ border: none; }}
        QComboBox::down-arrow {{
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid {c['text_dim']};
            margin-right: 8px;
        }}

        /* CRITICAL: Styles the popup list to be visible */
        QComboBox QAbstractItemView {{
            background-color: {c['bg_surface']};
            color: {c['text_main']};
            border: 1px solid {c['border']};
            selection-background-color: {c['primary']};
            selection-color: white;
            outline: none;
        }}

        QTableWidget, QListWidget, QTextEdit {{
            background-color: {c['bg_surface']};
            border: 1px solid {c['border']};
            border-radius: 6px;
            color: {c['text_main']};
            gridline-color: {c['border']};
        }}
        
        QHeaderView::section {{
            background-color: {c['bg_input']};
            color: {c['text_dim']};
            padding: 8px;
            border: none;
            font-weight: bold;
            border-bottom: 2px solid {c['border']};
        }}

        /* --- Toggle Switch (CheckBox) --- */
        QCheckBox {{
            spacing: 8px;
            font-weight: bold;
            color: {c['text_main']};
        }}
        QCheckBox::indicator {{
            width: 36px;
            height: 20px;
            border-radius: 10px;
        }}
        QCheckBox::indicator:unchecked {{
            background-color: {c['border']};
            border: 2px solid {c['border']};
        }}
        QCheckBox::indicator:checked {{
            background-color: {c['primary']};
            border: 2px solid {c['primary']};
        }}
        
        /* --- Menus & Status --- */
        QMenuBar {{ background-color: {c['bg_surface']}; border-bottom: 1px solid {c['border']}; }}
        QMenuBar::item:selected {{ background-color: {c['primary']}; color: white; }}
        
        QMenu {{ background-color: {c['bg_surface']}; border: 1px solid {c['border']}; }}
        QMenu::item {{ padding: 6px 24px; color: {c['text_main']}; }}
        QMenu::item:selected {{ background-color: {c['primary']}; color: white; }}
        
        QStatusBar {{ background-color: {c['bg_surface']}; color: {c['text_dim']}; border-top: 1px solid {c['border']}; }}
        
        /* --- Tooltips --- */
        QToolTip {{
            background-color: {c['bg_surface']};
            color: {c['text_main']};
            border: 1px solid {c['primary']};
            padding: 4px;
        }}
        """