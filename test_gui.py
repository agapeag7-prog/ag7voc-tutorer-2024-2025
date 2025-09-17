import sys
import random
import io
import threading
import os
import time
import psutil
import numpy as np
import json
import speech_recognition as sr
import pyttsx3
from collections import deque
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QLabel, QPushButton, 
                             QFrame, QScrollArea, QListWidget, QListWidgetItem,
                             QProgressBar, QSplitter, QSizePolicy, QMessageBox,
                             QTabWidget, QComboBox, QSlider, QSpinBox, QLineEdit, 
                             QGroupBox, QCheckBox, QTextEdit, QStackedWidget,
                             QToolButton, QMenu, QAction, QSystemTrayIcon, QStyle,
                             QDialog, QDialogButtonBox)
from PyQt5.QtCore import Qt, QTimer, QSize, QPropertyAnimation, QEasingCurve, pyqtSignal, QObject, QPoint
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon, QLinearGradient, QPainter, QPainterPath, QPixmap, QBrush, QTextCursor, QPen

from ai_engine import dqn_agent

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class AssistantSignals(QObject):
    show_suggestions = pyqtSignal(list, str)
    update_display = pyqtSignal(str)
    update_status = pyqtSignal(str, str, str)
    add_history_item = pyqtSignal(str, str)
    update_metrics = pyqtSignal(dict)

assistant_signals = AssistantSignals()

class VoicePreferences:
    def __init__(self):
        self.preferences = {
            "wake_words": ["assistant", "réveille", "reveille", "hey assistant"],
            "sleep_words": ["dors", "veille", "arrête", "stop", "silence"],
            "voice_speed": 180,
            "voice_volume": 1.0
        }
    
    def get_preference(self, key):
        return self.preferences.get(key, [])

voice_prefs = VoicePreferences()

def process_voice_command(command, forced_intent=None):
    print(f"Traitement de la commande: {command}")
    if forced_intent:
        print(f"Intention forcée: {forced_intent}")
    
    if "heure" in command.lower():
        current_time = time.strftime("%H:%M:%S")
        return f"Il est {current_time}"
    elif "date" in command.lower():
        current_date = time.strftime("%d/%m/%Y")
        return f"Nous sommes le {current_date}"
    elif "ouvre" in command.lower() and "explorateur" in command.lower():
        os.system("explorer .")
        return "J'ouvre l'explorateur de fichiers"
    elif "éteins" in command.lower() or "arrête" in command.lower():
        return "Fermeture de l'application"
    else:
        return f"J'ai entendu: {command}"

def get_top_intents_spacy_similarity(command):
    intents = [
        ("time_query", random.uniform(0.7, 0.9)),
        ("system_info", random.uniform(0.6, 0.8)),
        ("file_operation", random.uniform(0.5, 0.7))
    ]
    return intents

def listen(timeout=5):
    """Fonction d'écoute du microphone"""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        try:
            print("Écoute en cours...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=5)
            text = recognizer.recognize_google(audio, language="fr-FR")
            print(f"Reconnu: {text}")
            return text
        except sr.UnknownValueError:
            print("Je n'ai pas compris")
            return None
        except sr.RequestError as e:
            print(f"Erreur service reconnaissance vocale: {e}")
            return None
        except sr.WaitTimeoutError:
            print("Temps d'écoute dépassé")
            return None

def speak(text):
    """Fonction de synthèse vocale"""
    engine = pyttsx3.init()
    engine.setProperty('rate', 180)
    engine.setProperty('volume', 1.0)
    engine.say(text)
    engine.runAndWait()

def set_front_display_callback(callback):
    """Configure le callback d'affichage"""
    global display_callback
    display_callback = callback

def get_system_info():
    """Récupère les informations système"""
    return {
        'os_name': os.name,
        'version': sys.version,
        'ram': {
            'total': f"{psutil.virtual_memory().total / (1024**3):.1f} Go",
            'percentage': f"{psutil.virtual_memory().percent}%"
        },
        'hostname': os.environ.get('COMPUTERNAME', 'Unknown'),
        'ip_address': '127.0.0.1',
        'cpu_usage': f"{psutil.cpu_percent()}%",
        'process_count': len(psutil.pids())
    }

COMMAND_HISTORY = []
INTENT_LABELS_FR = {
    "time_query": "Demande d'heure",
    "system_info": "Information système",
    "file_operation": "Opération fichier"
}

class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory_file = "dqn_experiences.json"
        self.model_file = "dqn_model.h5"
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return None
        return random.uniform(0.1, 0.5)

    def get_training_metrics(self):
        return {
            "memory_size": len(self.memory),
            "epsilon": self.epsilon,
            "exploration_rate": f"{self.epsilon * 100:.1f}%",
            "batch_size": getattr(self, "batch_size", 32)
        }

class CircularProgress(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMaximum(100)
        self.setMinimum(0)
        self.setTextVisible(False)
        self.setFixedSize(80, 80)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        pen = QPen()
        pen.setWidth(6)
        pen.setColor(QColor("#ecf0f1"))
        painter.setPen(pen)
        painter.drawEllipse(3, 3, 74, 74)
        
        pen.setColor(QColor("#3498db"))
        painter.setPen(pen)
        
        span_angle = int(16 * 3.6 * self.value())
        painter.drawArc(3, 3, 74, 74, 90 * 16, -span_angle)
        
        font = QFont("Segoe UI", 12, QFont.Bold)
        painter.setFont(font)
        painter.setPen(QColor("#2c3e50"))
        painter.drawText(self.rect(), Qt.AlignCenter, f"{self.value()}%")

class ModernButton(QPushButton):
    def __init__(self, text, color, icon=None, parent=None):
        super().__init__(text, parent)
        self.color = color
        self.setFixedHeight(40)
        self.setStyleSheet(f"""
            QPushButton {{
                background: {color};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background: {self.adjust_color(color, 20)};
            }}
            QPushButton:pressed {{
                background: {self.adjust_color(color, -20)};
                padding: 9px 15px 7px 17px;
            }}
            QPushButton:disabled {{
                background: #95a5a6;
                color: #bdc3c7;
            }}
        """)
        
    def adjust_color(self, color, amount):
        if color.startswith('#'):
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            
            r = max(0, min(255, r + amount))
            g = max(0, min(255, g + amount))
            b = max(0, min(255, b + amount))
            
            return f"#{r:02x}{g:02x}{b:02x}"
        return color

class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        
    def write(self, text):
        if text.strip():
            QTimer.singleShot(0, lambda: self._append_text(text.strip()))
            
    def _append_text(self, text):
        self.text_widget.append(text)
        self.text_widget.moveCursor(QTextCursor.End)
        
    def flush(self):
        pass

class VirtualAssistant(QMainWindow):               
    def __init__(self):
        super().__init__()
        
        self.agent = dqn_agent
        print(f"Agent DQN initialisé: {len(self.agent.memory)} expériences")
        
        self.is_awake = False
        self.performance_data = []
        self.history = []
        self.last_command_success = 0.5
        self.batch_size = 32
        self.is_listening = False
        
        self.agent = dqn_agent
        
        assistant_signals.show_suggestions.connect(self.show_suggestions_safe)
        assistant_signals.update_display.connect(self.update_display_safe)
        assistant_signals.update_status.connect(self.update_status_safe)
        assistant_signals.add_history_item.connect(self.add_history_item_safe)
        assistant_signals.update_metrics.connect(self.update_metrics_safe)
        
        set_front_display_callback(self.afficher_message)
        self.initUI()
        self.init_timers()
        
        self.listening_thread = threading.Thread(target=self.listen_loop, daemon=True)
        self.listening_thread.start()

    def initUI(self):
        self.setWindowTitle('AG7VOC - Assistant Vocal Intelligent')
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        
        self.setStyleSheet("""
            QMainWindow {
                background: #f5f7fa;
            }
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QGroupBox {
                font-weight: bold;
                color: #2c3e50;
                border: 2px solid #dde4e6;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                background: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        sidebar = QFrame()
        sidebar.setFixedWidth(220)
        sidebar.setStyleSheet("""
            QFrame {
                background: #2c3e50;
                border: none;
            }
        """)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setSpacing(15)
        sidebar_layout.setContentsMargins(15, 20, 15, 20)
        
        logo_label = QLabel("AG7VOC")
        logo_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 20px;
                font-weight: bold;
                padding: 10px 0;
            }
        """)
        logo_label.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(logo_label)
        
        nav_buttons = [
            ("Tableau de bord", self.show_dashboard),
            ("Statistiques", self.show_statistics),
            ("Configurer la voix", self.show_voice_config),
            ("Commandes", self.show_commands),
            ("Préférences", self.show_preferences),
            ("Aide", self.show_help)
        ]
        
        for text, callback in nav_buttons:
            btn = QPushButton(text)
            btn.setStyleSheet("""
                QPushButton {
                    background: transparent;
                    color: #ecf0f1;
                    border: none;
                    text-align: left;
                    padding: 10px;
                    font-size: 13px;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background: #34495e;
                }
                QPushButton:pressed {
                    background: #2980b9;
                }
            """)
            btn.clicked.connect(callback)
            sidebar_layout.addWidget(btn)
        
        sidebar_layout.addStretch()
        
        status_frame = QFrame()
        status_frame.setStyleSheet("""
            QFrame {
                background: #34495e;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        status_layout = QVBoxLayout(status_frame)
        
        status_title = QLabel("Statut de l'assistant")
        status_title.setStyleSheet("color: white; font-weight: bold;")
        
        self.status_icon = QLabel("🔴")
        self.status_icon.setStyleSheet("font-size: 24px;")
        
        self.status_text = QLabel("En veille")
        self.status_text.setStyleSheet("color: white;")
        
        status_layout.addWidget(status_title)
        status_layout.addWidget(self.status_icon)
        status_layout.addWidget(self.status_text)
        
        sidebar_layout.addWidget(status_frame)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(15)
        content_layout.setContentsMargins(20, 20, 20, 20)
        
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background: white;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        header_layout = QHBoxLayout(header_frame)
        
        self.status_indicator = CircularProgress()
        self.status_indicator.setValue(0)
        
        status_info = QVBoxLayout()
        self.status_label = QLabel("État: En veille")
        self.status_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        
        self.status_subtitle = QLabel("Dites 'assistant' pour commencer")
        self.status_subtitle.setStyleSheet("color: #7f8c8d; font-size: 12px;")
        
        status_info.addWidget(self.status_label)
        status_info.addWidget(self.status_subtitle)
        
        header_layout.addWidget(self.status_indicator)
        header_layout.addSpacing(15)
        header_layout.addLayout(status_info)
        header_layout.addStretch()
        
        self.wake_button = ModernButton("Activer l'écoute", "#27ae60")
        self.wake_button.clicked.connect(self.toggle_listening)
        header_layout.addWidget(self.wake_button)
        
        content_layout.addWidget(header_frame)
        
        response_frame = QFrame()
        response_frame.setStyleSheet("""
            QFrame {
                background: white;
                border-radius: 10px;
                border: 1px solid #dde4e6;
            }
        """)
        response_layout = QVBoxLayout(response_frame)
        
        response_title = QLabel("Réponse de l'assistant")
        response_title.setStyleSheet("font-weight: bold; color: #2c3e50; font-size: 14px;")
        response_layout.addWidget(response_title)
        
        self.assistant_response = QLabel("Bienvenue ! Je suis votre assistant vocal AG7VOC.")
        self.assistant_response.setWordWrap(True)
        self.assistant_response.setStyleSheet("""
            QLabel {
                padding: 15px;
                font-size: 14px;
                color: #2c3e50;
                background: #f8f9fa;
                border-radius: 8px;
            }
        """)
        response_layout.addWidget(self.assistant_response)
        
        content_layout.addWidget(response_frame)
        
        buttons_grid = QGridLayout()
        buttons_grid.setSpacing(10)
        

        action_buttons = [
            ("Système", self.show_system_info, "#e74c3c"),
            ("Entraîner", self.trainAgent, "#9b59b6"),
            ("Suggestion", self.suggere_action, "#f39c12"),
            ("Historique", self.afficher_historique, "#3498db"),
            ("Préférences", self.show_preferences, "#95a5a6"),
            ("Test DQN", self.test_dqn_save, "#e67e22"),
            ("Explorateur", self.open_file_explorer, "#795548"),
            ("Test Micro", self.test_microphone, "#16a085")
        ]
        
        for i, (text, callback, color) in enumerate(action_buttons):
            btn = ModernButton(text, color)
            btn.clicked.connect(callback)
            buttons_grid.addWidget(btn, i // 4, i % 4)
        
        content_layout.addLayout(buttons_grid)
        
        learning_frame = QFrame()
        learning_frame.setStyleSheet("""
            QFrame {
                background: white;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        learning_layout = QVBoxLayout(learning_frame)
        
        learning_header = QHBoxLayout()
        learning_title = QLabel("Apprentissage de l'IA")
        learning_title.setStyleSheet("font-weight: bold; color: #2c3e50;")
        
        self.learning_value = QLabel("0%")
        self.learning_value.setStyleSheet("color: #3498db; font-weight: bold;")
        
        learning_header.addWidget(learning_title)
        learning_header.addStretch()
        learning_header.addWidget(self.learning_value)
        
        self.learning_progress = QProgressBar()
        self.learning_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #dde4e6;
                border-radius: 6px;
                text-align: center;
                background: #f8f9fa;
                height: 20px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #3498db, stop: 1 #2980b9);
                border-radius: 5px;
            }
        """)
        
        learning_layout.addLayout(learning_header)
        learning_layout.addWidget(self.learning_progress)
        content_layout.addWidget(learning_frame)
        
        journal_frame = QFrame()
        journal_frame.setStyleSheet("""
            QFrame {
                background: white;
                border-radius: 10px;
                border: 2px solid #3498db;
            }
        """)
        journal_layout = QVBoxLayout(journal_frame)
        
        journal_header = QHBoxLayout()
        journal_title = QLabel("JOURNAL DES ACTIVITÉS")
        journal_title.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #2c3e50;
                font-size: 16px;
                padding: 5px;
            }
        """)
        
        control_layout = QHBoxLayout()
        clear_btn = QPushButton("Effacer")
        clear_btn.setStyleSheet("""
            QPushButton {
                background: #e74c3c;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 10px;
            }
        """)
        
        export_btn = QPushButton("Exporter")
        export_btn.setStyleSheet("""
            QPushButton {
                background: #27ae60;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 10px;
            }
        """)
        control_layout.addWidget(clear_btn)
        control_layout.addWidget(export_btn)
        control_layout.addStretch()
        
        journal_header.addWidget(journal_title)
        journal_header.addLayout(control_layout)
        
        journal_layout.addLayout(journal_header)
        
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setStyleSheet("""
            QTextEdit {
                background: #2c3e50;
                color: #ecf0f1;
                border: 1px solid #34495e;
                border-radius: 6px;
                font-family: 'Consolas', 'Monospace';
                font-size: 11px;
                padding: 8px;
            }
        """)
        self.output_text.setMinimumHeight(250)

        palette = self.output_text.palette()
        gradient = QLinearGradient(0, 0, 0, 400)
        gradient.setColorAt(0, QColor(44, 62, 80))
        gradient.setColorAt(1, QColor(52, 73, 94))
        palette.setBrush(QPalette.Base, QBrush(gradient))
        self.output_text.setPalette(palette)
        
        journal_layout.addWidget(self.output_text)
        content_layout.addWidget(journal_frame)
        
        main_layout.addWidget(sidebar)
        main_layout.addWidget(content_widget)

        history_panel = QFrame()
        history_panel.setFixedWidth(350)
        history_panel.setStyleSheet("""
            QFrame {
                background: #f8f9fa;
                border-left: 3px solid #3498db;
                border-radius: 0 10px 10px 0;
            }
        """)
        history_layout = QVBoxLayout(history_panel)
        history_layout.setContentsMargins(15, 20, 15, 20)
        history_layout.setSpacing(10)

        history_title = QLabel("Historique des activités")
        history_title.setStyleSheet("""
            QLabel {
                font-size: 17px;
                font-weight: bold;
                color: #3498db;
                padding-bottom: 8px;
            }
        """)
        history_layout.addWidget(history_title)

        self.output_text.setParent(None)
        self.output_text.setMinimumHeight(500)
        self.output_text.setStyleSheet("""
            QTextEdit {
                background: #2c3e50;
                color: #ecf0f1;
                border: 1px solid #3498db;
                border-radius: 8px;
                font-family: 'Consolas', 'Monospace';
                font-size: 12px;
                padding: 10px;
            }
        """)
        history_layout.addWidget(self.output_text)

        search_box = QLineEdit()
        search_box.setPlaceholderText("Rechercher dans l'historique...")
        search_box.setStyleSheet("""
            QLineEdit {
                border: 1px solid #3498db;
                border-radius: 6px;
                padding: 6px;
                font-size: 12px;
            }
        """)
        history_layout.addWidget(search_box)

        def filter_history():
            query = search_box.text().lower()
            all_text = self.output_text.toPlainText().split('\n')
            filtered = [line for line in all_text if query in line.lower()]
            self.output_text.clear()
            self.output_text.append('\n'.join(filtered) if filtered else "Aucun résultat.")

        search_box.textChanged.connect(filter_history)

        main_layout.addWidget(history_panel)
        
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        
        self.redirect_stdout()
        self.setup_tray_icon()
        
        self.afficher_message("=== AG7VOC ASSISTANT VOCAL ===")
        self.afficher_message("Système initialisé avec succès")
        self.afficher_message(f"Agent DQN: {len(self.agent.memory)} expériences chargées")
        self.afficher_message("Prêt à recevoir des commandes vocales")

        self.central_stack = QStackedWidget()
        main_layout.addWidget(self.central_stack)

        self.dashboard_widget = content_widget  # ton dashboard existant
        self.commands_widget = QWidget()        # à remplir selon tes besoins

        # Ajout de la classe StatsWidget
        class StatsWidget(QWidget):
            def __init__(self, agent, parent=None):
                super().__init__(parent)
                self.agent = agent
                layout = QVBoxLayout(self)
                layout.setSpacing(20)
                layout.setContentsMargins(40, 30, 40, 30)

                title = QLabel("Statistiques de l'IA")
                title.setStyleSheet("font-size: 20px; font-weight: bold; color: #3498db;")
                layout.addWidget(title)

                self.stats_labels = {}
                stats_keys = ["memory_size", "epsilon", "exploration_rate", "batch_size"]
                for key in stats_keys:
                    lbl = QLabel(f"{key}: ...")
                    lbl.setStyleSheet("font-size: 15px; color: #2c3e50;")
                    layout.addWidget(lbl)
                    self.stats_labels[key] = lbl

                layout.addStretch()

            def update_stats_labels(self, metrics):
                for key, lbl in self.stats_labels.items():
                    value = metrics.get(key, "...")
                    lbl.setText(f"{key}: {value}")

        self.stats_widget = StatsWidget(self.agent)
        self.preferences_widget = QWidget()     # à remplir selon tes besoins
        self.help_widget = QWidget()            # à remplir selon tes besoins
        self.voice_config_widget = VoiceConfigWidget()
                
        self.central_stack.addWidget(self.dashboard_widget)
        self.central_stack.addWidget(self.commands_widget)
        self.central_stack.addWidget(self.stats_widget)
        self.central_stack.addWidget(self.preferences_widget)
        self.central_stack.addWidget(self.help_widget)
        self.central_stack.addWidget(self.voice_config_widget)

        self.central_stack.setCurrentWidget(self.dashboard_widget)

    def toggle_listening(self):
        """Active ou désactive l'écoute"""
        if not self.is_awake:
            self.wake_up_assistant()
        else:
            self.put_to_sleep()

    def call_intent_command(intent, command_text=None):
        """
        Appelle directement la commande correspondant à une intention.
        Si command_text est fourni, il est utilisé comme paramètre.
        """
        from ag7voc import (
            read_file, write_file, delete_file, create_folder, list_files,
            rename_file, move_file, show_events, add_event, delete_event, modify_event,
            get_system_info, launch_app, shutdown_computer, restart_computer, lock_computer,
            search_web, send_email, show_help, process_voice_command, INTENT_LABELS_FR
        )

        intent_map = {
            "read_file": read_file,
            "write_file": write_file,
            "delete_file": delete_file,
            "create_folder": create_folder,
            "list_files": list_files,
            "rename_file": rename_file,
            "move_file": move_file,
            "show_events": show_events,
            "add_event": add_event,
            "delete_event": delete_event,
            "modify_event": modify_event,
            "get_time": lambda: process_voice_command("il est quelle heure"),
            "get_date": lambda: process_voice_command("quelle date"),
            "show_help": show_help,
            "launch_app": lambda: launch_app(command_text or "explorateur"),
            "shutdown": shutdown_computer,
            "restart": restart_computer,
            "lock": lock_computer,
            "search_web": lambda: search_web(command_text or ""),
            "send_email": lambda: send_email(command_text or ""),
            "weather": lambda: process_voice_command("météo"),
            "list_drives": lambda: process_voice_command("lister les lecteurs"),
            "historique": lambda: process_voice_command("montre l'historique"),
            "system_info": get_system_info,
        }

        func = intent_map.get(intent)
        if func:
            try:
                if intent in ["write_file", "add_event", "delete_event", "modify_event", "search_web", "send_email", "launch_app"]:
                    func(command_text)
                else:
                    func()
            except Exception as e:
                print(f"Erreur appel commande '{intent}': {e}")
        else:
            print(f"Intention '{intent}' non reconnue.")

    
    def wake_up_assistant(self):
        """Réveille l'assistant"""
        self.is_awake = True
        self.update_status("État : Réveillé", "#2ecc71")
        self.status_icon.setText("🟢")
        self.status_text.setText("Réveillé")
        self.wake_button.setText("Mettre en veille")
        self.wake_button.color = "#e74c3c"
        self.afficher_message("Assistant réveillé - En écoute active")
        speak("Je suis réveillé. Comment puis-je vous aider ?")

    def put_to_sleep(self):
        """Met l'assistant en veille"""
        self.is_awake = False
        self.update_status("État : En veille", "#2c3c")
        self.status_icon.setText("🔴")
        self.status_text.setText("En veille")
        self.wake_button.setText("Activer l'écoute")
        self.wake_button.color = "#27ae60"
        self.afficher_message("Assistant mis en veille")
        speak("Je me mets en veille. Dites 'assistant' si vous avez besoin de moi.")

    def update_status(self, text, color):
        """Met à jour le statut de l'assistant"""
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {color};")
        self.status_subtitle.setText("Prêt à répondre à vos commandes" if self.is_awake else "Dites 'assistant' pour commencer")

    def init_timers(self):
        """Initialise les timers pour les mises à jour périodiques"""
        self.metrics_timer = QTimer()
        self.metrics_timer.timeout.connect(self.update_real_time_metrics)
        self.metrics_timer.start(2000)
        
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.animate_status)
        self.status_timer.start(100)

    def animate_status(self):
        """Animation du statut circulaire"""
        if self.is_awake:
            current = self.status_indicator.value()
            if current < 100:
                self.status_indicator.setValue(current + 2)
        else:
            current = self.status_indicator.value()
            if current > 0:
                self.status_indicator.setValue(current - 2)

    def update_real_time_metrics(self):
        """Met à jour les métriques en temps réel"""
        try:
            if self.agent is None:
                return
                
            metrics = {
                'q_score': f"{self.agent.epsilon:.3f}",
                'exploration_rate': f"{self.agent.epsilon * 100:.1f}%",
                'memory_size': f"{len(self.agent.memory):,}",
                'accuracy': f"{min(100, int(100 * len(self.agent.memory) / 2000))}%"
            }
            
            assistant_signals.update_metrics.emit(metrics)
            
            progress = min(100, int(100 * len(self.agent.memory) / 2000))
            self.learning_progress.setValue(progress)
            self.learning_value.setText(f"{progress}%")
            
        except Exception as e:
            print(f"Erreur mise à jour métriques: {e}")

    def listen_loop(self):
        """Boucle d'écoute principale"""
        while True:
            try:
                self.check_automation()  # Ajout ici
                if not self.is_awake:
                    text = listen(timeout=1)
                    if text and any(word in text.lower() for word in voice_prefs.get_preference("wake_words")):
                        self.wake_up_assistant()
                        time.sleep(2)
                else:
                    text = listen(timeout=3)
                    if text:
                        self.process_command(text)
                        if any(word in text.lower() for word in voice_prefs.get_preference("sleep_words")):
                            self.put_to_sleep()
            except Exception as e:
                print(f"Erreur écoute: {e}")
                time.sleep(1)

    def process_command(self, text):
        try:
            self.afficher_message(f"Commande reçue: {text}")
            self.history.append(text)

            from ag7voc import process_voice_command, get_top_intents_spacy_similarity, INTENT_LABELS_FR, get_command_suggestions

            response = process_voice_command(text)
            self.afficher_message(f"Réponse: {response}")
            speak(response)

            suggestions = get_command_suggestions(text, top_n=3)
            if suggestions:
                self.afficher_message("Suggestions IA contextuelles :")
                for kw, label_fr, intent, score in suggestions:
                    self.afficher_message(f"- {label_fr} : \"{kw}\" (score: {score:.2f})")
                recent_intents = [h[1] for h in self.history[-5:] if isinstance(h, tuple) and len(h) > 1]
                if any(intent == ri for _, _, intent, _ in suggestions for ri in recent_intents):
                    self.afficher_message("Action similaire détectée dans l'historique. Voulez-vous répéter ?")
            else:
                self.afficher_message("Aucune suggestion IA pertinente.")

        except Exception as e:
            self.afficher_message(f"Erreur traitement commande: {str(e)}")
            self.last_command_success = 0.1

    def afficher_message(self, message):
        """Affiche un message dans le journal"""
        try:
            timestamp = time.strftime('%H:%M:%S')
            formatted_message = f"[{timestamp}] {message}"
            self.output_text.append(formatted_message)
            self.output_text.moveCursor(QTextCursor.End)
        except Exception as e:
            print(f"Erreur affichage message: {e}")

    def redirect_stdout(self):
        """Redirige stdout vers le widget de texte"""
        sys.stdout = StdoutRedirector(self.output_text)

    def setup_tray_icon(self):
        """Configure l'icône de la barre système"""
        try:
            self.tray_icon = QSystemTrayIcon(self)
            self.tray_icon.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
            
            tray_menu = QMenu()
            show_action = tray_menu.addAction("Afficher")
            show_action.triggered.connect(self.show)
            
            quit_action = tray_menu.addAction("Quitter")
            quit_action.triggered.connect(QApplication.quit)
            
            self.tray_icon.setContextMenu(tray_menu)
            self.tray_icon.show()
            self.tray_icon.activated.connect(self.tray_icon_activated)
        except Exception as e:
            print(f"Erreur configuration tray icon: {e}")

    def tray_icon_activated(self, reason):
        """Gère les clics sur l'icône de la barre système"""
        if reason == QSystemTrayIcon.DoubleClick:
            self.show()
            self.activateWindow()

    def closeEvent(self, event):
        """Gère la fermeture de l'application"""
        event.ignore()
        self.hide()
        self.tray_icon.showMessage(
            "AG7VOC",
            "L'assistant continue de fonctionner en arrière-plan",
            QSystemTrayIcon.Information,
            2000
        )

    def show_dashboard(self):
        self.central_stack.setCurrentWidget(self.dashboard_widget)

    def show_commands(self):
        self.central_stack.setCurrentWidget(self.commands_widget)

    def show_statistics(self):
        self.central_stack.setCurrentWidget(self.stats_widget)

    def show_preferences(self):
        self.central_stack.setCurrentWidget(self.preferences_widget)

    def show_voice_config(self):
        self.central_stack.setCurrentWidget(self.voice_config_widget)
    
    def show_help(self):
        self.central_stack.setCurrentWidget(self.help_widget)

    def show_system_info(self):
        info = get_system_info()
        self.afficher_message("=== INFORMATIONS SYSTÈME ===")
        for key, value in info.items():
            if isinstance(value, dict):
                self.afficher_message(f"{key}:")
                for k, v in value.items():
                    self.afficher_message(f"  {k}: {v}")
            else:
                self.afficher_message(f"{key}: {value}")
        speak("Voici les informations de votre système.")

    def trainAgent(self):
        if self.agent:
            try:
                loss = self.agent.replay(self.batch_size)
                if loss is not None:
                    self.afficher_message(f"Entraînement terminé. Perte: {loss:.4f}")
                    speak("Entraînement terminé avec succès.")
                else:
                    self.afficher_message("Données insuffisantes pour l'entraînement")
            except Exception as e:
                self.afficher_message(f"Erreur entraînement: {str(e)}")
        else:
            self.afficher_message("Agent DQN non disponible")

    def suggere_action(self):
        if self.history:
            last_command = self.history[-1]
            suggestions = get_top_intents_spacy_similarity(last_command)
            self.afficher_message("Suggestions IA:")
            for intent, score in suggestions:
                label = INTENT_LABELS_FR.get(intent, intent)
                self.afficher_message(f"- {label} (score: {score:.2f})")
        else:
            self.afficher_message("Aucune commande à suggérer.")

    def afficher_historique(self):
        self.afficher_message("Historique des commandes:")
        if self.history:
            for i, cmd in enumerate(self.history[-10:], 1):  # Affiche les 10 dernières
                self.afficher_message(f"{i}. {cmd}")
        else:
            self.afficher_message("Aucune commande enregistrée.")

    def test_dqn_save(self):
        if self.agent:
            try:
                test_state = np.random.rand(1, 5)
                test_next_state = np.random.rand(1, 5)
                self.agent.remember(test_state, 0, 10, test_next_state, False)
                self.afficher_message("Test DQN réussi - Expérience sauvegardée")
            except Exception as e:
                self.afficher_message(f"Erreur test DQN: {str(e)}")
        else:
            self.afficher_message("Agent DQN non disponible")

    def open_file_explorer(self):
        try:
            os.system("explorer .")
            self.afficher_message("Explorateur de fichiers ouvert")
        except Exception as e:
            self.afficher_message(f"Erreur ouverture explorateur: {str(e)}")

    def test_microphone(self):
        try:
            self.afficher_message("Test du microphone en cours...")
            speak("Test du microphone. Parlez maintenant.")
            
            text = listen(timeout=5)
            if text:
                self.afficher_message(f"Test réussi: '{text}'")
                speak(f"J'ai entendu: {text}")
            else:
                self.afficher_message("Aucun son détecté")
                speak("Je n'ai rien entendu")
                
        except Exception as e:
            self.afficher_message(f"Erreur test microphone: {str(e)}")

    def export_logs(self):
        """Exporte les logs vers un fichier"""
        try:
            filename = f"ag7voc_logs_{time.strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.output_text.toPlainText())
            self.afficher_message(f"Logs exportés: {filename}")
        except Exception as e:
            self.afficher_message(f"Erreur export logs: {str(e)}")

    def show_suggestions_safe(self, suggestions, original_command):
        QTimer.singleShot(0, lambda: self.show_suggestions(suggestions, original_command))

    def update_display_safe(self, message):
        QTimer.singleShot(0, lambda: self.assistant_response.setText(message))

    def update_status_safe(self, text, color, icon="ℹ️"):
        QTimer.singleShot(0, lambda: self.update_status(f"{icon} {text}", color))

    def add_history_item_safe(self, msg, item_type="info"):
        QTimer.singleShot(0, lambda: self.add_history_item(msg, item_type))

    def update_metrics_safe(self, metrics):
        QTimer.singleShot(0, lambda: self.update_metrics(metrics))

    def show_suggestions(self, suggestions, original_command):
        self.afficher_message(f"Suggestions pour '{original_command}': {suggestions}")

    def add_history_item(self, msg, item_type):
        colors = {
            "info": "#3498db", "success": "#27ae60", 
            "warning": "#f39c12", "error": "#e74c3c", 
            "command": "#9b59b6"
        }
        colored_msg = f"<font color='{colors.get(item_type, '#2c3e50')}'>{msg}</font>"
        self.output_text.append(colored_msg)

    def update_metrics(self, metrics):
        if hasattr(self, "stats_widget"):
            self.stats_widget.update_stats_labels(metrics)

    def check_automation(self):
        """
        Propose des routines automatiques selon l'historique :
        Si une commande A est souvent suivie d'une commande B,
        alors après A, l'assistant propose automatiquement B.
        """
        try:
            with open("history.json", "r", encoding="utf-8") as f:
                history = json.load(f)
            if len(history) < 2:
                return

            sequence_counts = {}
            for i in range(len(history) - 1):
                cmd_a, intent_a = history[i]
                cmd_b, intent_b = history[i + 1]
                key = (intent_a, intent_b)
                sequence_counts[key] = sequence_counts.get(key, 0) + 1

            if self.history:
                last_cmd = self.history[-1]
                from ag7voc import get_intent_spacy_similarity
                last_intent = get_intent_spacy_similarity(last_cmd)
                candidates = [(b, count) for (a, b), count in sequence_counts.items() if a == last_intent]
                if candidates:
                    next_intent, _ = max(candidates, key=lambda x: x[1])
                    from ag7voc import INTENT_LABELS_FR
                    label_fr = INTENT_LABELS_FR.get(next_intent, next_intent)
                    self.afficher_message(f"Suggestion IA : Après cette commande, vous exécutez souvent '{label_fr}'. Voulez-vous la lancer ?")
                    speak(f"Voulez-vous que je lance la commande suivante : {label_fr} ?")
                    answer = listen(timeout=5)
                    if answer and "oui" in answer.lower():
                        self.call_intent_command(next_intent)
        except Exception as e:
            print(f"Erreur analyse automatisation : {e}")
    
def compute_reward(command, success_rate):
    """Calcule la récompense pour l'apprentissage par renforcement"""
    base_reward = 1.0 if success_rate > 0.7 else -1.0
    complexity_bonus = min(2.0, len(command.split()) / 5)
    return base_reward + complexity_bonus

def get_current_state(command, success_rate, history_length):
    """Simule un état pour le DQN"""
    return np.random.rand(5)

class VoiceConfigWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        from voice_manager import voice_manager
        from voice_preferences import voice_prefs

        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(40, 30, 40, 30)

        title = QLabel("Configuration de la voix")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #3498db;")
        layout.addWidget(title)

        speed_label = QLabel("Vitesse de la voix")
        speed_slider = QSlider(Qt.Horizontal)
        speed_slider.setMinimum(100)
        speed_slider.setMaximum(300)
        speed_slider.setValue(voice_manager.tts_engine.getProperty('rate'))
        speed_slider.setTickInterval(10)
        speed_slider.setTickPosition(QSlider.TicksBelow)
        layout.addWidget(speed_label)
        layout.addWidget(speed_slider)

        volume_label = QLabel("Volume de la voix")
        volume_slider = QSlider(Qt.Horizontal)
        volume_slider.setMinimum(0)
        volume_slider.setMaximum(100)
        volume_slider.setValue(int(voice_manager.tts_engine.getProperty('volume') * 100))
        volume_slider.setTickInterval(10)
        volume_slider.setTickPosition(QSlider.TicksBelow)
        layout.addWidget(volume_label)
        layout.addWidget(volume_slider)

        voice_label = QLabel("Type de voix")
        voice_combo = QComboBox()
        voices = voice_manager.tts_engine.getProperty('voices')
        for v in voices:
            voice_combo.addItem(v.name, v.id)
        current_voice_id = voice_manager.tts_engine.getProperty('voice')
        idx = voice_combo.findData(current_voice_id)
        if idx >= 0:
            voice_combo.setCurrentIndex(idx)
        layout.addWidget(voice_label)
        layout.addWidget(voice_combo)

        # Mode de reconnaissance
        mode_label = QLabel("Mode de reconnaissance vocale")
        mode_combo = QComboBox()
        mode_combo.addItems(["google", "vosk"])
        mode_combo.setCurrentText(voice_prefs.get_preference("recognition_mode"))
        layout.addWidget(mode_label)
        layout.addWidget(mode_combo)

        # Langue
        lang_label = QLabel("Langue")
        lang_edit = QLineEdit()
        lang_edit.setText(voice_prefs.get_preference("language"))
        layout.addWidget(lang_label)
        layout.addWidget(lang_edit)

        # Sensibilité VAD
        vad_label = QLabel("Sensibilité détection voix (VAD)")
        vad_slider = QSlider(Qt.Horizontal)
        vad_slider.setMinimum(0)
        vad_slider.setMaximum(100)
        vad_slider.setValue(int(voice_prefs.get_preference("vad_sensitivity") * 100))
        vad_slider.setTickInterval(5)
        vad_slider.setTickPosition(QSlider.TicksBelow)
        layout.addWidget(vad_label)
        layout.addWidget(vad_slider)

        # Bouton test voix
        test_btn = QPushButton("Tester la voix")
        test_btn.setStyleSheet("background: #27ae60; color: white; font-weight: bold; border-radius: 8px; padding: 8px;")
        layout.addWidget(test_btn)

        # Sauvegarder
        save_btn = QPushButton("Sauvegarder la configuration")
        save_btn.setStyleSheet("background: #3498db; color: white; font-weight: bold; border-radius: 8px; padding: 8px;")
        layout.addWidget(save_btn)

        layout.addStretch()

        # Callbacks
        def update_voice_settings():
            voice_manager.tts_engine.setProperty('rate', speed_slider.value())
            voice_manager.tts_engine.setProperty('volume', volume_slider.value() / 100)
            selected_voice_id = voice_combo.currentData()
            voice_manager.tts_engine.setProperty('voice', selected_voice_id)
            voice_manager.setup_voice_settings()
            # Préférences
            voice_prefs.preferences["voice_speed"] = speed_slider.value()
            voice_prefs.preferences["voice_volume"] = volume_slider.value() / 100
            voice_prefs.preferences["voice_id"] = selected_voice_id
            voice_prefs.preferences["recognition_mode"] = mode_combo.currentText()
            voice_prefs.preferences["language"] = lang_edit.text()
            voice_prefs.preferences["vad_sensitivity"] = vad_slider.value() / 100

        speed_slider.valueChanged.connect(update_voice_settings)
        volume_slider.valueChanged.connect(update_voice_settings)
        voice_combo.currentIndexChanged.connect(update_voice_settings)
        mode_combo.currentIndexChanged.connect(update_voice_settings)
        lang_edit.textChanged.connect(update_voice_settings)
        vad_slider.valueChanged.connect(update_voice_settings)

        def test_voice():
            update_voice_settings()
            voice_manager.speak("Ceci est un test de la voix de l'assistant.", async_mode=False)

        test_btn.clicked.connect(test_voice)

        def save_config():
            update_voice_settings()
            # Sauvegarde dans le fichier JSON
            try:
                import json
                with open("voice_preferences.json", "w", encoding="utf-8") as f:
                    json.dump(voice_prefs.preferences, f, ensure_ascii=False, indent=2)
                QMessageBox.information(self, "Sauvegarde", "Configuration vocale sauvegardée avec succès.")
            except Exception as e:
                QMessageBox.warning(self, "Erreur", f"Erreur sauvegarde: {e}")

        save_btn.clicked.connect(save_config)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(245, 247, 250))
    palette.setColor(QPalette.WindowText, QColor(44, 62, 80))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(245, 247, 250))
    palette.setColor(QPalette.Text, QColor(44, 62, 80))
    palette.setColor(QPalette.Button, QColor(52, 152, 219))
    palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.Highlight, QColor(52, 152, 219))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)
    
    window = VirtualAssistant()
    window.show()
    
    sys.exit(app.exec_())
