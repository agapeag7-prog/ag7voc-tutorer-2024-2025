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

# Définition des signaux pour la communication thread-safe
class AssistantSignals(QObject):
    show_suggestions = pyqtSignal(list, str)
    update_display = pyqtSignal(str)
    update_status = pyqtSignal(str, str, str)
    add_history_item = pyqtSignal(str, str)
    update_metrics = pyqtSignal(dict)

assistant_signals = AssistantSignals()

# Configuration des préférences vocales
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

# Simulation des fonctions manquantes
def process_voice_command(command, forced_intent=None):
    print(f"Traitement de la commande: {command}")
    if forced_intent:
        print(f"Intention forcée: {forced_intent}")
    
    # Simulation de quelques commandes de base
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
    # Simulation de la reconnaissance d'intention
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

# Constantes simulées
COMMAND_HISTORY = []
INTENT_LABELS_FR = {
    "time_query": "Demande d'heure",
    "system_info": "Information système",
    "file_operation": "Opération fichier"
}

# Simulation DQN Agent
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

dqn_agent = DQNAgent()

# Classes d'interface améliorées
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
        
        # Draw background circle
        pen = QPen()
        pen.setWidth(6)
        pen.setColor(QColor("#ecf0f1"))
        painter.setPen(pen)
        painter.drawEllipse(3, 3, 74, 74)
        
        # Draw progress arc
        pen.setColor(QColor("#3498db"))
        painter.setPen(pen)
        
        span_angle = int(16 * 3.6 * self.value())
        painter.drawArc(3, 3, 74, 74, 90 * 16, -span_angle)
        
        # Draw text
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
        
        # Connecter les signaux
        assistant_signals.show_suggestions.connect(self.show_suggestions_safe)
        assistant_signals.update_display.connect(self.update_display_safe)
        assistant_signals.update_status.connect(self.update_status_safe)
        assistant_signals.add_history_item.connect(self.add_history_item_safe)
        assistant_signals.update_metrics.connect(self.update_metrics_safe)
        
        set_front_display_callback(self.afficher_message)
        self.initUI()
        self.init_timers()
        
        # Démarre l'écoute du mot-clé dès le lancement
        self.listening_thread = threading.Thread(target=self.listen_loop, daemon=True)
        self.listening_thread.start()

    def initUI(self):
        self.setWindowTitle('AG7VOC - Assistant Vocal Intelligent')
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        
        # Style global de l'application
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
        
        # Sidebar
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
        
        # Logo et titre
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
        
        # Menu de navigation
        nav_buttons = [
            ("🏠 Tableau de bord", self.show_dashboard),
            ("🎤 Commandes", self.show_commands),
            ("📊 Statistiques", self.show_statistics),
            ("⚙️ Préférences", self.show_preferences),
            ("❓ Aide", self.show_help)
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
        
        # Status en bas de la sidebar
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
        
        # Contenu principal
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(15)
        content_layout.setContentsMargins(20, 20, 20, 20)
        
        # Header avec indicateur de statut
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background: white;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        header_layout = QHBoxLayout(header_frame)
        
        # Indicateur circulaire
        self.status_indicator = CircularProgress()
        self.status_indicator.setValue(0)
        
        # Informations de statut
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
        
        # Bouton d'action principal
        self.wake_button = ModernButton("🎤 Activer l'écoute", "#27ae60")
        self.wake_button.clicked.connect(self.toggle_listening)
        header_layout.addWidget(self.wake_button)
        
        content_layout.addWidget(header_frame)
        
        # Réponse de l'assistant
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
        
        self.assistant_response = QLabel("👋 Bienvenue ! Je suis votre assistant vocal AG7VOC.")
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
        
        # Grille de boutons d'action
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
        
        # Progression de l'apprentissage
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
        
        # Journal d'activité - PARTIE AMÉLIORÉE
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
        journal_title = QLabel("📝 JOURNAL DES ACTIVITÉS")
        journal_title.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #2c3e50;
                font-size: 16px;
                padding: 5px;
            }
        """)
        
        # Boutons de contrôle du journal
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
        
        # Zone de texte du journal avec style amélioré
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

        # Ajouter un dégradé de fond pour mieux voir les nouveaux messages
        palette = self.output_text.palette()
        gradient = QLinearGradient(0, 0, 0, 400)
        gradient.setColorAt(0, QColor(44, 62, 80))
        gradient.setColorAt(1, QColor(52, 73, 94))
        palette.setBrush(QPalette.Base, QBrush(gradient))
        self.output_text.setPalette(palette)
        
        journal_layout.addWidget(self.output_text)
        content_layout.addWidget(journal_frame)
        
        # Ajouter les sections à la disposition principale
        main_layout.addWidget(sidebar)
        main_layout.addWidget(content_widget)
        
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        
        self.redirect_stdout()
        self.setup_tray_icon()
        
        # Message de bienvenue
        self.afficher_message("=== AG7VOC ASSISTANT VOCAL ===")
        self.afficher_message("Système initialisé avec succès")
        self.afficher_message(f"Agent DQN: {len(self.agent.memory)} expériences chargées")
        self.afficher_message("Prêt à recevoir des commandes vocales")

    def toggle_listening(self):
        """Active ou désactive l'écoute"""
        if not self.is_awake:
            self.wake_up_assistant()
        else:
            self.put_to_sleep()

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
        self.update_status("État : En veille", "#e74c3c")
        self.status_icon.setText("🔴")
        self.status_text.setText("En veille")
        self.wake_button.setText("🎤 Activer l'écoute")
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
                if not self.is_awake:
                    # En veille, écoute seulement le mot-clé
                    text = listen(timeout=1)
                    if text and any(word in text.lower() for word in voice_prefs.get_preference("wake_words")):
                        self.wake_up_assistant()
                        time.sleep(2)
                else:
                    # Réveillé, écoute les commandes
                    text = listen(timeout=3)
                    if text:
                        self.process_command(text)
                        # Vérifier les commandes de mise en veille
                        if any(word in text.lower() for word in voice_prefs.get_preference("sleep_words")):
                            self.put_to_sleep()
            except Exception as e:
                print(f"Erreur écoute: {e}")
                time.sleep(1)

    def process_command(self, text):
        """Traite une commande vocale avec toutes les fonctionnalités du backend"""
        try:
            self.afficher_message(f"Commande reçue: {text}")
            self.history.append(text)

            # Appel du backend AG7VOC pour traiter la commande
            from ag7voc import process_voice_command, get_top_intents_spacy_similarity, INTENT_LABELS_FR

            # Suggestions IA automatiques AVANT exécution
            suggestions = get_top_intents_spacy_similarity(text)
            if suggestions:
                self.afficher_message("Suggestions IA (avant exécution):")
                for intent, score in suggestions:
                    label = INTENT_LABELS_FR.get(intent, intent)
                    self.afficher_message(f"- {label} (score: {score:.2f})")

            # Exécution réelle de la commande
            response = process_voice_command(text)
            self.afficher_message(f"Réponse: {response}")
            speak(response)

            # Suggestions IA automatiques APRÈS exécution (optionnel)
            # suggestions_after = get_top_intents_spacy_similarity(response)
            # if suggestions_after:
            #     self.afficher_message("Suggestions IA (après exécution):")
            #     for intent, score in suggestions_after:
            #         label = INTENT_LABELS_FR.get(intent, intent)
            #         self.afficher_message(f"- {label} (score: {score:.2f})")

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

    # Méthodes pour les différentes vues
    def show_dashboard(self):
        self.afficher_message("Affichage du tableau de bord")

    def show_commands(self):
        self.afficher_message("Affichage des commandes")

    def show_statistics(self):
        self.afficher_message("Affichage des statistiques")

    def show_preferences(self):
        self.afficher_message("Affichage des préférences")

    def show_help(self):
        self.afficher_message("Affichage de l'aide")

    # Méthodes pour les actions des boutons
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
                    self.afficher_message(f"✅ Entraînement terminé. Perte: {loss:.4f}")
                    speak("Entraînement terminé avec succès.")
                else:
                    self.afficher_message("❌ Données insuffisantes pour l'entraînement")
            except Exception as e:
                self.afficher_message(f"❌ Erreur entraînement: {str(e)}")
        else:
            self.afficher_message("❌ Agent DQN non disponible")

    def suggere_action(self):
        if self.history:
            last_command = self.history[-1]
            suggestions = get_top_intents_spacy_similarity(last_command)
            self.afficher_message("💡 Suggestions IA:")
            for intent, score in suggestions:
                label = INTENT_LABELS_FR.get(intent, intent)
                self.afficher_message(f"- {label} (score: {score:.2f})")
        else:
            self.afficher_message("ℹ️ Aucune commande à suggérer.")

    def afficher_historique(self):
        self.afficher_message("📋 Historique des commandes:")
        if self.history:
            for i, cmd in enumerate(self.history[-10:], 1):  # Affiche les 10 dernières
                self.afficher_message(f"{i}. {cmd}")
        else:
            self.afficher_message("ℹ️ Aucune commande enregistrée.")

    def test_dqn_save(self):
        if self.agent:
            try:
                test_state = np.random.rand(1, 5)
                test_next_state = np.random.rand(1, 5)
                self.agent.remember(test_state, 0, 10, test_next_state, False)
                self.afficher_message("✅ Test DQN réussi - Expérience sauvegardée")
            except Exception as e:
                self.afficher_message(f"❌ Erreur test DQN: {str(e)}")
        else:
            self.afficher_message("❌ Agent DQN non disponible")

    def open_file_explorer(self):
        try:
            os.system("explorer .")
            self.afficher_message("📁 Explorateur de fichiers ouvert")
        except Exception as e:
            self.afficher_message(f"❌ Erreur ouverture explorateur: {str(e)}")

    def test_microphone(self):
        try:
            self.afficher_message("🎤 Test du microphone en cours...")
            speak("Test du microphone. Parlez maintenant.")
            
            text = listen(timeout=5)
            if text:
                self.afficher_message(f"✅ Test réussi: '{text}'")
                speak(f"J'ai entendu: {text}")
            else:
                self.afficher_message("❌ Aucun son détecté")
                speak("Je n'ai rien entendu")
                
        except Exception as e:
            self.afficher_message(f"❌ Erreur test microphone: {str(e)}")

    def export_logs(self):
        """Exporte les logs vers un fichier"""
        try:
            filename = f"ag7voc_logs_{time.strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.output_text.toPlainText())
            self.afficher_message(f"✅ Logs exportés: {filename}")
        except Exception as e:
            self.afficher_message(f"❌ Erreur export logs: {str(e)}")

    # Méthodes pour gérer les signaux thread-safe
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
        self.afficher_message(f"💡 Suggestions pour '{original_command}': {suggestions}")

    def add_history_item(self, msg, item_type):
        colors = {
            "info": "#3498db", "success": "#27ae60", 
            "warning": "#f39c12", "error": "#e74c3c", 
            "command": "#9b59b6"
        }
        colored_msg = f"<font color='{colors.get(item_type, '#2c3e50')}'>{msg}</font>"
        self.output_text.append(colored_msg)

    def update_metrics(self, metrics):
        for key, value in metrics.items():
            self.afficher_message(f"📊 {key}: {value}")

# Fonctions manquantes simulées
def compute_reward(command, success_rate):
    """Calcule la récompense pour l'apprentissage par renforcement"""
    base_reward = 1.0 if success_rate > 0.7 else -1.0
    complexity_bonus = min(2.0, len(command.split()) / 5)
    return base_reward + complexity_bonus

def get_current_state(command, success_rate, history_length):
    """Simule un état pour le DQN"""
    return np.random.rand(5)

# Point d'entrée principal
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Appliquer une palette de couleurs globale
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