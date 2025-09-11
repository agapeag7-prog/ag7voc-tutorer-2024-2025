import sys
import random
import io
import threading
import os
import time
import psutil
import numpy as np
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
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon, QLinearGradient, QPainter, QPainterPath, QPixmap, QBrush, QTextCursor

from ag7voc import (
    process_voice_command,
    get_top_intents_spacy_similarity,
    listen,
    speak,
    set_front_display_callback,
    COMMAND_HISTORY,
    get_system_info,
    INTENT_LABELS_FR,
)
from ai_engine import DQNAgent, ACTIONS, get_current_state, compute_reward, analyze_user_sentiment
from voice_manager import voice_manager
from voice_preferences import voice_prefs

# Import or define assistant_signals
from ag7voc import assistant_signals

class VirtualAssistant(QMainWindow):
    def __init__(self):
        super().__init__()
        
        try:
            from ai_engine import dqn_agent
            if hasattr(dqn_agent, 'memory'):
                self.agent = dqn_agent
                print(f"Agent DQN initialisé: {len(self.agent.memory)} expériences")
            else:
                self.agent = None
                print("Agent DQN non disponible")
        except Exception as e:
            print(f"Erreur initialisation DQN: {e}")
            self.agent = None
        
        self.performance_data = []
        self.is_awake = False
        self.history = []
        self.last_command_success = 0.5
        self.batch_size = 32
        
        assistant_signals.show_suggestions.connect(self.show_suggestions_safe)
        assistant_signals.update_display.connect(self.update_display_safe)
        assistant_signals.update_status.connect(self.update_status_safe)
        assistant_signals.add_history_item.connect(self.add_history_item_safe)
        assistant_signals.update_metrics.connect(self.update_metrics_safe)
        # assistant_signals.update_learning_stats.connect(self.update_learning_stats_safe)
        # assistant_signals.show_notification.connect(self.show_notification_safe)
        
        set_front_display_callback(self.afficher_message)
        self.initUI()
        self.init_timers()
        
        self.wait_for_wake_word()
        
        self.metrics_timer = QTimer()
        self.metrics_timer.timeout.connect(self.update_real_time_metrics)
        self.metrics_timer.start(2000)
        
        self.wait_for_wake_word()

    def open_file_explorer(self):
        try:
            from file_explorer import open_file_explorer
            selected_path = open_file_explorer("open")
            if selected_path:
                self.output_text.append(f"Fichier sélectionné : {selected_path}")
                speak(f"Fichier sélectionné : {selected_path}")
            else:
                self.output_text.append("Aucune sélection")
                speak("Aucune sélection")
        except Exception as e:
            self.output_text.append(f"Erreur explorateur: {e}")
    
    def show_suggestions_safe(self, suggestions, original_command):
        """Affiche les suggestions de manière thread-safe"""
        try:
            self.show_suggestions_dialog(suggestions, original_command)
        except Exception as e:
            print(f"Erreur affichage suggestions: {e}")
    
    def test_dqn_save(self):
        """Teste la sauvegarde DQN"""
        try:
            if self.agent is None:
                print("Agent DQN non initialisé")
                return
                
            test_state = np.array([1, 2, 3, 4, 5])
            test_next_state = np.array([1, 2, 3, 4, 6])
            
            self.agent.remember(test_state, 0, 10, test_next_state, False)
            print("Expérience test sauvegardée")
            
            if os.path.exists(self.agent.memory_file):
                print(f"Fichier existe: {os.path.abspath(self.agent.memory_file)}")
                with open(self.agent.memory_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    print(f"Contenu: {content}")
            else:
                print("Fichier n'existe pas")
                
        except Exception as e:
            print(f"Erreur test DQN: {e}")
    
    def save_experience(self, state, action, reward, next_state, done):
        import json
        try:
            exp = {
                "state": state.tolist() if hasattr(state, 'tolist') else state,
                "action": action,
                "reward": reward,
                "next_state": next_state.tolist() if hasattr(next_state, 'tolist') else next_state,
                "done": done
            }
            
            abs_path = os.path.abspath(self.memory_file)
            print(f"Sauvegarde expérience dans: {abs_path}")
            
            with open(self.memory_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(exp) + "\n")
                
            print(f"Expérience sauvegardée: {exp}")
                
        except Exception as e:
            print(f"Erreur sauvegarde expérience: {e}")
    
    def update_display_safe(self, msg):
        """Mise à jour thread-safe de l'affichage principal"""
        try:
            self.assistant_response.setText(msg)
            if "Calibration" in msg:
                self.status_indicator.setValue(30)
                self.status_label.setText("État : Calibration")
                self.status_label.setStyleSheet("color: #f39c12; font-size: 16px; font-weight: bold;")
            elif "J'écoute" in msg:
                self.status_indicator.setValue(70)
                self.status_label.setText("État : Écoute active")
                self.status_label.setStyleSheet("color: #27ae60; font-size: 16px; font-weight: bold;")
            elif "réveillé" in msg.lower():
                self.status_indicator.setValue(100)
                self.status_label.setText("État : Réveillé")
                self.status_label.setStyleSheet("color: #2ecc71; font-size: 16px; font-weight: bold;")
            elif "veille" in msg.lower():
                self.status_indicator.setValue(0)
                self.status_label.setText("État : En veille")
                self.status_label.setStyleSheet("color: #e74c3c; font-size: 16px; font-weight: bold;")
        except Exception as e:
            print(f"Erreur update_display_safe: {e}")

    def update_status_safe(self, text, color, icon="ℹ️"):
        """Mise à jour thread-safe du statut"""
        try:
            self.status_label.setText(f"{icon} {text}")
            self.status_label.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: bold;")
        except Exception as e:
            print(f"Erreur update_status_safe: {e}")

    def add_history_item_safe(self, msg, item_type="info"):
        """Ajout thread-safe à l'historique"""
        try:
            # Ajouter au journal de sortie
            self.output_text.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
            
            # Définir les couleurs selon le type
            colors = {
                "info": "#3498db",
                "success": "#27ae60", 
                "warning": "#f39c12",
                "error": "#e74c3c",
                "command": "#9b59b6"
            }
            
            if hasattr(self, 'history_widget'):
                item = QListWidgetItem(msg)
                item.setForeground(QColor(colors.get(item_type, "#2c3e50")))
                self.history_widget.addItem(item)
                self.history_widget.scrollToBottom()
                
        except Exception as e:
            print(f"Erreur add_history_item_safe: {e}")

    def update_metrics_safe(self, metrics):
        """Mise à jour thread-safe des métriques"""
        try:
            for key, value in metrics.items():
                label = self.findChild(QLabel, f"metric_{key}")
                if label:
                    label.setText(str(value))
                        
            if hasattr(self, 'performance_data') and self.performance_data:
                if hasattr(self, 'dashboard_tab'):
                    self.dashboard_tab.update_graph(self.performance_data)
                        
        except Exception as e:
            print(f"Erreur update_metrics_safe: {e}")

    def show_suggestions_safe(self, suggestions, original_command):
        """Affiche les suggestions de manière thread-safe"""
        try:
            self.show_suggestions_dialog(suggestions, original_command)
        except Exception as e:
            print(f"Erreur affichage suggestions: {e}")
    
    def afficher_message(self, msg):
        """Utilise les signaux pour les mises à jour UI thread-safe"""
        try:
            if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "show_suggestions":
                assistant_signals.show_suggestions.emit(msg[1]["suggestions"], msg[1]["original_command"])
            else:
                assistant_signals.update_display.emit(msg)
                assistant_signals.add_history_item.emit(msg, "info")
        except Exception as e:
            print(f"Erreur afficher_message: {e}")
    
    def initUI(self):
        self.setWindowTitle('AG7VOC - Assistant Vocal Intelligent')
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        
        self.history_widget = QListWidget()
        self.history_widget.setMaximumHeight(150)
        
        assistant_signals.update_display.connect(self.update_display_safe)
        assistant_signals.update_status.connect(self.update_status_safe)
        assistant_signals.add_history_item.connect(self.add_history_item_safe)
        assistant_signals.update_metrics.connect(self.update_metrics_safe)
        
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(236, 240, 241))
        palette.setColor(QPalette.WindowText, QColor(44, 62, 80))
        palette.setColor(QPalette.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
        self.setPalette(palette)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        title_bar = QFrame()
        title_bar.setFixedHeight(40)
        title_bar.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #2c3e50, stop: 1 #34495e);
                border: none;
            }
        """)
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(15, 0, 15, 0)
        
        title_label = QLabel("AG7VOC Assistant Vocal")
        title_label.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
        
        minimize_btn = QToolButton()
        minimize_btn.setText("─")
        minimize_btn.setStyleSheet("QToolButton { color: white; border: none; font-weight: bold; }")
        minimize_btn.clicked.connect(self.showMinimized)
        
        close_btn = QToolButton()
        close_btn.setText("✕")
        close_btn.setStyleSheet("QToolButton { color: white; border: none; font-weight: bold; }")
        close_btn.clicked.connect(self.close)
        
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        title_layout.addWidget(minimize_btn)
        title_layout.addWidget(close_btn)
        
        main_layout.addWidget(title_bar)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(20)
        content_layout.setContentsMargins(20, 20, 20, 20)
        
        header_frame = QFrame()
        header_layout = QHBoxLayout(header_frame)
        
        # Définition simple de CircularProgress si non importé
        class CircularProgress(QProgressBar):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.setMaximum(100)
                self.setMinimum(0)
                self.setTextVisible(False)
                self.setFixedSize(60, 60)
                self.setStyleSheet("""
                    QProgressBar {
                        border: none;
                        border-radius: 30px;
                        background: #ecf0f1;
                    }
                    QProgressBar::chunk {
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                            stop:0 #3498db, stop:1 #2ecc71);
                        border-radius: 30px;
                    }
                """)

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
        
        content_layout.addWidget(header_frame)
        
        response_frame = QFrame()
        response_frame.setStyleSheet("""
            QFrame {
                background: white;
                border-radius: 15px;
                border: 2px solid #bdc3c7;
            }
        """)
        response_layout = QVBoxLayout(response_frame)
        
        self.assistant_response = QLabel("👋 Bienvenue ! Je suis votre assistant vocal AG7VOC.")
        self.assistant_response.setWordWrap(True)
        self.assistant_response.setAlignment(Qt.AlignCenter)
        self.assistant_response.setStyleSheet("""
            QLabel {
                padding: 20px;
                font-size: 14px;
                color: #2c3e50;
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #ecf0f1, stop: 1 #bdc3c7);
                border-radius: 13px;
            }
        """)
        
        response_layout.addWidget(self.assistant_response)
        content_layout.addWidget(response_frame)
        
        learning_frame = QFrame()
        learning_layout = QVBoxLayout(learning_frame)
        
        learning_header = QHBoxLayout()
        learning_header.addWidget(QLabel("Apprentissage de l'IA"))
        learning_header.addStretch()
        
        self.learning_value = QLabel("0%")
        self.learning_value.setStyleSheet("color: #3498db; font-weight: bold;")
        learning_header.addWidget(self.learning_value)
        
        self.learning_progress = QProgressBar()
        self.learning_progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #bdc3c7;
                border-radius: 6px;
                text-align: center;
                background: #ecf0f1;
                height: 20px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #3498db, stop: 1 #2980b9);
                border-radius: 4px;
            }
        """)
        
        learning_layout.addLayout(learning_header)
        learning_layout.addWidget(self.learning_progress)
        content_layout.addWidget(learning_frame)
        
        buttons_grid = QGridLayout()
        buttons_grid.setSpacing(10)
        
        action_buttons = [
            ("Parler", self.on_listen_command, "#27ae60"),
            ("Entraîner", self.trainAgent, "#9b59b6"),
            ("Suggestion", self.suggere_action, "#f39c12"),
            ("Système", self.show_system_info, "#e74c3c"),
            ("Historique", self.afficher_historique, "#3498db"),
            ("Préférences", self.show_preferences, "#95a5a6"),
            ("Test DQN", self.test_dqn_save, "#e67e22"),
            ("Explorateur", self.open_file_explorer, "#795548")
        ]
        
        for i, (text, callback, color) in enumerate(action_buttons):
            btn = QPushButton(text)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {color};
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 12px;
                    font-weight: bold;
                    font-size: 11px;
                }}
                QPushButton:hover {{
                    background: {color};
                    opacity: 0.9;
                }}
                QPushButton:pressed {{
                    background: {color};
                    opacity: 0.8;
                }}
            """)
            btn.clicked.connect(callback)
            buttons_grid.addWidget(btn, i // 3, i % 3)
        
        content_layout.addLayout(buttons_grid)
        
        output_group = QGroupBox("Journal d'activité")
        output_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #2c3e50;
                border: 2px solid #bdc3c7;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 15px;
            }
        """)
        output_layout = QVBoxLayout(output_group)
        
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setStyleSheet("""
            QTextEdit {
                background: #2c3e50;
                color: #ecf0f1;
                border: 1px solid #34495e;
                border-radius: 8px;
                font-family: 'Consolas', monospace;
                font-size: 10px;
            }
        """)
        output_layout.addWidget(self.output_text)
        
        content_layout.addWidget(output_group)
        main_layout.addWidget(content_widget)
        
        self.setMinimumSize(800, 600)
        self.resize(1000, 700)
        
        self.redirect_stdout()
        
        self.setup_tray_icon()
    
    def setup_tray_icon(self):
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        
        tray_menu = QMenu()
        show_action = QAction("Afficher", self)
        quit_action = QAction("Quitter", self)
        
        show_action.triggered.connect(self.show)
        quit_action.triggered.connect(self.close)
        
        tray_menu.addAction(show_action)
        tray_menu.addAction(quit_action)
        
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()
        
        self.tray_icon.activated.connect(self.tray_icon_activated)
        
    def tray_icon_activated(self, reason):
        if reason == QSystemTrayIcon.DoubleClick:
            self.show()

    def init_timers(self):
        self.metrics_timer = QTimer()
        self.metrics_timer.timeout.connect(self.update_real_time_metrics)
        self.metrics_timer.start(2000)
        
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.animate_status)
        self.status_timer.start(100)
        
    def animate_status(self):
        if self.is_awake:
            current = self.status_indicator.value()  # <-- Ajoute les parenthèses
            if current < 100:
                self.status_indicator.setValue(current + 2)
        else:
            current = self.status_indicator.value()  # <-- Ajoute les parenthèses
            if current > 0:
                self.status_indicator.setValue(current - 2)
    
    def redirect_stdout(self):
        """Redirige stdout vers la zone de sortie"""
        class StdoutRedirector:
            def __init__(self, text_widget):
                self.text_widget = text_widget
                
            def write(self, text):
                if text.strip():
                    from PyQt5.QtCore import QTimer
                    QTimer.singleShot(0, lambda: self.text_widget.append(text.strip()))
                    
            def flush(self):
                pass
                
        sys.stdout = StdoutRedirector(self.output_text)

    def update_real_time_metrics(self):
        """Met à jour les métriques en temps réel avec des données réelles"""
        try:
            if self.agent is None:
                return
                
            if not hasattr(self.agent, 'epsilon') or not hasattr(self.agent, 'memory'):
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
            
        except Exception as e:
            print(f"Erreur mise à jour métriques: {e}")

    def show_system_info(self):
        """Affiche les informations système réelles"""
        try:
            system_info = get_system_info()
            info_text = f"""
=== INFORMATIONS SYSTÈME WINDOWS ===
Système: {system_info['os_name']}
Version: {system_info['version']}
RAM: {system_info['ram']['total']} (Utilisée: {system_info['ram']['percentage']})
Hostname: {system_info['hostname']}
IP: {system_info['ip_address']}
CPU: {system_info['cpu_usage']}
Processus: {system_info['process_count']}
            """
            
            if 'disks' in system_info:
                info_text += "\n=== DISQUES ==="
                for device, disk in system_info['disks'].items():
                    info_text += f"\n{device}: {disk['free']} libre sur {disk['total']} ({disk['percentage']} utilisé)"
            
            self.output_text.append(info_text)
            speak("Voici les informations de votre système Windows.")
            
        except Exception as e:
            self.output_text.append(f"Erreur récupération infos système: {e}")

    def on_listen_command(self):
        """Déclenché quand on clique sur le bouton Parler"""
        if not self.is_awake:
            assistant_signals.update_status.emit("État : En veille", "#e74c3c", "🔴")
            self.afficher_message("<i>L'assistant est en veille. Dites le mot-clé pour le réveiller.</i>")
            speak("Je suis en veille. Dites assistant ou réveille-toi pour me réveiller.")
            assistant_signals.update_status.emit("État : Écoute du mot-clé", "#f39c12", "🟡")
            self.assistant_response.setText("Écoute du mot-clé...")
            
            threading.Thread(target=self.listen_for_wake_word, daemon=True).start()
            
        else:
            assistant_signals.update_status.emit("État : Écoute", "#f39c12", "🟡")
            self.assistant_response.setText("Assistant : J'écoute...")
            threading.Thread(target=self.process_command, daemon=True).start()

    def listen_for_wake_word(self):
        try:
            print("Écoute du mot de réveil...")
            commande = listen()
            print(f"Commande reçue: '{commande}'")
            
            if commande:
                commande_lower = commande.lower()
                wake_words = voice_prefs.get_preference("wake_words")
                print(f"Mots de réveil recherchés: {wake_words}")
                
                detected = any(wake_word in commande_lower for wake_word in wake_words)
                print(f"Mot de réveil détecté: {detected}")
                
                if detected:
                    self.is_awake = True
                    assistant_signals.update_status.emit("État : Réveillé", "#2ecc71")
                    self.afficher_message("<b>Assistant réveillé !</b>")
                    speak("Je suis réveillé. Comment puis-je vous aider ?")
                    self.listen_for_commands()
                else:
                    self.afficher_message(f"<i>Commande ignorée: '{commande}'</i>")
            else:
                print("Aucune commande détectée")
                
        except Exception as e:
            print(f"Erreur écoute mot de réveil: {e}")

    def test_microphone(self):
        
        import speech_recognition as sr
        """Teste le microphone et la reconnaissance vocale"""
        try:
            print("Test du microphone...")
            speak("Test du microphone. Parlez maintenant.")
            
            r = sr.Recognizer()
            with sr.Microphone() as source:
                print("Calibration...")
                r.adjust_for_ambient_noise(source, duration=1)
                print("Parlez maintenant...")
                audio = r.listen(source, timeout=5, phrase_time_limit=3)
                
                try:
                    text = r.recognize_google(audio, language="fr-FR")
                    print(f"Test réussi: '{text}'")
                    speak(f"J'ai entendu: {text}")
                    return True
                except sr.UnknownValueError:
                    print("Je n'ai rien compris")
                    speak("Je n'ai rien entendu")
                    return False
                    
        except Exception as e:
            print(f"Erreur test microphone: {e}")
            return False
    
    def process_command(self):
        try:
            commande = listen()
            if commande:
                self.afficher_message(f"<b>Vous :</b> {commande}")
                if any(w in commande.lower() for w in ["dors", "va en veille", "arrête d'écouter"]):
                    self.is_awake = False
                    assistant_signals.update_status.emit("État : En veille", "#e74c3c")
                    self.afficher_message("<b>Assistant en veille. Dites 'assistant' pour le réveiller.</b>")
                    speak("Je passe en veille. Dites 'assistant' pour me réveiller.")
                    self.wait_for_wake_word()
                    return
                
                old_stdout = sys.stdout
                sys.stdout = self.output_text
                
                try:
                    self.traiter_commande(commande)
                finally:
                    sys.stdout = old_stdout
                    
            else:
                assistant_signals.update_status.emit("État : Réveillé", "#2ecc71")
                self.afficher_message("<i>Aucune commande détectée.</i>")
        except Exception as e:
            print(f"Erreur traitement commande: {e}")

    def traiter_commande(self, commande):
        """Traite une commande avec intégration DQN complète"""
        try:
            start_time = time.time()
            
            if self.agent is None:
                # Mode dégradé sans DQN
                process_voice_command(commande)
                return
            
            # Analyse de la commande
            top_intents = get_top_intents_spacy_similarity(commande)
            if not top_intents:
                speak("Je n'ai pas compris. Pouvez-vous répéter ?")
                return
                
            best_intent, best_score = top_intents[0]
            
            if best_score < 0.6:
                speak("Je ne suis pas sûr d'avoir compris. Pouvez-vous préciser ?")
                return

            old_stdout = sys.stdout
            sys.stdout = self.output_text
            
            try:
                process_voice_command(commande, forced_intent=best_intent)
                self.last_command_success = 1.0
            except Exception as e:
                self.output_text.append(f"Erreur: {e}")
                self.last_command_success = 0.0
            finally:
                sys.stdout = old_stdout
                
            self.update_real_time_metrics()
            self.afficher_historique()
            
        except Exception as e:
            self.output_text.append(f"Erreur traitement commande: {e}")
            self.last_command_success = 0.0
    
    def test_dqn_save(self):
        """Teste la sauvegarde DQN avec des données valides"""
        try:
            if self.agent is None:
                print("Agent DQN non initialisé")
                return
                
            test_state = np.array([[1, 2, 3, 4, 5]])  # Format 2D
            test_next_state = np.array([[1, 2, 3, 4, 6]])  # Format 2D
            
            self.agent.remember(test_state, 0, 10, test_next_state, False)
            print("Expérience test sauvegardée (format 2D)")
            
            if os.path.exists(self.agent.memory_file):
                with open(self.agent.memory_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    print(f"Contenu: {content}")
                    
        except Exception as e:
            print(f"Erreur test DQN: {e}")
    
    def trainAgent(self):
        """Entraîne l'agent DQN avec les données réelles"""
        try:
            if self.agent is None:
                self.output_text.append("Agent DQN non initialisé")
                return
                
            if len(self.agent.memory) > self.batch_size:
                self.output_text.append("Entraînement de l'IA en cours...")
                
                print(f"Format mémoire: {len(self.agent.memory)} expériences")
                if self.agent.memory:
                    state_sample = self.agent.memory[0][0]
                    print(f"Format état: {state_sample.shape}")
                
                loss = self.agent.replay(self.batch_size)
                
                if loss is not None:
                    self.output_text.append(f"Entraînement terminé. Perte: {loss:.4f}")
                    self.afficher_message("<i>Entraînement terminé. L'assistant est maintenant plus intelligent !</i>")
                    
                    progress = min(100, int(100 * len(self.agent.memory) / 2000))
                    self.learning_progress.setValue(progress)
                    
                else:
                    self.output_text.append("Erreur lors de l'entraînement")
                    
            else:
                needed = self.batch_size - len(self.agent.memory)
                self.afficher_message(f"<i>Pas assez de données ({len(self.agent.memory)}/{self.batch_size}). Il manque {needed} interactions.</i>")
                self.output_text.append(f"Données insuffisantes: {len(self.agent.memory)}/{self.batch_size}")
                
        except Exception as e:
            self.output_text.append(f"Erreur entraînement: {e}")
            self.afficher_message("<i>Erreur lors de l'entraînement de l'IA.</i>")
    
    # def update_real_time_metrics(self):
    #     """Met à jour les métriques DQN en temps réel"""
    #     try:
    #         if self.agent is None:
    #             return
                
    #         metrics = self.agent.get_training_metrics()
            
    #         dqn_metrics = {
    #             'q_score': f"{self.agent.epsilon:.3f}",
    #             'exploration_rate': f"{self.agent.epsilon * 100:.1f}%",
    #             'memory_size': f"{metrics['memory_size']:,}",
    #             'accuracy': f"{min(100, int(100 * metrics['memory_size'] / 5000))}%"
    #         }
            
    #         assistant_signals.update_status.emit("État : En veille", "#e74c3c", "🔴")
            
    #         progress = min(100, int(100 * metrics['memory_size'] / 5000))
    #         self.learning_progress.setValue(progress)
    #         self.learning_value.setText(f"{progress}%")
    #         assistant_signals.update_status.emit("État : Écoute du mot-clé", "#f39c12", "🟡")
    
    #         learning_stats = {
    #             'progress': progress,
    #             'metrics': dqn_metrics,
    #             'performance_data': self.performance_data
    #         }
    #         assistant_signals.update_learning_stats.emit(learning_stats)
            
    #     except Exception as e:
    #         print(f"Erreur mise à jour métriques: {e}")

    def update_learning_stats_safe(self, stats):
        """Mise à jour thread-safe des statistiques d'apprentissage"""
        try:
            progress = stats.get('progress', 0)
            self.learning_progress.setValue(progress)
            self.learning_value.setText(f"{progress}%")
            
            if 'metrics' in stats:
                self.update_metrics_safe(stats['metrics'])
                
        except Exception as e:
            print(f"Erreur mise à jour stats apprentissage: {e}")
    
    def show_notification_safe(self, title, message):
        """Affiche une notification de système de manière thread-safe"""
        try:
            if hasattr(self, 'tray_icon') and self.tray_icon:
                self.tray_icon.showMessage(title, message, QSystemTrayIcon.Information, 3000)
        except Exception as e:
            print(f"Erreur notification: {e}")
            
    
    
    def listen_for_commands(self):
        """Écoute continue des commandes"""
        self.is_awake = True
        assistant_signals.update_status.emit("État : En écoute", "#2ecc71")
        self.afficher_message("<b>Assistant en écoute active</b>")
        print("Mode écoute active activé")
        
        while self.is_awake:
            try:
                print("En attente de commande...")
                commande = listen()
                
                if commande:
                    print(f"Commande reçue: '{commande}'")
                    self.afficher_message(f"<b>Vous :</b> {commande}")
                    
                    commande_lower = commande.lower()
                    sleep_words = voice_prefs.get_preference("sleep_words")
                    
                    if any(sleep_word in commande_lower for sleep_word in sleep_words):
                        self.is_awake = False
                        assistant_signals.update_status.emit("État : En veille", "#e74c3c")
                        self.afficher_message("<b>Assistant en veille. Dites 'assistant' pour le réveiller.</b>")
                        speak("Je passe en veille. Dites 'assistant' pour me réveiller.")
                        break
                    
                    old_stdout = sys.stdout
                    sys.stdout = self.output_text
                    
                    try:
                        self.traiter_commande(commande)
                    finally:
                        sys.stdout = old_stdout
                        
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Erreur écoute continue: {e}")
                time.sleep(1)

    
    def wait_for_wake_word(self):
        """Attente du mot de réveil"""
        assistant_signals.update_status.emit("État : En veille", "#e74c3c", "🔴")
        self.afficher_message("<i>L'assistant est en veille. Dites un mot-clé pour le réveiller.</i>")
        speak("Je suis en veille. Dites 'assistant' ou 'réveille-toi' pour me réveiller.")
        assistant_signals.update_status.emit("État : Écoute du mot-clé", "#f39c12", "🟡")
        self.assistant_response.setText("Écoute du mot-clé...")
        
        threading.Thread(target=self.listen_for_wake_word, daemon=True).start()

    # def trainAgent(self):
    #     """Entraîne l'agent DQN avec des données réelles"""
    #     try:
    #         if len(self.agent.memory) > self.batch_size:
    #             self.output_text.append("Début de l'entraînement de l'IA...")
    #             loss = self.agent.replay(self.batch_size)
                
    #             if loss is not None:
    #                 self.output_text.append(f"Entraînement terminé. Perte: {loss:.4f}")
    #                 self.afficher_message("<i>Entraînement terminé. L'assistant est maintenant plus intelligent !</i>")
                    
    #                 # Mettre à jour la progression réelle
    #                 progress = min(100, int(100 * len(self.agent.memory) / 2000))
    #                 self.learning_progress.setValue(progress)
                    
    #                 self.update_real_time_metrics()
    #             else:
    #                 self.output_text.append("Erreur lors de l'entraînement")
                    
    #         else:
    #             needed = self.batch_size - len(self.agent.memory)
    #             self.afficher_message(f"<i>Pas assez de données ({len(self.agent.memory)}/{self.batch_size}). Il manque {needed} interactions.</i>")
    #             self.output_text.append(f"Données insuffisantes pour l'entraînement: {len(self.agent.memory)}/{self.batch_size}")
                
    #     except Exception as e:
    #         self.output_text.append(f"Erreur entraînement: {e}")
    #         self.afficher_message("<i>Erreur lors de l'entraînement de l'IA.</i>")
    
    
    def show_suggestions_dialog(self, suggestions, original_command):
        """Affiche une boîte de dialogue avec des suggestions"""
        try:
            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QFrame
            
            dialog = QDialog(self)
            dialog.setWindowTitle("Suggestions de commandes")
            dialog.setMinimumWidth(400)
            
            layout = QVBoxLayout(dialog)
            
            # Message
            layout.addWidget(QLabel(f"Je n'ai pas compris: '{original_command}'\n\nVoulez-vous dire :"))
            
            # Boutons pour chaque suggestion
            self.suggestion_actions = []
            
            for i, (kw, label, intent_name, similarity) in enumerate(suggestions, 1):
                btn = QPushButton(f"{i}. {label}")
                btn.setStyleSheet("""
                    QPushButton {
                        text-align: left;
                        padding: 10px;
                        margin: 2px;
                        background: #e3f2fd;
                        border: 1px solid #bbdefb;
                        border-radius: 4px;
                    }
                    QPushButton:hover {
                        background: #bbdefb;
                    }
                """)
                
                action_data = {
                    "intent": intent_name,
                    "original_command": original_command,
                    "suggested_keyword": kw
                }
                self.suggestion_actions.append(action_data)
                
                btn.clicked.connect(lambda checked, idx=i-1: self.use_suggestion(idx))
                layout.addWidget(btn)
            
            cancel_btn = QPushButton("Annuler")
            cancel_btn.clicked.connect(dialog.reject)
            layout.addWidget(cancel_btn)
            
            dialog.exec_()
            
        except Exception as e:
            print(f"Erreur affichage suggestions: {e}")
            
    def use_suggestion(self, suggestion_index):
        """Utilise la suggestion sélectionnée"""
        try:
            if 0 <= suggestion_index < len(self.suggestion_actions):
                action_data = self.suggestion_actions[suggestion_index]
                
                for widget in self.findChildren(QDialog):
                    if widget.windowTitle() == "Suggestions de commandes":
                        widget.accept()
                
                speak(f"Exécution de: {action_data['suggested_keyword']}")
                
                self.traiter_commande_avec_intention(
                    action_data['original_command'], 
                    action_data['intent']
                )
                
        except Exception as e:
            print(f"Erreur utilisation suggestion: {e}")
    
    def traiter_commande_avec_intention(self, commande, intention):
        """Traite une commande avec une intention forcée"""
        try:
            old_stdout = sys.stdout
            sys.stdout = self.output_text
            
            try:
                process_voice_command(commande, forced_intent=intention)
            finally:
                sys.stdout = old_stdout
                
        except Exception as e:
            self.output_text.append(f"Erreur traitement commande: {e}")
        
    def suggere_action(self):
        """Suggère une action basée sur l'état actuel"""
        try:
            if self.agent is None:
                self.output_text.append("Agent DQN non initialisé")
                return
                
            last_cmd_idx = 0
            if self.history:
                last_cmd = self.history[-1][1] if len(self.history[-1]) > 1 else ""
                if last_cmd in INTENT_LABELS_FR:
                    last_cmd_idx = list(INTENT_LABELS_FR.keys()).index(last_cmd)
            
            heure = int(time.strftime("%H"))
            apps_ouverts = len(psutil.pids()) // 100
            last_error = 0
            humeur = 0
            
            state = get_current_state(last_cmd_idx, heure, apps_ouverts, last_error, humeur)
            action_idx = self.agent.act(state)
            action = ACTIONS[action_idx]
            
            action_translations = {
                "executer_commande": "Exécuter une commande",
                "suggérer_pause": "Suggérer une pause",
                "proposer_fermeture_onglets": "Proposer de fermer les onglets inutilisés",
                "proposer_musique": "Proposer d'écouter de la musique",
                "demander_precisions": "Demander des précisions",
                "aucune_action": "Ne rien faire"
            }
            
            action_fr = action_translations.get(action, action)
            self.afficher_message(f"Suggestion IA : {action_fr}")
            speak(f"Je vous suggère : {action_fr}")
            
            self.output_text.append(f"Suggestion IA: {action} (État: {state})")
            
        except Exception as e:
            self.output_text.append(f"Erreur suggestion: {e}")
            self.afficher_message("<i>Erreur lors de la génération de suggestion.</i>")

    def show_preferences(self):
        """Affiche l'onglet des préférences"""
        try:
            # Importer ou définir VoicePreferencesTab si non déjà importé
            try:
                from voice_preferences import VoicePreferencesTab
            except ImportError:
                # Définition minimale si l'import échoue
                from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
                class VoicePreferencesTab(QWidget):
                    def __init__(self):
                        super().__init__()
                        layout = QVBoxLayout(self)
                        layout.addWidget(QLabel("Préférences vocales (définition minimale)"))

            # Créer l'onglet des préférences s'il n'existe pas
            if not hasattr(self, 'preferences_tab'):
                self.preferences_tab = VoicePreferencesTab()
            
            # Créer un dialog pour afficher les préférences
            dialog = QDialog(self)
            dialog.setWindowTitle("Préférences Vocales")
            dialog.setModal(True)
            dialog.setMinimumSize(600, 500)
            
            layout = QVBoxLayout(dialog)
            layout.addWidget(self.preferences_tab)
            
            # Boutons de fermeture
            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)
            
            dialog.exec_()
            
        except Exception as e:
            self.output_text.append(f"Erreur ouverture préférences: {e}")
            QMessageBox.warning(self, "Erreur", "Impossible d'ouvrir les préférences")
    
    def simulateInteraction(self):
        """Simulation d'interaction avec des données réelles"""
        self.interaction_timer.stop()
        
        # Messages basés sur l'état réel
        current_time = time.strftime("%H:%M")
        memory_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent()
        
        interactions = [
            f"J'ai terminé votre demande à {current_time}. CPU: {cpu_usage}%, RAM: {memory_usage}%",
            "Voici les informations demandées. Besoin de précisions supplémentaires ?",
            f"Traitement terminé. Système: {cpu_usage}% CPU, {memory_usage}% RAM utilisée",
            f"J'ai accompli la tâche. Performance: {cpu_usage}% CPU",
            f"Opération réussie à {current_time}. État système: {memory_usage}% RAM utilisée"
        ]
        
        self.assistant_response.setText(random.choice(interactions))

    def choisir_disque_et_fichier(self):
        """Interface pour choisir un disque et un fichier"""
        try:
            from ag7voc import list_drives
            drives = list_drives()
            
            self.output_text.append("=== SELECTION DE DISQUE ===")
            self.output_text.append("Disques disponibles:")
            for idx, drive in enumerate(drives, 1):
                self.output_text.append(f"{idx}. {drive}")
            
            self.afficher_message("<b>Disques disponibles :</b>")
            for idx, drive in enumerate(drives, 1):
                self.afficher_message(f"{idx}. {drive}")
            
            speak("Veuillez dire le numéro du disque à explorer.")
            self.output_text.append("En attente de la sélection du disque...")
            
            # Cette partie nécessiterait une interface graphique pour la sélection
            # Pour l'instant, on logue seulement
            self.output_text.append("Sélection de disque - fonctionnalité à implémenter")
            
        except Exception as e:
            self.output_text.append(f"Erreur sélection disque: {e}")

    def show_intent_dialog(self, top_intents):
        """Affiche un dialogue pour choisir une intention"""
        try:
            msg = QMessageBox(self)
            msg.setWindowTitle("Choix d'intention")
            
            text = "Veuillez choisir une intention parmi les suivantes :\n\n"
            for idx, (intent, score) in enumerate(top_intents, 1):
                label_fr = INTENT_LABELS_FR.get(intent, intent.replace('_', ' '))
                text += f"{idx}. {label_fr} (confiance: {score:.2f})\n"
            
            msg.setText(text)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            
            self.output_text.append("Dialogue d'intention affiché")
            
        except Exception as e:
            self.output_text.append(f"Erreur dialogue intention: {e}")

    def afficher_historique(self):
        """Affiche l'historique réel des commandes"""
        try:
            from ag7voc import get_history
            history = get_history()
            
            if not hasattr(self, 'history_widget'):
                self.output_text.append("Widget historique non disponible")
                return
                
            self.history_widget.clear()
            self.output_text.append("HISTORIQUE DES COMMANDES")
            
            for idx, (cmd, intent) in enumerate(history, 1):
                label_fr = INTENT_LABELS_FR.get(intent, intent.replace('_', ' ')) if intent else "Inconnue"
                item_text = f"{idx}. {cmd} [{label_fr}]"
                
                item = QListWidgetItem(item_text)
                self.history_widget.addItem(item)
                self.output_text.append(item_text)
            
            self.history_widget.scrollToBottom()
            
        except Exception as e:
            self.output_text.append(f"Erreur chargement historique: {e}")


    def initInteractionTimer(self):
        """Initialise le timer d'interaction"""
        self.interaction_timer = QTimer()
        self.interaction_timer.timeout.connect(self.simulateInteraction)

    def closeEvent(self, event):
        """Gère la fermeture de l'application"""
        try:
            self.is_awake = False
            self.metrics_timer.stop()
            self.status_timer.stop()
            
            if self.agent:
                self.agent.save_model()
                self.output_text.append("Poids du réseau neuronal sauvegardés")
            
            self.tray_icon.hide()
            event.accept()
            
        except Exception as e:
            self.output_text.append(f"Erreur fermeture: {e}")
            event.accept()
            
class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        
    def write(self, text):
        if text.strip():
            text_str = str(text).strip()
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(0, lambda: self.text_widget.append(text_str))
            
    def flush(self):
        pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    app.setStyle('Fusion')
    
    assistant = VirtualAssistant()
    assistant.show()
    
    assistant.output_text.append("=" * 50)
    assistant.output_text.append("AG7VOC Assistant Vocal Démarré")
    assistant.output_text.append(f"Heure: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    assistant.output_text.append(f"Processus: {len(psutil.pids())}")
    assistant.output_text.append(f"Mémoire: {psutil.virtual_memory().percent}% utilisée")
    assistant.output_text.append("=" * 50)
    assistant.output_text.append("En attente du mot de réveil...")
    assistant.output_text.append("Dites 'assistant' pour commencer")
    
    sys.exit(app.exec_())