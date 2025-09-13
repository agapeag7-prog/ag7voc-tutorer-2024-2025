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
from PyQt5.QtGui import QMovie
import urllib.request

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
        
        self.is_awake = False
        self.performance_data = []
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
        # Démarre l'écoute du mot-clé dès le lancement
        threading.Thread(target=self.listen_loop, daemon=True).start()

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
            # ("Parler", self.on_listen_command, "#27ae60"),  # <-- À supprimer ou commenter
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
            
    
    
    def listen_loop(self):
        """Boucle principale : écoute du mot-clé puis des commandes tant que l'app est ouverte"""
        while True:
            self.wait_for_wake_word_blocking()
            self.listen_for_commands_blocking()

    def wait_for_wake_word_blocking(self):
        """Écoute bloquante du mot-clé (veille)"""
        self.is_awake = False
        assistant_signals.update_status.emit("État : En veille", "#e74c3c", "🔴")
        self.afficher_message("<i>L'assistant est en veille. Dites un mot-clé pour le réveiller.</i>")
        speak("Je suis en veille. Dites 'assistant' ou 'réveille-toi' pour me réveiller.")
        self.assistant_response.setText("Écoute du mot-clé...")
        while not self.is_awake:
            commande = listen()
            if commande:
                commande_lower = commande.lower()
                wake_words = voice_prefs.get_preference("wake_words")
                if any(wake_word in commande_lower for wake_word in wake_words):
                    self.is_awake = True
                    assistant_signals.update_status.emit("État : Réveillé", "#2ecc71")
                    self.afficher_message("<b>Assistant réveillé !</b>")
                    speak("Je suis réveillé. Comment puis-je vous aider ?")
                else:
                    self.afficher_message(f"<i>Commande ignorée: '{commande}'</i>")
            time.sleep(0.5)

    def listen_for_commands_blocking(self):
        """Écoute bloquante des commandes tant que l'assistant est réveillé"""
        assistant_signals.update_status.emit("État : En écoute", "#2ecc71")
        self.afficher_message("<b>Assistant en écoute active</b>")
        while self.is_awake:
            commande = listen()
            if commande:
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

    def initUI(self):
        self.setWindowTitle('AG7VOC - Assistant Vocal Intelligent')
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))

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
            # ("Parler", self.on_listen_command, "#27ae60"),  # <-- À supprimer ou commenter
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
       
    def show_preferences(self):
        """Affiche les préférences vocales actuelles dans la zone de sortie"""
        try:
            prefs = voice_prefs.preferences if hasattr(voice_prefs, "preferences") else {}
            self.output_text.append("=== Préférences vocales ===")
            for key, value in prefs.items():
                self.output_text.append(f"{key}: {value}")
        except Exception as e:
            self.output_text.append(f"Erreur affichage préférences: {e}")
        
    def suggere_action(self):
        """Affiche les suggestions IA pour la dernière commande vocale"""
        try:
            if self.history:
                last_command = self.history[-1]
            else:
                last_command = ""
            if not last_command:
                self.output_text.append("Aucune commande à suggérer.")
                return
            suggestions = get_top_intents_spacy_similarity(last_command)
            if suggestions:
                self.output_text.append("Suggestions IA :")
                for intent, score in suggestions:
                    label = INTENT_LABELS_FR.get(intent, intent)
                    self.output_text.append(f"- {label} (score: {score:.2f})")
            else:
                self.output_text.append("Aucune suggestion disponible.")
        except Exception as e:
            self.output_text.append(f"Erreur suggestion IA: {e}")
    
    def afficher_historique(self):
        """Affiche l'historique des commandes dans la zone de sortie"""
        try:
            self.output_text.append("=== Historique des commandes ===")
            if self.history:
                for i, cmd in enumerate(self.history, 1):
                    self.output_text.append(f"{i}. {cmd}")
            else:
                self.output_text.append("Aucune commande enregistrée.")
        except Exception as e:
            self.output_text.append(f"Erreur affichage historique: {e}")
    
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
            
    
    
    def listen_loop(self):
        """Boucle principale : écoute du mot-clé puis des commandes tant que l'app est ouverte"""
        while True:
            self.wait_for_wake_word_blocking()
            self.listen_for_commands_blocking()

    def wait_for_wake_word_blocking(self):
        """Écoute bloquante du mot-clé (veille)"""
        self.is_awake = False
        assistant_signals.update_status.emit("État : En veille", "#e74c3c", "🔴")
        self.afficher_message("<i>L'assistant est en veille. Dites un mot-clé pour le réveiller.</i>")
        speak("Je suis en veille. Dites 'assistant' ou 'réveille-toi' pour me réveiller.")
        self.assistant_response.setText("Écoute du mot-clé...")
        while not self.is_awake:
            commande = listen()
            if commande:
                commande_lower = commande.lower()
                wake_words = voice_prefs.get_preference("wake_words")
                if any(wake_word in commande_lower for wake_word in wake_words):
                    self.is_awake = True
                    assistant_signals.update_status.emit("État : Réveillé", "#2ecc71")
                    self.afficher_message("<b>Assistant réveillé !</b>")
                    speak("Je suis réveillé. Comment puis-je vous aider ?")
                else:
                    self.afficher_message(f"<i>Commande ignorée: '{commande}'</i>")
            time.sleep(0.5)

    def listen_for_commands_blocking(self):
        """Écoute bloquante des commandes tant que l'assistant est réveillé"""
        assistant_signals.update_status.emit("État : En écoute", "#2ecc71")
        self.afficher_message("<b>Assistant en écoute active</b>")
        while self.is_awake:
            commande = listen()
            if commande:
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