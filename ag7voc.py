import sqlite3
import os
import logging
import spacy
import dateparser
import speech_recognition as sr
import numpy as np
import torch
import silero_vad
import vosk
import sounddevice as sd
import queue
import webbrowser
import subprocess
import platform
import re
import sys
import string
import json
import time
import pyttsx3
import winreg
import psutil
import socket
from pathlib import Path
from datetime import datetime

from file_explorer import open_file_explorer as file_explorer_open, VoiceControlledFileExplorer

from ai_engine import DQNAgent, ACTIONS, compute_reward, get_current_state, analyze_user_sentiment

from voice_preferences import voice_prefs

from voice_manager import voice_manager

from PyQt5.QtCore import QObject, pyqtSignal

FORBIDDEN_FOLDERS = [
    "C:\\Windows",
    "C:\\Program Files",
    "C:\\Program Files (x86)",
    "C:\\Users\\Default",
    "C:\\$Recycle.Bin",
    "C:\\System Volume Information"
]

class AssistantSignals(QObject):
    show_suggestions = pyqtSignal(object, object)
    update_display = pyqtSignal(object)
    update_status = pyqtSignal(str, str, str)
    add_history_item = pyqtSignal(str, str)
    update_metrics = pyqtSignal(dict)
    update_learning_stats = pyqtSignal(dict)
    show_notification = pyqtSignal(str, str)

assistant_signals = AssistantSignals()

try:
    from ai_engine import DQNAgent, ACTIONS, compute_reward, get_current_state, analyze_user_sentiment
    DQN_AVAILABLE = True
    print("Module DQN chargé avec succès")
except ImportError as e:
    print(f"Module DQN non disponible: {e}")
    DQN_AVAILABLE = False
    
    DQNAgent = None
    ACTIONS = []
    def compute_reward(*args): return 0
    def get_current_state(*args): return [0, 0, 0, 0, 0]
    def analyze_user_sentiment(text): return 0.5


try:
    from file_explorer import open_file_explorer, VoiceControlledFileExplorer
    select_drive = None  # À remplacer si tu as une fonction dédiée
except ImportError as e:
    print(f"Import explorateur fichiers: {e}")
    open_file_explorer = None
    select_drive = None

FRONT_DISPLAY_CALLBACK = None

def set_front_display_callback(callback):
    global FRONT_DISPLAY_CALLBACK
    FRONT_DISPLAY_CALLBACK = callback

def display_on_front(msg):
    """Affiche un message de manière thread-safe"""
    if FRONT_DISPLAY_CALLBACK:
        try:
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(0, lambda: FRONT_DISPLAY_CALLBACK(msg))
        except Exception as e:
            print(f"Erreur affichage message: {e}")

logging.basicConfig(
    filename="assistant.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

DB_PATH = "assistant.db"
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "history.json")

try:
    nlp = spacy.load("fr_core_news_md")
except:
    print("Modèle spaCy français non trouvé. Installation: python -m spacy download fr_core_news_md")
    nlp = None

dqn_agent = DQNAgent(5, len(ACTIONS))

INTENT_LABELS_FR = {
    "add_event": "Ajouter un événement",
    "show_events": "Afficher les événements",
    "delete_event": "Supprimer un événement",
    "modify_event": "Modifier un événement",
    "read_file": "Lire un fichier",
    "write_file": "Écrire un fichier",
    "delete_file": "Supprimer un fichier",
    "create_folder": "Créer un dossier",
    "list_files": "Lister les fichiers",
    "rename_file": "Renommer un fichier",
    "move_file": "Déplacer un fichier",
    "search_files": "Recherche de fichiers/dossiers",
    "get_time": "Obtenir l'heure",
    "get_date": "Obtenir la date",
    "show_help": "Aide",
    "launch_app": "Lancer une application",
    "shutdown": "Éteindre",
    "restart": "Redémarrer",
    "lock": "Verrouiller",
    "search_web": "Chercher sur internet",
    "send_email": "Envoyer un email",
    "weather": "Météo",
    "list_drives": "Lister les lecteurs",
    "historique": "Historique",
    "system_info": "Informations système"
}

INTENTS = {
    "add_event": [
        "ajouter", "créer", "noter", "planifier", "programmer", "prévoir", "inscrire", "enregistrer",
        "nouvel événement", "nouveau rendez-vous", "prévoir un rendez-vous", "ajoute un événement", "ajoute un rendez-vous",
        "prévoir une réunion", "prévoir une tâche", "prévoir une activité", "prévoir une sortie", "prévoir un appel",
        "rendez-vous demain", "rendez-vous à 15h", "rendez-vous avec Alice", "rendez-vous professionnel", "rendez-vous personnel",
        "ajoute une réunion", "ajoute une tâche", "ajoute une activité", "ajoute une sortie", "ajoute un appel"
    ],
    "show_events": [
        "afficher", "voir", "liste", "lister", "montrer", "quels sont", "quels événements", "mes événements", "mes rendez-vous",
        "affiche les événements", "montre les rendez-vous", "qu'ai-je aujourd'hui", "qu'ai-je demain",
        "quels sont mes rendez-vous", "quels sont mes événements", "quels sont mes tâches", "quels sont mes réunions",
        "agenda", "calendrier", "planning", "planning de la semaine", "planning du jour"
    ],
    "delete_event": [
        "supprimer", "enlever", "retirer", "effacer", "annuler", "supprime l'événement", "supprime le rendez-vous", "annule",
        "annule le rendez-vous", "annule la réunion", "annule la tâche", "annule l'activité", "annule la sortie", "annule l'appel"
    ],
    "modify_event": [
        "modifier", "changer", "éditer", "corriger", "mettre à jour", "modifie l'événement", "modifie le rendez-vous",
        "modifie la réunion", "modifie la tâche", "modifie l'activité", "modifie la sortie", "modifie l'appel",
        "change l'heure du rendez-vous", "change la date de la réunion", "change le lieu de la sortie"
    ],
    "read_file": [
        "lire", "ouvrir", "afficher", "montre le contenu", "montre le fichier", "lis le fichier", "ouvre le fichier",
        "affiche le document", "affiche le texte", "affiche le rapport", "affiche la note", "affiche la facture"
    ],
    "write_file": [
        "écrire", "ajouter", "créer", "enregistrer", "sauvegarder", "écris dans le fichier", "ajoute au fichier",
        "écris dans le document", "ajoute au document", "écris une note", "écris un rapport", "écris une facture"
    ],
    "delete_file": [
        "supprimer", "effacer", "retirer", "détruire", "supprime le fichier", "efface le fichier",
        "supprime le document", "efface le document", "supprime la note", "supprime le rapport", "supprime la facture"
    ],
    "create_folder": [
        "créer", "ajouter", "nouveau dossier", "crée un dossier", "ajoute un dossier", "nouveau répertoire",
        "crée un répertoire", "crée un dossier de travail", "crée un dossier personnel", "crée un dossier projet"
    ],
    "list_files": [
        "lister", "afficher", "voir", "liste des fichiers", "montre les fichiers", "quels fichiers", "quels dossiers",
        "affiche les documents", "affiche les notes", "affiche les rapports", "affiche les factures"
    ],
    "rename_file": [
        "renommer", "changer le nom", "modifie le nom", "renomme le fichier", "renommer le dossier",
        "renomme le document", "renomme la note", "renomme le rapport", "renomme la facture"
    ],
    "move_file": [
        "déplacer", "transférer", "bouger", "déplace le fichier", "déplace dans", "transfère dans",
        "déplace le document", "déplace la note", "déplace le rapport", "déplace la facture"
    ],
    "get_time": [
        "quelle heure", "donne l'heure", "il est quelle heure", "heure actuelle", "heure",
        "donne-moi l'heure", "quelle est l'heure", "heure précise", "heure exacte"
    ],
    "get_date": [
        "quelle date", "donne la date", "date du jour", "on est quel jour", "date",
        "donne-moi la date", "quelle est la date", "date précise", "date exacte"
    ],
    "show_help": [
        "aide", "que peux-tu faire", "aide-moi", "liste des commandes", "comment ça marche", "quelles sont tes fonctions",
        "quelles sont tes capacités", "quels sont tes services", "quels sont tes outils", "quels sont tes modules",
        "qu'est ce que tu peux faire", "qu'est ce que tu sais faire", "montre-moi l'aide", "affiche l'aide"
    ],
    "launch_app": [
        "ouvre", "ouvrir", "lance", "lancer", "démarre", "démarrer", "start", "open",
        "ouvre la calculatrice", "ouvre le bloc-notes", "ouvre notepad", "ouvre paint", "ouvre navigateur",
        "ouvre chrome", "ouvre edge", "ouvre internet", "lance le navigateur", "lance chrome", "lance edge",
        "lance internet", "ouvre excel", "ouvre word", "ouvre powerpoint", "ouvre vscode", "ouvre spotify",
        "ouvre git", "lance git", "ouvre excel", "lance excel", "ouvre word", "lance word", "ouvre powerpoint", 
        "lance powerpoint", "ouvre vscode", "lance vscode", "ouvre spotify", "lance spotify", "ouvre le terminal",
        "lance le terminal"
    ],
    "shutdown": [
        "éteins l'ordinateur", "arrête l'ordinateur", "ferme la session", "éteindre", "arrêter", "shutdown",
        "éteins le pc", "arrête le pc", "ferme le pc", "éteins la machine", "arrête la machine"
    ],
    "restart": [
        "redémarre l'ordinateur", "redémarrer", "restart", "relance l'ordinateur",
        "redémarre le pc", "relance le pc", "redémarre la machine", "relance la machine"
    ],
    "lock": [
        "verrouille l'ordinateur", "verrouiller", "lock", "verrouille la session",
        "verrouille le pc", "verrouille la machine", "verrouille l'écran"
    ],
    "search_web": [
        "cherche sur internet", "recherche", "trouve", "cherche", "recherche sur google", "trouve sur internet",
        "recherche sur le web", "cherche sur le web", "recherche sur bing", "cherche sur bing", "recherche sur yahoo", "cherche sur yahoo"
    ],
    "send_email": [
        "envoie un mail", "envoie un email", "envoie mail", "envoie un courriel", "envoie un message", "envoie un e-mail",
        "envoie un courrier", "envoie une lettre", "envoie un sms", "envoie un texto", "envoie courriel"
    ],
    "weather": [
        "météo", "quel temps", "fait-il beau", "quel temps fait-il", "donne la météo", "prévisions météo",
        "prévisions du temps", "temps aujourd'hui", "temps demain", "temps ce week-end", "temps ce soir"
    ],
    "list_drives": [
        "lister les lecteurs", "afficher les lecteurs", "voir les lecteurs", "liste des lecteurs", "montre les lecteurs"
    ],
    "historique": [
        "montre l'historique", "affiche l'historique", "voir l'historique", "liste des commandes", "historique des commandes"
    ],
    "system_info": [
        "informations système", "statut du système", "état du pc", "configuration système", "spécifications techniques",
        "quelle est ma configuration", "info système", "système info", "caractéristiques pc", "config pc"
    ],
    "search_files": [
        "recherche fichier", "recherche de fichier", "trouve fichier", "cherche fichier", "recherche dossier", "trouve dossier", "cherche dossier",
        "trouve document", "cherche document", "recherche dans mes fichiers", "recherche dans mes dossiers", "recherche des dossiers"
    ],
    
    "read_after_search": [
        "lire ce fichier", "ouvrir ce résultat", "afficher le contenu",
        "montre-moi ce fichier", "voir le résultat", "lire ce document"
    ],
    "edit_after_search": [
        "modifier ce fichier", "éditer ce résultat", "changer ce fichier",
        "corriger ce document", "ajouter du texte"
    ],
    "file_actions": [
        "actions pour ce fichier", "options pour ce fichier", "que peux-tu faire avec ce fichier",
        "menu fichier", "opérations sur ce fichier"
    ],
    "navigate_results": [
        "suivant", "résultat suivant", "prochain",
        "précédent", "résultat précédent", "avant"
    ],
    "file_info": [
        "information sur ce fichier", "détails de ce fichier", "propriétés de ce fichier",
        "stats de ce fichier", "caractéristiques de ce fichier"
    ],
    
    "open_explorer": [
        "ouvrir explorateur", "explorateur fichiers", "navigateur fichiers",
        "voir fichiers", "lister fichiers", "explorer disque",
        "parcourir dossiers", "ouvrir dossier", "navigation fichiers"
    ],
    
    "select_file": [
        "sélectionner ce fichier", "sélectionne le fichier", "choisir ce fichier", "prendre ce fichier",
        "sélectionner élément", "choisir élément", "prendre élément", "prend ce fichier"
    ],
    
    "navigate_back": [
        "retour arrière", "revenir en arrière", "précédent",
        "dossier précédent", "retour dossier", "retour"
    ],
    
    "navigate_forward": [
        "suivant", "avancer", "dossier suivant"
    ],
    
    
    "confirm_selection": [
        "valider sélection", "confirmer choix", "accepter sélection",
        "choisir ceci", "sélectionner ça", "confirmer"
    ]
}

RECOGNITION_MODE = "google"
VOSK_MODEL_PATH = os.path.join(os.path.dirname(__file__), "vosk-model-fr")

COMMAND_HISTORY = []

IS_AWAKE = True
WAKE_WORDS = ["assistant", "réveille-toi", "hey assistant"]
SLEEP_WORDS = ["dors", "va en veille", "arrête d'écouter"]

LAST_SEARCH_RESULTS = []
LAST_SEARCH_QUERY = ""
CURRENT_SEARCH_INDEX = 0

class WindowsProgramDetector:
    def __init__(self):
        self.installed_programs = {}
        self.detect_installed_programs()
    
    def detect_installed_programs(self):
        """Détecte les programmes installés sur Windows via le registre"""
        try:
            registry_paths = [
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
                (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
            ]
            
            for hkey, path in registry_paths:
                try:
                    key = winreg.OpenKey(hkey, path)
                    i = 0
                    while True:
                        try:
                            subkey_name = winreg.EnumKey(key, i)
                            subkey = winreg.OpenKey(key, subkey_name)
                            
                            try:
                                display_name = winreg.QueryValueEx(subkey, "DisplayName")[0]
                                install_location = winreg.QueryValueEx(subkey, "InstallLocation")[0] if winreg.QueryValueEx(subkey, "InstallLocation")[0] else ""
                                display_icon = winreg.QueryValueEx(subkey, "DisplayIcon")[0] if winreg.QueryValueEx(subkey, "DisplayIcon")[0] else ""
                                
                                exe_path = self._find_exe_path(install_location, display_icon)
                                
                                if display_name and exe_path and os.path.exists(exe_path):
                                    clean_name = display_name.split('(')[0].strip().lower()
                                    self.installed_programs[clean_name] = {
                                        'path': exe_path,
                                        'name': display_name
                                    }
                                
                            except (OSError, FileNotFoundError):
                                pass
                            
                            winreg.CloseKey(subkey)
                            i += 1
                        except OSError:
                            break
                    
                    winreg.CloseKey(key)
                except OSError:
                    pass
                    
        except Exception as e:
            print(f"Erreur détection programmes Windows: {e}")
    
    def _find_exe_path(self, install_dir, display_icon):
        """Trouve le chemin de l'exécutable"""
        if display_icon and display_icon.lower().endswith('.exe'):
            return display_icon.split(',')[0]  # Prendre le premier chemin si plusieurs icônes
        
        if install_dir and os.path.exists(install_dir):
            for root, dirs, files in os.walk(install_dir):
                for file in files:
                    if file.lower().endswith('.exe') and not file.lower().endswith('uninstall.exe'):
                        exe_path = os.path.join(root, file)
                        if 'uninstall' not in exe_path.lower() and 'setup' not in exe_path.lower():
                            return exe_path
        
        return None
    
    def find_program(self, program_name):
        program_name = program_name.lower()
        
        if program_name in self.installed_programs:
            return self.installed_programs[program_name]
        
        for installed_name, program_info in self.installed_programs.items():
            if program_name in installed_name or installed_name in program_name:
                return program_info
        
        return None

program_detector = WindowsProgramDetector()

def save_history():
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(COMMAND_HISTORY, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Erreur lors de la sauvegarde de l'historique : {e}")

def load_history():
    global COMMAND_HISTORY
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                COMMAND_HISTORY = json.load(f)
        except Exception as e:
            print(f"Erreur lors du chargement de l'historique : {e}")
            COMMAND_HISTORY = []
    else:
        COMMAND_HISTORY = []

def clear_history():
    global COMMAND_HISTORY
    COMMAND_HISTORY = []
    save_history()

load_history()

tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 160)
voices = tts_engine.getProperty('voices')
for voice in voices:
    if 'french' in voice.name.lower() or 'français' in voice.name.lower():
        tts_engine.setProperty('voice', voice.id)
        break

def speak(text):
    display_on_front(f"[Assistant] {text}")
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception as e:
        display_on_front(f"[Assistant] (Erreur vocalisation : {e})")

def listen():
    global RECOGNITION_MODE, VOSK_MODEL_PATH
    if RECOGNITION_MODE == "google":
        r = sr.Recognizer()
        with sr.Microphone() as source:
            display_on_front("Assistant : Calibration du bruit ambiant...")
            r.adjust_for_ambient_noise(source, duration=1)
            display_on_front("Assistant : J'écoute (Online)...")
            try:
                audio = r.listen(source, timeout=10, phrase_time_limit=10)
                command = r.recognize_google(audio, language="fr-FR")
                return command
            except sr.WaitTimeoutError:
                display_on_front("Assistant : Aucune voix détectée.")
                speak("Aucune voix détectée.")
                return ""
            except sr.UnknownValueError:
                display_on_front("Assistant : Je n'ai pas compris.")
                speak("Je n'ai pas compris.")
                return ""
            except sr.RequestError as e:
                display_on_front(f"Assistant : Erreur de service vocal : {e}")
                speak("Erreur de service vocal.")
                return ""
    elif RECOGNITION_MODE == "vosk":
        import json
        if not os.path.exists(VOSK_MODEL_PATH):
            msg = (
                f"Modèle Vosk non trouvé à l'emplacement : {VOSK_MODEL_PATH}\n"
                "Veuillez télécharger un modèle français depuis https://alphacephei.com/vosk/models "
                "et le placer dans ce dossier sous le nom 'vosk-model-fr'."
            )
            print(msg)
            speak("Modèle Vosk non trouvé. Veuillez installer le modèle hors ligne.")
            return ""
        if not os.path.exists(os.path.join(VOSK_MODEL_PATH, "model.conf")):
            msg = (
                f"Le dossier {VOSK_MODEL_PATH} ne contient pas de modèle Vosk valide.\n"
                "Vérifiez que le modèle est bien décompressé et complet."
            )
            print(msg)
            speak("Le dossier du modèle Vosk est incomplet ou corrompu.")
            return ""
        try:
            model = vosk.Model(VOSK_MODEL_PATH)
        except Exception as e:
            msg = f"Erreur lors du chargement du modèle Vosk : {e}"
            print(msg)
            speak("Erreur lors du chargement du modèle Vosk.")
            return ""
        q = queue.Queue()

        def callback(indata, frames, time, status):
            q.put(bytes(indata))

        with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                               channels=1, callback=callback):
            rec = vosk.KaldiRecognizer(model, 16000)
            print("Assistant : J'écoute (offline, Vosk)...")
            speak("J'écoute.")
            while True:
                data = q.get()
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text = result.get("text", "")
                    return text
    else:
        speak("Mode de reconnaissance inconnu.")
        return ""

def ask_feedback():
    speak("Est-ce que cela vous convient ? Dites oui ou non.")
    response = listen()
    if response and "oui" in response.lower():
        speak("Merci pour votre retour.")
        logging.info("Feedback utilisateur : Oui")
    elif response and "non" in response.lower():
        speak("D'accord, je vais essayer de m'améliorer.")
        logging.info("Feedback utilisateur : Non")
    else:
        speak("Je n'ai pas compris votre retour.")
        logging.info("Feedback utilisateur : Incompréhensible")

def register(username, password):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        conn.close()
        logging.info(f"Nouvel utilisateur inscrit : {username}")
        speak("Inscription réussie.")
    except sqlite3.IntegrityError:
        logging.warning(f"Nom d'utilisateur déjà utilisé : {username}")
        speak("Nom d'utilisateur déjà utilisé.")
    except Exception as e:
        logging.error(f"Erreur inscription : {e}")
        speak("Erreur lors de l'inscription.")
    ask_feedback()

def login(username, password):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = c.fetchone()
        conn.close()
        if user:
            logging.info(f"Connexion réussie pour : {username}")
            speak("Connexion réussie.")
            result = True
        else:
            logging.warning(f"Échec de connexion pour : {username}")
            speak("Échec de la connexion.")
            result = False
    except Exception as e:
        logging.error(f"Erreur connexion : {e}")
        speak("Erreur lors de la connexion.")
        result = False
    ask_feedback()
    return result

def logout():
    logging.info("Déconnexion utilisateur.")
    speak("Déconnexion réussie.")
    ask_feedback()

def add_event(date_str, event):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO events (date, event) VALUES (?, ?)", (date_str, event))
        conn.commit()
        conn.close()
        logging.info(f"Événement ajouté : {date_str} - {event}")
        speak("Événement ajouté.")
    except Exception as e:
        logging.error(f"Erreur ajout événement : {e}")
        speak("Erreur lors de l'ajout de l'événement.")
    ask_feedback()

def show_events():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id, date, event FROM events")
        events = c.fetchall()
        conn.close()
        if not events:
            speak("Aucun événement.")
            logging.info("Aucun événement à afficher.")
        else:
            for e in events:
                event_str = f"{e[0]} - {e[1]} : {e[2]}"
                display_on_front(event_str)
                speak(event_str)
            logging.info("Affichage des événements réussi.")
    except Exception as e:
        logging.error(f"Erreur affichage événements : {e}")
        speak("Erreur lors de l'affichage des événements.")
    ask_feedback()

def delete_event(event_id):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM events WHERE id=?", (event_id,))
        conn.commit()
        conn.close()
        logging.info(f"Événement supprimé : ID {event_id}")
        speak("Événement supprimé.")
    except Exception as e:
        logging.error(f"Erreur suppression événement : {e}")
        speak("Erreur lors de la suppression de l'événement.")
    ask_feedback()

def modify_event(event_id, new_event):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("UPDATE events SET event=? WHERE id=?", (new_event, event_id))
        conn.commit()
        conn.close()
        logging.info(f"Événement modifié : ID {event_id} -> {new_event}")
        speak("Événement modifié.")
    except Exception as e:
        logging.error(f"Erreur modification événement : {e}")
        speak("Erreur lors de la modification de l'événement.")
    ask_feedback()

def offer_search_actions(results):
    """Propose des actions après une recherche"""
    global CURRENT_SEARCH_INDEX
    
    if not results:
        speak("Aucun résultat à afficher.")
        return None
    
    # Afficher le premier résultat
    CURRENT_SEARCH_INDEX = 0
    current_result = results[CURRENT_SEARCH_INDEX]
    is_dir = os.path.isdir(current_result)
    
    result_type = "dossier" if is_dir else "fichier"
    speak(f"Premier {result_type} trouvé: {os.path.basename(current_result)}")
    display_on_front(f"1/{len(results)}: {os.path.basename(current_result)}")
    
    # Proposer des actions
    speak("Que voulez-vous faire? Vous pouvez dire: lire, ouvrir, suivant, précédent, ou actions pour plus d'options.")
    action = listen()
    
    if action:
        if "lire" in action.lower() and not is_dir:
            return read_file_after_search(current_result)
        elif "ouvrir" in action.lower():
            return open_file_default(current_result)
        elif "suivant" in action.lower() and len(results) > 1:
            CURRENT_SEARCH_INDEX = (CURRENT_SEARCH_INDEX + 1) % len(results)
            return offer_search_actions(results)
        elif "précédent" in action.lower() and len(results) > 1:
            CURRENT_SEARCH_INDEX = (CURRENT_SEARCH_INDEX - 1) % len(results)
            return offer_search_actions(results)
        elif "actions" in action.lower() or "option" in action.lower():
            return offer_file_actions(current_result)
        elif "recherche" in action.lower() or "nouveau" in action.lower():
            return search_files_vocal()
        else:
            speak("Action non reconnue. Veuillez réessayer.")
            return offer_search_actions(results)
    
    return None

def offer_file_actions(file_path):
    """Propose des actions sur un fichier spécifique"""
    is_dir = os.path.isdir(file_path)
    file_name = os.path.basename(file_path)
    
    speak(f"Options pour {file_name}. Que voulez-vous faire?")
    
    if is_dir:
        actions = [
            ("ouvrir", "Ouvrir le dossier"),
            ("lister", "Lister le contenu"),
            ("rechercher", "Rechercher dans ce dossier"),
            ("retour", "Retour aux résultats")
        ]
    else:
        # Détecter le type de fichier
        file_ext = os.path.splitext(file_path)[1].lower()
        
        actions = [
            ("lire", "Lire le contenu"),
            ("ouvrir", "Ouvrir avec l'application par défaut"),
            ("modifier", "Modifier le fichier"),
            ("renommer", "Renommer le fichier"),
            ("copier", "Copier le fichier"),
            ("déplacer", "Déplacer le fichier"),
            ("supprimer", "Supprimer le fichier"),
            ("info", "Informations sur le fichier"),
            ("retour", "Retour aux résultats")
        ]
        
        # Actions spécifiques selon le type de fichier
        if file_ext in ['.txt', '.csv', '.json', '.xml']:
            actions.insert(2, ("analyser", "Analyser le contenu"))
        elif file_ext in ['.pdf']:
            actions.insert(1, ("résumer", "Résumer le document"))
    
    # Énoncer les options disponibles
    options_text = "Options: "
    for i, (action, description) in enumerate(actions, 1):
        options_text += f"{i}. {description}. "
        if i % 3 == 0:  # Limiter la longueur des phrases
            speak(options_text)
            options_text = ""
    
    if options_text:
        speak(options_text)
    
    speak("Dites le numéro de l'action ou son nom.")
    choice = listen()
    
    if choice:
        # Gestion par numéro
        if choice.isdigit():
            index = int(choice) - 1
            if 0 <= index < len(actions):
                action = actions[index][0]
            else:
                speak("Numéro invalide.")
                return offer_file_actions(file_path)
        # Gestion par nom d'action
        else:
            action = None
            for a, desc in actions:
                if a in choice.lower():
                    action = a
                    break
            
            if not action:
                speak("Action non reconnue.")
                return offer_file_actions(file_path)
        
        # Exécuter l'action
        if action == "lire":
            return read_file_after_search(file_path)
        elif action == "ouvrir":
            return open_file_default(file_path)
        elif action == "modifier":
            return edit_file_after_search(file_path)
        elif action == "renommer":
            return rename_file_after_search(file_path)
        elif action == "copier":
            return copy_file_after_search(file_path)
        elif action == "déplacer":
            return move_file_after_search(file_path)
        elif action == "supprimer":
            return delete_file_after_search(file_path)
        elif action == "info":
            return get_file_info(file_path)
        elif action == "analyser":
            return analyze_file_content(file_path)
        elif action == "résumer":
            return summarize_document(file_path)
        elif action == "lister":
            return list_directory_content(file_path)
        elif action == "rechercher":
            speak("Que voulez-vous rechercher dans ce dossier?")
            new_query = listen()
            if new_query:
                return search_files_vocal_in_directory(file_path, new_query)
        elif action == "retour":
            return offer_search_actions(LAST_SEARCH_RESULTS)
    
    return file_path

def read_file_after_search(file_path):
    """Lit un fichier après une recherche"""
    try:
        if os.path.isdir(file_path):
            speak("C'est un dossier, je ne peux pas lire son contenu directement.")
            return file_path
            
        # Vérifier la taille du fichier
        file_size = os.path.getsize(file_path)
        if file_size > 5 * 1024 * 1024:  # 5MB
            speak("Ce fichier est trop volumineux pour être lu. Voulez-vous l'ouvrir avec une application?")
            response = listen()
            if response and "oui" in response.lower():
                return open_file_default(file_path)
            return file_path
        
        # Détecter le type de fichier pour une lecture adaptée
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return read_pdf_file(file_path)
        elif file_ext in ['.txt', '.csv', '.json', '.xml', '.py', '.js', '.html', '.css']:
            return read_text_file(file_path)
        else:
            speak("Je ne peux pas lire ce type de fichier directement. Voulez-vous l'ouvrir avec une application?")
            response = listen()
            if response and "oui" in response.lower():
                return open_file_default(file_path)
            return file_path
        
    except Exception as e:
        speak(f"Impossible de lire le fichier: {str(e)}")
        return file_path

def read_text_file(file_path):
    """Lit un fichier texte"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Limiter la lecture pour les gros fichiers
        if len(content) > 2000:
            speak("Le fichier est long. Je vais lire les premières lignes.")
            preview = content[:2000] + "..."
            display_on_front(f"Contenu de {os.path.basename(file_path)} (extrait):\n{preview}")
        else:
            display_on_front(f"Contenu de {os.path.basename(file_path)}:\n{content}")
        
        speak("Voici le contenu du fichier. Je vais en lire une partie.")
        
        # Lecture vocale des premières lignes
        lines = content.split('\n')
        lines_to_read = min(10, len(lines))
        
        for i in range(lines_to_read):
            if lines[i].strip():
                # Ne pas lire les lignes trop longues
                if len(lines[i]) > 200:
                    speak(f"Ligne {i+1} contient beaucoup de texte.")
                else:
                    speak(f"Ligne {i+1}: {lines[i].strip()}")
        
        # Proposer de continuer
        if len(lines) > lines_to_read:
            speak("Voulez-vous que je continue la lecture?")
            response = listen()
            if response and "oui" in response.lower():
                for i in range(lines_to_read, min(lines_to_read + 10, len(lines))):
                    if lines[i].strip():
                        speak(f"Ligne {i+1}: {lines[i].strip()}")
        
        return file_path
        
    except Exception as e:
        speak(f"Impossible de lire le fichier: {str(e)}")
        return file_path

def read_pdf_file(file_path):
    """Tente de lire un fichier PDF"""
    try:
        # Essayer d'importer PyPDF2 pour lire les PDF
        try:
            import PyPDF2
        except ImportError:
            speak("La lecture de PDF nécessite le module PyPDF2. Voulez-vous l'installer?")
            response = listen()
            if response and "oui" in response.lower():
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPDF2"])
                import PyPDF2
            else:
                speak("Je ne peux pas lire le PDF sans PyPDF2. Voulez-vous l'ouvrir avec une application?")
                response = listen()
                if response and "oui" in response.lower():
                    return open_file_default(file_path)
                return file_path
        
        # Lire le PDF
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            if len(pdf_reader.pages) == 0:
                speak("Le PDF est vide ou corrompu.")
                return file_path
            
            # Lire les premières pages
            speak(f"Le PDF contient {len(pdf_reader.pages)} pages. Je vais lire les premières.")
            
            text = ""
            pages_to_read = min(3, len(pdf_reader.pages))
            
            for i in range(pages_to_read):
                page = pdf_reader.pages[i]
                text += page.extract_text() + "\n"
            
            if text.strip():
                # Limiter la lecture
                if len(text) > 1500:
                    text = text[:1500] + "..."
                
                display_on_front(f"Contenu du PDF {os.path.basename(file_path)} (extrait):\n{text}")
                speak("Voici un extrait du PDF:")
                
                # Lire par paragraphes
                paragraphs = [p for p in text.split('\n\n') if p.strip()]
                for i, para in enumerate(paragraphs[:3]):
                    if para.strip():
                        speak(f"Paragraphe {i+1}: {para.strip()}")
            else:
                speak("Je n'ai pas pu extraire de texte de ce PDF. Il est peut-être scanné ou crypté.")
        
        return file_path
        
    except Exception as e:
        speak(f"Impossible de lire le PDF: {str(e)}")
        return file_path

def edit_file_after_search(file_path):
    """Modifie un fichier après une recherche"""
    try:
        if os.path.isdir(file_path):
            speak("C'est un dossier, je ne peux pas le modifier.")
            return file_path
        
        # Vérifier que le fichier est modifiable
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext in ['.exe', '.dll', '.sys', '.bin']:
            speak("Ce type de fichier ne peut pas être modifié pour des raisons de sécurité.")
            return file_path
            
        # Lire le contenu actuel
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                current_content = f.read()
        except UnicodeDecodeError:
            speak("Ce fichier n'est pas un fichier texte modifiable.")
            return file_path
        
        display_on_front(f"Contenu actuel de {os.path.basename(file_path)}:\n{current_content[:500]}{'...' if len(current_content) > 500 else ''}")
        
        speak("Que voulez-vous faire? Vous pouvez: ajouter du texte à la fin, remplacer tout le contenu, ou insérer à une position spécifique.")
        action = listen()
        
        if action and "ajouter" in action.lower():
            speak("Que voulez-vous ajouter à la fin du fichier?")
            new_content = listen()
            if new_content:
                with open(file_path, 'a', encoding='utf-8') as f:
                    f.write("\n" + new_content)
                speak("Contenu ajouté avec succès.")
                
        elif action and "remplacer" in action.lower():
            speak("Quel est le nouveau contenu?")
            new_content = listen()
            if new_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                speak("Fichier remplacé avec succès.")
                
        elif action and "insérer" in action.lower():
            speak("À quelle ligne voulez-vous insérer du texte? Dites le numéro ou 'au début'.")
            position = listen()
            
            if position:
                lines = current_content.split('\n')
                
                if position.isdigit():
                    line_num = int(position) - 1
                    if 0 <= line_num <= len(lines):
                        speak("Que voulez-vous insérer à cette position?")
                        insert_content = listen()
                        if insert_content:
                            lines.insert(line_num, insert_content)
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write('\n'.join(lines))
                            speak("Contenu inséré avec succès.")
                    else:
                        speak("Numéro de ligne invalide.")
                elif "début" in position.lower():
                    speak("Que voulez-vous insérer au début?")
                    insert_content = listen()
                    if insert_content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(insert_content + '\n' + current_content)
                        speak("Contenu inséré au début avec succès.")
                else:
                    speak("Position non reconnue.")
        else:
            speak("Modification annulée.")
            
        return file_path
        
    except Exception as e:
        speak(f"Impossible de modifier le fichier: {str(e)}")
        return file_path

def open_file_default(file_path):
    """Ouvre un fichier avec l'application par défaut"""
    try:
        if os.path.isdir(file_path):
            os.startfile(file_path)
            speak("Dossier ouvert.")
        else:
            os.startfile(file_path)
            speak("Fichier ouvert avec l'application par défaut.")
        return file_path
    except Exception as e:
        speak(f"Impossible d'ouvrir le fichier: {str(e)}")
        return file_path

def get_file_info(file_path):
    """Affiche des informations sur le fichier"""
    try:
        stat = os.stat(file_path)
        size = stat.st_size
        mtime = datetime.fromtimestamp(stat.st_mtime)
        ctime = datetime.fromtimestamp(stat.st_ctime)
        
        size_str = ""
        if size < 1024:
            size_str = f"{size} octets"
        elif size < 1024 * 1024:
            size_str = f"{size/1024:.1f} Ko"
        else:
            size_str = f"{size/(1024*1024):.1f} Mo"
        
        is_dir = os.path.isdir(file_path)
        file_type = "Dossier" if is_dir else "Fichier"
        
        info_text = f"""
        {file_type}: {os.path.basename(file_path)}
        Chemin: {file_path}
        Taille: {size_str}
        Créé le: {ctime.strftime('%d/%m/%Y à %H:%M')}
        Modifié le: {mtime.strftime('%d/%m/%Y à %H:%M')}
        """
        
        display_on_front(info_text)
        speak(f"Informations sur {os.path.basename(file_path)}: {file_type}, taille {size_str}, modifié le {mtime.strftime('%d %m %Y')}.")
        
        return file_path
        
    except Exception as e:
        speak(f"Impossible d'obtenir les informations du fichier: {str(e)}")
        return file_path

def analyze_file_content(file_path):
    """Analyse le contenu d'un fichier"""
    try:
        if os.path.isdir(file_path):
            speak("C'est un dossier, pas un fichier à analyser.")
            return file_path
            
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            return analyze_csv_file(file_path)
        elif file_ext in ['.txt', '.json', '.xml']:
            return analyze_text_file(file_path)
        else:
            speak("Je ne peux pas analyser ce type de fichier.")
            return file_path
            
    except Exception as e:
        speak(f"Impossible d'analyser le fichier: {str(e)}")
        return file_path

def analyze_csv_file(file_path):
    """Analyse un fichier CSV"""
    try:
        import csv
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Détecter le dialecte CSV
            try:
                dialect = csv.Sniffer().sniff(f.read(1024))
                f.seek(0)
            except:
                dialect = csv.excel
                
            reader = csv.reader(f, dialect)
            rows = list(reader)
            
            if not rows:
                speak("Le fichier CSV est vide.")
                return file_path
                
            headers = rows[0] if rows else []
            row_count = len(rows) - 1 if headers else len(rows)
            
            analysis = f"""
            Fichier CSV: {os.path.basename(file_path)}
            Lignes: {row_count}
            Colonnes: {len(headers)}
            En-têtes: {', '.join(headers)}
            """
            
            display_on_front(analysis)
            speak(f"Le fichier CSV contient {row_count} lignes et {len(headers)} colonnes. Les en-têtes sont: {', '.join(headers)}.")
            
            # Afficher un aperçu des données
            if row_count > 0:
                speak("Voici un aperçu des premières lignes:")
                preview_lines = min(3, row_count)
                for i in range(1, preview_lines + 1):
                    if i < len(rows):
                        row_preview = ", ".join(rows[i][:3])  # Premières 3 colonnes
                        if len(rows[i]) > 3:
                            row_preview += ", ..."
                        speak(f"Ligne {i}: {row_preview}")
            
        return file_path
        
    except Exception as e:
        speak(f"Impossible d'analyser le CSV: {str(e)}")
        return file_path

def analyze_text_file(file_path):
    """Analyse un fichier texte"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Statistiques de base
        lines = content.split('\n')
        words = content.split()
        characters = len(content)
        
        analysis = f"""
        Fichier: {os.path.basename(file_path)}
        Lignes: {len(lines)}
        Mots: {len(words)}
        Caractères: {characters}
        """
        
        display_on_front(analysis)
        speak(f"Le fichier contient {len(lines)} lignes, {len(words)} mots et {characters} caractères.")
        
        # Rechercher des motifs courants
        if any(email in content for email in ['@', '.com', '.fr']):
            speak("Le fichier semble contenir des adresses email.")
        
        if any(phone in content for phone in ['+33', '01', '02', '03', '04', '05', '06', '07']):
            speak("Le fichier semble contenir des numéros de téléphone.")
            
        return file_path
        
    except Exception as e:
        speak(f"Impossible d'analyser le fichier texte: {str(e)}")
        return file_path

def summarize_document(file_path):
    """Tente de résumer un document"""
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            # Pour PDF, on utilise la même méthode que read_pdf_file
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    if len(pdf_reader.pages) == 0:
                        speak("Le PDF est vide ou corrompu.")
                        return file_path
                    
                    # Extraire le texte des premières pages
                    text = ""
                    for i in range(min(5, len(pdf_reader.pages))):
                        page = pdf_reader.pages[i]
                        text += page.extract_text() + "\n"
                    
                    if text.strip():
                        # Créer un résumé très basique (premières phrases)
                        sentences = text.split('.')
                        summary = '.'.join(sentences[:3]) + '.' if len(sentences) > 3 else text
                        
                        display_on_front(f"Résumé de {os.path.basename(file_path)}:\n{summary}")
                        speak("Voici un résumé du document:")
                        speak(summary)
                    else:
                        speak("Je n'ai pas pu extraire de texte pour créer un résumé.")
            except:
                speak("Je ne peux pas résumer ce PDF. Voulez-vous l'ouvrir avec une application?")
        
        elif file_ext in ['.txt', '.csv', '.json', '.xml']:
            # Pour les fichiers texte, lire les premières lignes
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Prendre les premières lignes comme résumé
            lines = content.split('\n')
            summary_lines = min(10, len(lines))
            summary = '\n'.join(lines[:summary_lines])
            
            if len(lines) > summary_lines:
                summary += "\n..."
                
            display_on_front(f"Résumé de {os.path.basename(file_path)}:\n{summary}")
            speak("Voici le début du document:")
            
            # Lire les premières lignes
            for i in range(min(5, len(lines))):
                if lines[i].strip():
                    speak(f"Ligne {i+1}: {lines[i].strip()}")
        
        else:
            speak("Je ne peux pas résumer ce type de fichier.")
            
        return file_path
        
    except Exception as e:
        speak(f"Impossible de résumer le document: {str(e)}")
        return file_path

def list_directory_content(directory_path):
    """Liste le contenu d'un dossier"""
    try:
        if not os.path.isdir(directory_path):
            speak("Ce n'est pas un dossier valide.")
            return directory_path
            
        items = os.listdir(directory_path)
        if not items:
            speak("Le dossier est vide.")
            return directory_path
            
        files = []
        folders = []
        
        for item in items:
            item_path = os.path.join(directory_path, item)
            if os.path.isdir(item_path):
                folders.append(item)
            else:
                files.append(item)
                
        speak(f"Le dossier contient {len(folders)} sous-dossiers et {len(files)} fichiers.")
        
        # Afficher les dossiers
        if folders:
            speak("Sous-dossiers:")
            for i, folder in enumerate(folders[:5]):
                speak(f"{i+1}. {folder}")
            if len(folders) > 5:
                speak(f"Et {len(folders) - 5} autres dossiers.")
                
        # Afficher les fichiers
        if files:
            speak("Fichiers:")
            for i, file in enumerate(files[:5]):
                speak(f"{i+1}. {file}")
            if len(files) > 5:
                speak(f"Et {len(files) - 5} autres fichiers.")
                
        display_on_front(f"Contenu de {os.path.basename(directory_path)}:\n\nDossiers: {', '.join(folders[:10])}\n\nFichiers: {', '.join(files[:10])}")
        
        return directory_path
        
    except Exception as e:
        speak(f"Impossible de lister le contenu du dossier: {str(e)}")
        return directory_path

def search_files_vocal_in_directory(directory_path, query):
    """Recherche dans un dossier spécifique"""
    global LAST_SEARCH_RESULTS, LAST_SEARCH_QUERY, CURRENT_SEARCH_INDEX
    
    if not os.path.isdir(directory_path):
        speak("Ce n'est pas un dossier valide.")
        return None
        
    LAST_SEARCH_QUERY = query
    CURRENT_SEARCH_INDEX = 0

    speak(f"Recherche de '{query}' en cours dans {os.path.basename(directory_path)}...")
    results = []
    for root, dirs, files in os.walk(directory_path):
        for name in files + dirs:
            if query.lower() in name.lower():
                results.append(os.path.join(root, name))
        if len(results) > 50:
            break

    LAST_SEARCH_RESULTS = results

    if results:
        speak(f"J'ai trouvé {len(results)} résultats dans ce dossier.")
        return offer_search_actions(results)
    else:
        speak("Aucun résultat trouvé dans ce dossier.")
        return None

def open_file_explorer(mode="open", initial_path=None):
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    selected_path = file_explorer_open(mode=mode, initial_path=initial_path)
    if not selected_path:
        speak("Aucune sélection effectuée.")
        return None
    return selected_path

def read_file(filename=None):
    """Lit un fichier avec sélection vocale"""
    if not filename:
        speak("Quel fichier voulez-vous lire ?")
        filename = open_file_explorer("open")
        if not filename:
            return ""
    try:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                content = f.read()
                print(content)
                speak("Fichier lu avec succès")
                return content
        else:
            speak("Fichier non trouvé")
            return ""
    except Exception as e:
        speak(f"Erreur lors de la lecture du fichier: {e}")
        return ""

def write_file(filename, content):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        logging.info(f"Fichier écrit : {filename}")
        speak("Fichier écrit.")
    except Exception as e:
        logging.error(f"Erreur écriture fichier {filename} : {e}")
        speak("Erreur lors de l'écriture du fichier.")
    ask_feedback()

def delete_file(filename):
    try:
        if os.path.exists(filename):
            os.remove(filename)
            logging.info(f"Fichier supprimé : {filename}")
            speak("Fichier supprimé.")
        else:
            speak("Fichier non trouvé.")
            logging.warning(f"Fichier non trouvé : {filename}")
    except Exception as e:
        logging.error(f"Erreur suppression fichier {filename} : {e}")
        speak("Erreur lors de la suppression du fichier.")
    ask_feedback()

def create_folder(foldername=None):
    """Crée un dossier avec sélection vocale"""
    try:
        if not foldername:
            speak("Où voulez-vous créer le dossier ?")
            parent_dir = open_file_explorer("select_folder")  # Correction ici
            if not parent_dir:
                return
            speak("Comment voulez-vous nommer le dossier ?")
            foldername = listen()
            if not foldername:
                speak("Nom de dossier non reconnu")
                return
            foldername = os.path.join(parent_dir, foldername)
        if not os.path.exists(foldername):
            os.makedirs(foldername)
            speak(f"Dossier {os.path.basename(foldername)} créé avec succès")
            logging.info(f"Dossier créé: {foldername}")
        else:
            speak("Le dossier existe déjà")
            logging.warning(f"Dossier existe déjà: {foldername}")
    except Exception as e:
        error_msg = f"Erreur création dossier: {e}"
        speak(error_msg)
        logging.error(error_msg)

def list_files(path=None):
    global LAST_SEARCH_PATH
    if not path:
        path = LAST_SEARCH_PATH
    if not path or not os.path.isdir(path):
        speak("Aucun dossier sélectionné pour lister les fichiers.")
        return []
    try:
        files = os.listdir(path)
        display_on_front(f"Fichiers et dossiers dans {path} : {files}")
        feedback(True, "listage des fichiers")
        return files
    except Exception as e:
        feedback(False, "listage des fichiers")
        logging.error(f"Erreur listage fichiers dans {path} : {e}")
        return []

def rename_file(old_name, new_name):
    try:
        if os.path.exists(old_name):
            os.rename(old_name, new_name)
            logging.info(f"Fichier renommé : {old_name} -> {new_name}")
            speak("Fichier renommé.")
        else:
            speak("Fichier non trouvé.")
            logging.warning(f"Fichier non trouvé pour renommage : {old_name}")
    except Exception as e:
        logging.error(f"Erreur renommage fichier {old_name} : {e}")
        speak("Erreur lors du renommage du fichier.")
    ask_feedback()

def move_file(src=None, dst=None):
    if not src:
        speak("Quel fichier voulez-vous déplacer ?")
        src = select_path("open")
        if not src:
            feedback(False, "déplacement du fichier")
            return
    if not dst:
        speak("Où voulez-vous déplacer le fichier ?")
        dst = select_path("select_folder")
        if not dst:
            feedback(False, "déplacement du fichier")
            return
    try:
        if os.path.exists(src):
            import shutil
            dst_path = os.path.join(dst, os.path.basename(src))
            shutil.move(src, dst_path)
            logging.info(f"Fichier déplacé : {src} -> {dst}")
            speak("Fichier déplacé.")
        else:
            speak("Fichier non trouvé.")
            logging.warning(f"Fichier non trouvé pour déplacement : {src}")
    except Exception as e:
        logging.error(f"Erreur déplacement fichier {src} : {e}")
        speak("Erreur lors du déplacement du fichier.")
    ask_feedback()
    
def move_file_with_gui(src=None, dst=None):
    """Déplace un fichier avec interface graphique"""
    try:
        if not src:
            speak("Quel fichier voulez-vous déplacer ?")
            src = open_file_explorer("open")
            if not src:
                return
        
        if not dst:
            speak("Où voulez-vous déplacer le fichier ?")
            dst = open_file_explorer("select_folder")
            if not dst:
                return
        
        if os.path.exists(src):
            import shutil
            dst_path = os.path.join(dst, os.path.basename(src))
            shutil.move(src, dst_path)
            speak("Fichier déplacé avec succès")
        else:
            speak("Fichier source non trouvé")
            
    except Exception as e:
        speak(f"Erreur déplacement fichier: {e}")

def rename_file_after_search(file_path):
    """Renomme un fichier après une recherche"""
    try:
        if os.path.isdir(file_path):
            speak("C'est un dossier. Comment voulez-vous le renommer?")
        else:
            speak("Comment voulez-vous renommer ce fichier?")
            
        new_name = listen()
        if new_name:
            directory = os.path.dirname(file_path)
            new_path = os.path.join(directory, new_name)
            
            if os.path.exists(new_path):
                speak("Un élément avec ce nom existe déjà. Voulez-vous le remplacer?")
                response = listen()
                if not response or "non" in response.lower():
                    speak("Renommage annulé.")
                    return file_path
                    
            os.rename(file_path, new_path)
            speak("Élément renommé avec succès.")
            return new_path
        else:
            speak("Renommage annulé.")
            return file_path
            
    except Exception as e:
        speak(f"Impossible de renommer: {str(e)}")
        return file_path

def copy_file_after_search(file_path):
    """Copie un fichier après une recherche"""
    try:
        if os.path.isdir(file_path):
            speak("Où voulez-vous copier ce dossier?")
        else:
            speak("Où voulez-vous copier ce fichier?")
            
        speak("Dites le chemin du dossier de destination.")
        dest_dir = listen()
        
        if not dest_dir or not os.path.isdir(dest_dir):
            speak("Dossier de destination invalide.")
            return file_path
            
        dest_path = os.path.join(dest_dir, os.path.basename(file_path))
        
        if os.path.exists(dest_path):
            speak("Un élément avec ce nom existe déjà dans la destination. Voulez-vous le remplacer?")
            response = listen()
            if not response or "non" in response.lower():
                speak("Copie annulée.")
                return file_path
                
        import shutil
        if os.path.isdir(file_path):
            shutil.copytree(file_path, dest_path)
        else:
            shutil.copy2(file_path, dest_path)
            
        speak("Élément copié avec succès.")
        return file_path
        
    except Exception as e:
        speak(f"Impossible de copier: {str(e)}")
        return file_path

def move_file_after_search(file_path):
    """Déplace un fichier après une recherche"""
    try:
        if os.path.isdir(file_path):
            speak("Où voulez-vous déplacer ce dossier?")
        else:
            speak("Où voulez-vous déplacer ce fichier?")
            
        speak("Dites le chemin du dossier de destination.")
        dest_dir = listen()
        
        if not dest_dir or not os.path.isdir(dest_dir):
            speak("Dossier de destination invalide.")
            return file_path
            
        dest_path = os.path.join(dest_dir, os.path.basename(file_path))
        
        if os.path.exists(dest_path):
            speak("Un élément avec ce nom existe déjà dans la destination. Voulez-vous le remplacer?")
            response = listen()
            if not response or "non" in response.lower():
                speak("Déplacement annulé.")
                return file_path
                
        import shutil
        shutil.move(file_path, dest_path)
            
        speak("Élément déplacé avec succès.")
        return dest_path
        
    except Exception as e:
        speak(f"Impossible de déplacer: {str(e)}")
        return file_path

def delete_file_after_search(file_path):
    """Supprime un fichier après une recherche"""
    try:
        if os.path.isdir(file_path):
            speak(f"Voulez-vous vraiment supprimer le dossier {os.path.basename(file_path)} et tout son contenu?")
        else:
            speak(f"Voulez-vous vraiment supprimer le fichier {os.path.basename(file_path)}?")
            
        response = listen()
        if response and "oui" in response.lower():
            import shutil
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)
            speak("Élément supprimé avec succès.")
            return None
        else:
            speak("Suppression annulée.")
            return file_path
            
    except Exception as e:
        speak(f"Impossible de supprimer: {str(e)}")
        return file_path

def rename_file_with_gui(old_name=None, new_name=None):
    """Renomme un fichier avec interface graphique"""
    try:
        if not old_name:
            speak("Quel fichier voulez-vous renommer ?")
            old_name = open_file_explorer("open")
            if not old_name:
                return
        
        if not new_name:
            speak("Comment voulez-vous le renommer ?")
            new_name = listen()
            if not new_name:
                return
            
            directory = os.path.dirname(old_name)
            new_name = os.path.join(directory, new_name)
        
        if os.path.exists(old_name):
            os.rename(old_name, new_name)
            speak("Fichier renommé avec succès")
        else:
            speak("Fichier non trouvé")
            
    except Exception as e:
        speak(f"Erreur renommage fichier: {e}")

def get_command_suggestions(command, top_n=3):
    """Retourne les commandes les plus proches de ce que l'utilisateur a dit"""
    if not command.strip():
        return []
    
    suggestions = []
    command_lower = command.lower()
    
    for intent, keywords in INTENTS.items():
        for kw in keywords:
            similarity = similar(command_lower, kw.lower())
            if similarity > 0.4:
                label_fr = INTENT_LABELS_FR.get(intent, intent.replace('_', ' '))
                suggestions.append((kw, label_fr, intent, similarity))
    
    suggestions.sort(key=lambda x: x[3], reverse=True)
    return suggestions[:top_n]

def similar(a, b):
    a_words = set(a.split())
    b_words = set(b.split())
    if not a_words or not b_words:
        return 0.0
    return len(a_words.intersection(b_words)) / len(a_words.union(b_words))

def get_intent_spacy(command):
    doc = nlp(command.lower())
    tokens = set([token.lemma_ for token in doc])

    for intent, keywords in INTENTS.items():
        for kw in keywords:
            kw_doc = nlp(kw.lower())
            kw_tokens = set([token.lemma_ for token in kw_doc])
            if kw_tokens.issubset(tokens):
                return intent
    return None

def get_intent_spacy_similarity(command, threshold=0.75):
    doc_cmd = nlp(command.lower())
    best_intent = None
    best_score = 0.0

    for intent, keywords in INTENTS.items():
        for kw in keywords:
            kw_doc = nlp(kw.lower())
            score = doc_cmd.similarity(kw_doc)
            if score > best_score:
                best_score = score
                best_intent = intent

    if best_score >= threshold:
        return best_intent
    else:
        return None

def get_top_intents_spacy_similarity(command, top_n=3):
    doc_cmd = nlp(command.lower())
    scores = []

    for intent, keywords in INTENTS.items():
        max_score = 0.0
        for kw in keywords:
            kw_doc = nlp(kw.lower())
            score = doc_cmd.similarity(kw_doc)
            if score > max_score:
                max_score = score
        scores.append((intent, max_score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]

def get_system_info():
    """Retourne les informations du système Windows"""
    try:
        system_info = {
            'os_name': platform.system(),
            'version': platform.version(),
            'build': platform.release(),
            'ram': {
                'total': f"{psutil.virtual_memory().total // (1024**3)} Go",
                'percentage': f"{psutil.virtual_memory().percent}%"
            },
            'hostname': socket.gethostname(),
            'ip_address': socket.gethostbyname(socket.gethostname()),
            'cpu_usage': f"{psutil.cpu_percent()}%",
            'process_count': len(psutil.pids())
        }
        
        # Informations sur les disques
        disks_info = {}
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disks_info[partition.device] = {
                    'mountpoint': partition.mountpoint,
                    'total': f"{usage.total // (1024**3)} Go",
                    'used': f"{usage.used // (1024**3)} Go",
                    'free': f"{usage.free // (1024**3)} Go",
                    'percentage': f"{usage.percent}%"
                }
            except PermissionError:
                continue
        
        system_info['disks'] = disks_info
        return system_info
        
    except Exception as e:
        print(f"Erreur récupération infos système: {e}")
        return {
            'os_name': platform.system(),
            'version': platform.version(),
            'error': str(e)
        }

def launch_app(command):
    """Lance une application détectée automatiquement sur Windows"""
    app_keywords = ["ouvre", "ouvrir", "lance", "lancer", "démarre", "démarrer", "start", "open"]
    
    clean_command = command.lower()
    for keyword in app_keywords:
        clean_command = clean_command.replace(keyword, "")
    clean_command = clean_command.strip()
    
    system_apps = {
        "calculatrice": "calc.exe",
        "bloc-notes": "notepad.exe",
        "paint": "mspaint.exe",
        "explorateur": "explorer.exe",
        "cmd": "cmd.exe",
        "powershell": "powershell.exe",
        "wordpad": "write.exe",
        "magnifier": "magnify.exe",
        "narrator": "narrator.exe",
        "git": "git-bash.exe",
        "xamp": "xampp-control.exe",
    }
    
    if clean_command in system_apps:
        try:
            subprocess.Popen(system_apps[clean_command])
            speak(f"Lancement de {clean_command}.")
            return
        except Exception as e:
            speak(f"Erreur lors du lancement de {clean_command}.")
            return
    
    program_info = program_detector.find_program(clean_command)
    
    if program_info:
        try:
            app_path = program_info['path']
            app_name = program_info['name']
            
            if app_path.endswith('.exe'):
                subprocess.Popen([app_path])
            else:
                os.startfile(app_path)
            
            speak(f"Lancement de {app_name}.")
            logging.info(f"Application lancée: {app_name} - {app_path}")
            
        except Exception as e:
            error_msg = f"Erreur lors du lancement de {clean_command}: {e}"
            speak(error_msg)
            logging.error(error_msg)
    else:
        common_apps_mapping = {
            "chrome": "google chrome",
            "navigateur": "google chrome",
            "internet": "google chrome",
            "firefox": "mozilla firefox",
            "edge": "microsoft edge",
            "excel": "microsoft excel",
            "word": "microsoft word",
            "powerpoint": "microsoft powerpoint",
            "outlook": "microsoft outlook",
            "photoshop": "adobe photoshop",
            "acrobat": "adobe acrobat",
            "vscode": "visual studio code",
            "code": "visual studio code",
            "spotify": "spotify",
            "discord": "discord",
            "zoom": "zoom",
            "teams": "microsoft teams"
        }
        
        if clean_command in common_apps_mapping:
            mapped_name = common_apps_mapping[clean_command]
            program_info = program_detector.find_program(mapped_name)
            if program_info:
                try:
                    app_path = program_info['path']
                    subprocess.Popen([app_path])
                    speak(f"Lancement de {mapped_name}.")
                    return
                except Exception as e:
                    pass
        
        speak(f"Application '{clean_command}' non trouvée. Voulez-vous que je la recherche sur internet?")
        response = listen()
        if response and "oui" in response.lower():
            search_web(f"télécharger {clean_command} windows")
        else:
            speak(f"d'accord")

def confirm_action(message):
    """Centralise la confirmation vocale pour les actions destructives ou critiques"""
    speak(message + " Dites oui ou non.")
    response = listen()
    return response and "oui" in response.lower()

def feedback(success, operation):
    """Feedback vocal et affichage pour chaque opération"""
    if success:
        msg = f"{operation} réussi."
        speak(msg)
        display_on_front(msg)
    else:
        msg = f"Erreur lors de {operation}."
        speak(msg)
        display_on_front(msg)

def show_suggestions(command):
    """Affiche les suggestions IA contextuelles dans l’interface"""
    from ag7voc import get_top_intents_spacy_similarity, INTENT_LABELS_FR
    suggestions = get_top_intents_spacy_similarity(command)
    if suggestions:
        display_on_front("Suggestions IA :")
        for intent, score in suggestions:
            label = INTENT_LABELS_FR.get(intent, intent)
            display_on_front(f"- {label} (score: {score:.2f})")

def select_path(mode="open"):
    path = open_file_explorer(mode)
    return path

# Exemple d’utilisation centralisée dans les opérations sur fichiers/dossiers :

def delete_file(filename=None):
    if not filename:
        speak("Quel fichier voulez-vous supprimer ?")
        filename = select_path("open")
        if not filename:
            feedback(False, "suppression du fichier")
            return
    if os.path.exists(filename):
        if confirm_action(f"Voulez-vous vraiment supprimer {filename} ?"):
            try:
                os.remove(filename)
                feedback(True, "suppression du fichier")
            except Exception as e:
                feedback(False, "suppression du fichier")
                logging.error(f"Erreur suppression fichier {filename} : {e}")
        else:
            speak("Suppression annulée.")
            display_on_front("Suppression annulée.")
    else:
        feedback(False, "suppression du fichier")

def write_file(filename=None, content=None):
    if not filename:
        speak("Dans quel fichier voulez-vous écrire ?")
        filename = select_path("save")
        if not filename:
            feedback(False, "écriture du fichier")
            return
    if os.path.exists(filename):
        if not confirm_action(f"Le fichier {filename} existe déjà. Voulez-vous l'écraser ?"):
            speak("Écriture annulée.")
            display_on_front("Écriture annulée.")
            return
    if not content:
        speak("Que voulez-vous écrire dans le fichier ?")
        content = listen()
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        feedback(True, "écriture du fichier")
    except Exception as e:
        feedback(False, "écriture du fichier")
        logging.error(f"Erreur écriture fichier {filename} : {e}")

def move_file(src=None, dst=None):
    if not src:
        speak("Quel fichier voulez-vous déplacer ?")
        src = select_path("open")
        if not src:
            feedback(False, "déplacement du fichier")
            return
    if not dst:
        speak("Où voulez-vous déplacer le fichier ?")
        dst = select_path("select_folder")
        if not dst:
            feedback(False, "déplacement du fichier")
            return
    if os.path.exists(src):
        if confirm_action(f"Voulez-vous vraiment déplacer {os.path.basename(src)} vers {dst} ?"):
            try:
                import shutil
                dst_path = os.path.join(dst, os.path.basename(src))
                shutil.move(src, dst_path)
                feedback(True, "déplacement du fichier")
            except Exception as e:
                feedback(False, "déplacement du fichier")
                logging.error(f"Erreur déplacement fichier {src} : {e}")
        else:
            speak("Déplacement annulé.")
            display_on_front("Déplacement annulé.")
    else:
        feedback(False, "déplacement du fichier")

def rename_file(old_name=None, new_name=None):
    if not old_name:
        speak("Quel fichier voulez-vous renommer ?")
        old_name = select_path("open")
        if not old_name:
            feedback(False, "renommage du fichier")
            return
    if not new_name:
        speak("Comment voulez-vous le renommer ?")
        new_name = listen()
        if not new_name:
            feedback(False, "renommage du fichier")
            return
        directory = os.path.dirname(old_name)
        new_name = os.path.join(directory, new_name)
    if os.path.exists(old_name):
        if confirm_action(f"Voulez-vous vraiment renommer {os.path.basename(old_name)} en {os.path.basename(new_name)} ?"):
            try:
                os.rename(old_name, new_name)
                feedback(True, "renommage du fichier")
            except Exception as e:
                feedback(False, "renommage du fichier")
                logging.error(f"Erreur renommage fichier {old_name} : {e}")
        else:
            speak("Renommage annulé.")
            display_on_front("Renommage annulé.")
    else:
        feedback(False, "renommage du fichier")

def create_folder(foldername=None):
    speak("Où voulez-vous créer le dossier ?")
    parent_dir = select_path("select_folder")
    if not parent_dir:
        feedback(False, "création du dossier")
        return
    speak("Comment voulez-vous nommer le dossier ?")
    foldername = listen()
    if not foldername:
        feedback(False, "création du dossier")
        return
    folder_path = os.path.join(parent_dir, foldername)
    if os.path.exists(folder_path):
        speak("Le dossier existe déjà.")
        display_on_front("Le dossier existe déjà.")
        return
    try:
        os.makedirs(folder_path)
        feedback(True, "création du dossier")
    except Exception as e:
        feedback(False, "création du dossier")
        logging.error(f"Erreur création dossier {folder_path} : {e}")

def list_files(path=None):
    global LAST_SEARCH_PATH
    if not path:
        path = LAST_SEARCH_PATH
    if not path or not os.path.isdir(path):
        speak("Aucun dossier sélectionné pour lister les fichiers.")
        return []
    try:
        files = os.listdir(path)
        display_on_front(f"Fichiers et dossiers dans {path} : {files}")
        feedback(True, "listage des fichiers")
        return files
    except Exception as e:
        feedback(False, "listage des fichiers")
        logging.error(f"Erreur listage fichiers dans {path} : {e}")
        return []

# Exemple d’intégration des suggestions IA après chaque commande
def process_voice_command(command, forced_intent=None):
    global dqn_agent
    
    start_time = time.time()
    command_complexity = len(command.split()) / 20.0
    
    if forced_intent:
        intent = forced_intent
    else:
        intent = get_intent_spacy_similarity(command)
    
    if not intent:
        speak("Désolé, je n'ai pas compris la commande.")
        
        if DQN_AVAILABLE and dqn_agent:
            next_state = get_current_state(0, time.localtime().tm_hour/24, 
                                         psutil.cpu_percent()/100, 
                                         psutil.virtual_memory().percent/100, 
                                         0.5)
            reward = compute_reward("negatif", time.time() - start_time, command_complexity)
            dqn_agent.remember(dqn_agent.state, ACTIONS.index("demander_precisions"), reward, next_state, False)
            dqn_agent.state = next_state
        
        return
                
    success = True
    try:
        if intent == "add_event":
            date_str = extract_date(command)
            event = command
            add_event(date_str, event)
        elif intent == "show_events":
            show_events()
        elif intent == "delete_event":
            event_id = ''.join(filter(str.isdigit, command))
            if event_id:
                delete_event(int(event_id))
            else:
                speak("Veuillez préciser l'identifiant de l'événement à supprimer.")
        elif intent == "modify_event":
            event_id = ''.join(filter(str.isdigit, command))
            new_event = command.split("en")[-1].strip() if "en" in command else ""
            if event_id and new_event:
                modify_event(int(event_id), new_event)
            else:
                speak("Veuillez préciser l'identifiant et le nouveau texte.")
        elif intent == "read_after_search":
            if LAST_SEARCH_RESULTS and CURRENT_SEARCH_INDEX < len(LAST_SEARCH_RESULTS):
                read_file_after_search(LAST_SEARCH_RESULTS[CURRENT_SEARCH_INDEX])
            else:
                speak("Aucun résultat de recherche récent. Veuillez d'abord effectuer une recherche.")
        
        elif intent == "edit_after_search":
            if LAST_SEARCH_RESULTS and CURRENT_SEARCH_INDEX < len(LAST_SEARCH_RESULTS):
                edit_file_after_search(LAST_SEARCH_RESULTS[CURRENT_SEARCH_INDEX])
            else:
                speak("Aucun résultat de recherche récent. Veuillez d'abord effectuer une recherche.")
        
        elif intent == "file_actions":
            if LAST_SEARCH_RESULTS and CURRENT_SEARCH_INDEX < len(LAST_SEARCH_RESULTS):
                offer_file_actions(LAST_SEARCH_RESULTS[CURRENT_SEARCH_INDEX])
            else:
                speak("Aucun résultat de recherche récent. Veuillez d'abord effectuer une recherche.")
        
        elif intent == "navigate_results":
            if LAST_SEARCH_RESULTS:
                if "suivant" in command or "prochain" in command:
                    CURRENT_SEARCH_INDEX = (CURRENT_SEARCH_INDEX + 1) % len(LAST_SEARCH_RESULTS)
                    speak(f"Résultat {CURRENT_SEARCH_INDEX + 1} sur {len(LAST_SEARCH_RESULTS)}: {os.path.basename(LAST_SEARCH_RESULTS[CURRENT_SEARCH_INDEX])}")
                elif "précédent" in command or "avant" in command:
                    CURRENT_SEARCH_INDEX = (CURRENT_SEARCH_INDEX - 1) % len(LAST_SEARCH_RESULTS)
                    speak(f"Résultat {CURRENT_SEARCH_INDEX + 1} sur {len(LAST_SEARCH_RESULTS)}: {os.path.basename(LAST_SEARCH_RESULTS[CURRENT_SEARCH_INDEX])}")
            else:
                speak("Aucun résultat de recherche récent. Veuillez d'abord effectuer une recherche.")
        
        elif intent == "file_info":
            if LAST_SEARCH_RESULTS and CURRENT_SEARCH_INDEX < len(LAST_SEARCH_RESULTS):
                get_file_info(LAST_SEARCH_RESULTS[CURRENT_SEARCH_INDEX])
            else:
                speak("Aucun résultat de recherche récent. Veuillez d'abord effectuer une recherche.")
        elif intent == "search_files":
            search_files_vocal()
        elif intent == "read_file":
            read_file()
        elif intent == "write_file":
            filename = command.split()[-1]
            speak("Que voulez-vous écrire dans le fichier ?")
            content = listen()
            write_file(filename, content)
        elif intent == "delete_file":
            filename = command.split()[-1]
            delete_file(filename)
        elif intent == "create_folder":
            create_folder()
        elif intent == "list_files":
            list_files()
        elif intent == "rename_file":
            rename_file_with_gui()
        elif intent == "move_file":
            move_file_with_gui()
        elif intent == "get_time":
            from datetime import datetime
            now = datetime.now().strftime("%H:%M")
            speak(f"Il est {now}")
        elif intent == "get_date":
            from datetime import datetime
            today = datetime.now().strftime("%d/%m/%Y")
            speak(f"Aujourd'hui, nous sommes le {today}")
        elif intent == "show_help":
            show_help()
        elif intent == "launch_app":
            launch_app(command)
        elif intent == "shutdown":
            shutdown_computer()
        elif intent == "restart":
            restart_computer()
        elif intent == "lock":
            lock_computer()
        elif intent == "search_web":
            search_web(command)
        elif intent == "send_email":
            send_email(command)
        elif intent == "system_info":
            system_info = get_system_info()
            info_text = f"""
    Système: {system_info['os_name']}
    Version: {system_info['version']}
    RAM: {system_info['ram']['total']} (Utilisée: {system_info['ram']['percentage']})
    Hostname: {system_info['hostname']}
    IP: {system_info['ip_address']}
    CPU: {system_info['cpu_usage']}
    Processus: {system_info['process_count']}
            """
            
            print(info_text)
            speak("Voici les informations de votre système Windows.")
            
            if 'disks' in system_info:
                disks_text = "\nDisques:\n"
                for device, disk in system_info['disks'].items():
                    disks_text += f"  - {device}: {disk['free']} libre sur {disk['total']} ({disk['percentage']} utilisé)\n"
                print(disks_text)
        else:
            speak("Commande non reconnue.")

        COMMAND_HISTORY.append((command, intent))
        save_history()

        if intent == "historique":
            speak("Voici l'historique des commandes.")
            history = get_history()
            for idx, (cmd, intent_hist) in enumerate(history, 1):
                label_fr = INTENT_LABELS_FR.get(intent_hist, intent_hist.replace('_', ' ')) if intent_hist else "Inconnue"
                display_on_front(f"{idx}. {cmd} [{label_fr}]")
                speak(f"Commande {idx}: {cmd}")
            return
        elif intent == "repeat_last_command":
            if COMMAND_HISTORY:
                last_command = COMMAND_HISTORY[-1][0]
                speak("Je rejoue la dernière commande.")
                process_voice_command(last_command)
            else:
                speak("Aucune commande précédente à rejouer.")
            return
    except Exception as e:
        success = False
        logging.error(f"Erreur exécution commande: {e}")
        speak("Désolé, une erreur s'est produite.")

    if DQN_AVAILABLE and dqn_agent:
        execution_time = time.time() - start_time
        current_hour = time.localtime().tm_hour / 24.0
        cpu_usage = psutil.cpu_percent() / 100.0
        memory_usage = psutil.virtual_memory().percent / 100.0
        
        next_state = get_current_state(1 if success else 0, current_hour, cpu_usage, memory_usage, 0.5)
        
        reward = compute_reward("neutre", execution_time, command_complexity)
        
        action_idx = ACTIONS.index("executer_commande")
        
        dqn_agent.remember(dqn_agent.state, action_idx, reward, next_state, False)
        dqn_agent.state = next_state
        
        if voice_prefs.get_preference("auto_feedback"):
            speak("Est-ce que cela vous convient ?")
            feedback = listen()
            
            if feedback and DQN_AVAILABLE:
                user_mood = analyze_user_sentiment(feedback)
                feedback_type = "positif" if user_mood > 0.6 else "negatif" if user_mood < 0.4 else "neutre"
                
                reward = compute_reward(feedback_type, execution_time, command_complexity)
                dqn_agent.remember(dqn_agent.state, action_idx, reward, next_state, True)

    if DQN_AVAILABLE:
        try:
            dqn_agent = DQNAgent(5, len(ACTIONS))
            print("Agent DQN initialisé")
        except Exception as e:
            print(f"Erreur initialisation agent DQN: {e}")
            dqn_agent = None
            DQN_AVAILABLE = False
    else:
        dqn_agent = None
def check_write_permissions(self):
    try:
        test_file = "test_write.txt"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("test")
        os.remove(test_file)
        print("Permissions d'écriture OK")
        return True
    except Exception as e:
        print(f"Erreur permissions: {e}")
        return False

def extract_date(command):
    date = dateparser.parse(command, languages=['fr'])
    if date:
        return date.strftime("%Y-%m-%d")
    else:
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d")

def show_help():
    help_text = (
        "Voici les commandes disponibles :\n - ajouter un événement\n - afficher les événements\n - supprimer un événement\n - modifier un événement\n - lire un fichier\n - écrire un fichier\n - supprimer un fichier\n"
        "- créer un dossier\n - lister les fichiers\n - renommer un fichier\n - déplacer un fichier\n - obtenir l'heure\n - obtenir la date\n - aide\n - lancer une application\n - éteindre\n - redémarrer\n - verrouiller\n"
        "- chercher sur internet\n - envoyer un email\n - météo\n - informations système"
    )
    print(help_text)
    speak(help_text)

def shutdown_computer():
    try:
        speak("Arrêt de l'ordinateur en cours.")
        subprocess.run(["shutdown", "/s", "/t", "0"], check=True)
    except Exception as e:
        logging.error(f"Erreur arrêt ordinateur: {e}")
        speak("Erreur lors de l'arrêt de l'ordinateur.")

def restart_computer():
    try:
        speak("Redémarrage de l'ordinateur en cours.")
        subprocess.run(["shutdown", "/r", "/t", "0"], check=True)
    except Exception as e:
        logging.error(f"Erreur redémarrage ordinateur: {e}")
        speak("Erreur lors du redémarrage de l'ordinateur.")

def lock_computer():
    try:
        speak("Ordinateur verrouillé.")
        subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"], check=True)
    except Exception as e:
        logging.error(f"Erreur verrouillage ordinateur: {e}")
        speak("Erreur lors du verrouillage de l'ordinateur.")

def select_drive():
    """Sélectionne un lecteur avec interface vocale"""
    try:
        drives = []
        for letter in string.ascii_uppercase:
            drive_path = f"{letter}:\\"
            if os.path.exists(drive_path):
                try:
                    drive_name = winreg.QueryValueEx(winreg.HKEY_LOCAL_MACHINE, 
                                                    f"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Explorer\\DriveIcons\\{letter}\\DefaultLabel")[0]
                except:
                    drive_name = f"Lecteur {letter}"
                
                drives.append((drive_path, drive_name))
        
        if not drives:
            speak("Aucun lecteur disponible")
            return None
        
        speak("Quel lecteur voulez-vous explorer ?")
        
        from test_gui import VoiceControlledFileExplorer
        from PyQt5.QtWidgets import QApplication, QListWidgetItem,QDialog,Qt
        
        app = QApplication.instance() or QApplication([])
        explorer = VoiceControlledFileExplorer(mode="select_drive")
        
        explorer.file_list.clear()
        for drive_path, drive_name in drives:
            try:
                usage = psutil.disk_usage(drive_path)
                free_space = f"{usage.free // (1024**3)} Go libre"
            except:
                free_space = "Espace inconnu"
            
            item = QListWidgetItem(f"{drive_name} ({drive_path}) - {free_space}")
            item.setData(Qt.UserRole, drive_path)
            explorer.file_list.addItem(item)
        
        if explorer.exec_() == QDialog.Accepted:
            speak(f"Lecteur {explorer.selected_path} sélectionné")
            return explorer.selected_path
        
        return None
        
    except Exception as e:
        print(f"Erreur sélection lecteur: {e}")
        speak("Erreur lors de la sélection du lecteur")
        return None

def search_web(command):
    pattern = r"(cherche sur internet|recherche sur google|trouve sur internet|cherche|recherche|trouve)"
    query = re.sub(pattern, "", command, flags=re.IGNORECASE).strip()
    if not query:
        speak("Que voulez-vous rechercher ?")
        query = listen()
    if query:
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        webbrowser.open(url)
        speak(f"Voici les résultats pour : {query}")
    else:
        speak("Aucune recherche effectuée.")

def send_email(command):
    to = ""
    subject = ""
    body = ""
    match = re.search(r"à ([\w\.-]+@[\w\.-]+)", command)
    if match:
        to = match.group(1)
    else:
        speak("À qui souhaitez-vous envoyer l'email ? Dites l'adresse email.")
        to = listen()
    speak("Quel est le sujet de l'email ?")
    subject = listen()
    speak("Quel est le message ?")
    body = listen()
    if to and subject and body:
        url = f"mailto:{to}?subject={subject}&body={body}"
        webbrowser.open(url)
        speak("Email prêt à être envoyé dans votre client de messagerie.")
    else:
        speak("Informations incomplètes pour envoyer l'email.")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            event TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def list_drives():
    drives = []
    for letter in string.ascii_uppercase:
        if os.path.exists(f"{letter}:\\"):
            drives.append(f"{letter}:/")
    return drives

def get_history():
    return COMMAND_HISTORY

LAST_SEARCH_PATH = None

def search_files_vocal():
    global LAST_SEARCH_RESULTS, LAST_SEARCH_QUERY, CURRENT_SEARCH_INDEX
    
    speak("Quel nom de fichier ou dossier voulez-vous rechercher ?")
    query = listen()
    if not query:
        speak("Recherche annulée.")
        return None

    LAST_SEARCH_QUERY = query
    CURRENT_SEARCH_INDEX = 0

    speak("Dans quel dossier voulez-vous effectuer la recherche ? Dites le chemin ou laissez vide pour tout le disque.")
    folder = listen()
    if not folder or not os.path.exists(folder):
        folder = "C:\\"

    # Vérifie si le dossier est autorisé
    for forbidden in FORBIDDEN_FOLDERS:
        if os.path.abspath(folder).startswith(forbidden):
            speak("Recherche dans ce dossier interdite pour des raisons de sécurité.")
            return None

    speak(f"Recherche de '{query}' en cours dans {folder}...")
    results = []
    for root, dirs, files in os.walk(folder):
        if any(os.path.abspath(root).startswith(f) for f in FORBIDDEN_FOLDERS):
            continue
        for name in files + dirs:
            if query.lower() in name.lower():
                results.append(os.path.join(root, name))
        if len(results) > 50:
            break

    LAST_SEARCH_RESULTS = results

    if results:
        speak(f"J'ai trouvé {len(results)} résultats.")
        return offer_search_actions(results)
    else:
        speak("Aucun résultat trouvé.")
        return None

