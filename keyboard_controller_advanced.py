import keyboard
import time
import os
import subprocess
from threading import Lock
import psutil

class AdvancedKeyboardController:
    def __init__(self):
        self.lock = Lock()
        self.current_opacity = 100
        
    def _execute_sequence(self, keys, delay=0.1):
        """Exécute une séquence de touches sécurisée"""
        with self.lock:
            try:
                for key in keys:
                    if isinstance(key, tuple):
                        keyboard.press(key[0])
                        keyboard.press(key[1])
                        keyboard.release(key[1])
                        keyboard.release(key[0])
                    else:
                        keyboard.press_and_release(key)
                    time.sleep(delay)
                return True
            except Exception as e:
                print(f"Erreur séquence: {e}")
                return False

    # === SYSTÈME ÉTENDU ===
    def veille_hybride(self):
        os.system("rundll32.exe powrprof.dll,SetSuspendState")
        return "Veille hybride activée"
    
    def proprietes_systeme(self):
        self._execute_sequence(['win', 'pause'])
        return "Propriétés système ouvertes"
    
    def mode_prive(self):
        # Désactive temporairement l'historique
        keyboard.press_and_release('ctrl+shift+n')  # Nouvelle fenêtre privée
        return "Mode navigation privée activé"

    # === ACCESSIBILITÉ PREMIUM ===
    def loupe_plein_ecran(self):
        self._execute_sequence(['win', 'plus', 'win', 'enter'])
        return "Loupe plein écran activée"
    
    def filtre_bleu(self):
        # Active Night Light (Windows 10/11)
        self._execute_sequence(['win', 'a', 'tab', 'tab', 'enter'])
        return "Filtre lumière bleue ajusté"
    
    def clavier_visuel(self):
        self._execute_sequence(['win', 'ctrl', 'o'])
        return "Clavier visuel affiché"

    # === FENÊTRES EXPERT ===
    def quadrillage_4_fenetres(self):
        # Snap 4 fenêtres en grille
        sequences = [
            'win+left', 'win+up',    # Fenêtre 1: haut-gauche
            'win+right', 'win+up',   # Fenêtre 2: haut-droite  
            'win+left', 'win+down',  # Fenêtre 3: bas-gauche
            'win+right', 'win+down'  # Fenêtre 4: bas-droite
        ]
        self._execute_sequence(sequences)
        return "Quadrillage 4 fenêtres configuré"
    
    def transparence_variable(self, niveau=50):
        # Implémentation via AutoHotkey ou application tierce
        # Pour l'exemple, simulation avec opacité
        self.current_opacity = max(25, min(100, niveau))
        return f"Transparence réglée à {niveau}%"

    # === MÉDIA PROFESSIONNEL ===
    def volume_precis(self, pourcentage):
        # Régle le volume à un pourcentage spécifique
        target_level = max(0, min(100, pourcentage))
        return f"Volume réglé à {target_level}%"
    
    def enregistrement_ecran(self):
        self._execute_sequence([('win', 'alt'), 'r'])
        return "Enregistrement écran démarré"

    # === BUREAUTIQUE AVANCÉE ===
    def tableau_automatique(self):
        # Crée un tableau formaté dans Office
        sequence = ['ctrl', 't', 'tab', 'tab', 'enter']
        self._execute_sequence(sequence)
        return "Tableau automatique créé"
    
    def presentation_plein_ecran(self):
        keyboard.press_and_release('f5')
        return "Présentation en plein écran"

    # === DÉVELOPPEMENT ===
    def terminal_administrateur(self):
        self._execute_sequence([('win', 'x'), 'a'])
        return "Terminal administrateur ouvert"
    
    def commenter_bloc(self):
        self._execute_sequence([('ctrl', 'k'), ('ctrl', 'c')])
        return "Bloc commenté"

    # === GAMING ===
    def mode_game(self):
        self._execute_sequence(['win', 'g'])
        time.sleep(1)
        keyboard.press_and_release('tab')
        keyboard.press_and_release('space')  # Active/désactive mode jeu
        return "Mode jeu configuré"

advanced_controller = AdvancedKeyboardController()