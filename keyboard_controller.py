import keyboard
import time
import os
import subprocess
from threading import Lock

class KeyboardController:
    def __init__(self):
        self.lock = Lock()
        
    def _safe_keypress(self, keys, delay=0.1):
        """Exécution sécurisée des touches avec verrouillage"""
        with self.lock:
            try:
                if isinstance(keys, str):
                    keyboard.press_and_release(keys)
                else:
                    for key in keys:
                        keyboard.press_and_release(key)
                        time.sleep(delay)
                time.sleep(0.5)
                return True
            except Exception as e:
                print(f"Erreur clavier: {e}")
                return False
    
    # === BUREAU VIRTUEL ===
    def bureau_virtuel_nouveau(self):
        if self._safe_keypress('win+ctrl+d'):
            return "Nouveau bureau virtuel créé"
        return "Erreur création bureau virtuel"
    
    def bureau_virtuel_suivant(self):
        if self._safe_keypress('win+ctrl+right'):
            return "Bureau virtuel suivant"
        return "Erreur navigation bureau"
    
    def bureau_virtuel_precedent(self):
        if self._safe_keypress('win+ctrl+left'):
            return "Bureau virtuel précédent"
        return "Erreur navigation bureau"
    
    def fermer_bureau_virtuel(self):
        if self._safe_keypress('win+ctrl+f4'):
            return "Bureau virtuel fermé"
        return "Erreur fermeture bureau"
    
    # === CAPTURE ÉCRAN ===
    def capture_ecran(self):
        if self._safe_keypress('win+shift+s'):
            return "Outil de capture activé - Sélectionnez la zone"
        return "Erreur activation capture"
    
    def capture_ecran_plein(self):
        if self._safe_keypress('print_screen'):
            return "Capture d'écran complet effectuée"
        return "Erreur capture écran"
    
    # === GESTION FENÊTRES ===
    def fenetre_snap_gauche(self):
        if self._safe_keypress('win+left'):
            return "Fenêtre ancrée à gauche"
        return "Erreur ancrage fenêtre"
    
    def fenetre_snap_droite(self):
        if self._safe_keypress('win+right'):
            return "Fenêtre ancrée à droite"
        return "Erreur ancrage fenêtre"
    
    def fenetre_maximiser(self):
        if self._safe_keypress('win+up'):
            return "Fenêtre maximisée"
        return "Erreur maximisation"
    
    def fenetre_minimiser(self):
        if self._safe_keypress('win+down'):
            return "Fenêtre minimisée"
        return "Erreur minimisation"
    
    # === SYSTÈME ===
    def verrouiller_ordinateur(self):
        if self._safe_keypress('win+l'):
            return "Ordinateur verrouillé"
        return "Erreur verrouillage"
    
    def minimiser_toutes_fenetres(self):
        if self._safe_keypress('win+d'):
            return "Toutes les fenêtres minimisées"
        return "Erreur minimisation"
    
    def gestionnaire_taches(self):
        if self._safe_keypress('ctrl+shift+esc'):
            return "Gestionnaire de tâches ouvert"
        return "Erreur ouverture gestionnaire"

keyboard_controller = KeyboardController()