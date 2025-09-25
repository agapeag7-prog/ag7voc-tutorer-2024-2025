import json
import os
import speech_recognition as sr

class VoicePreferences:
    def __init__(self, config_file="voice_preferences.json"):
        self.config_file = config_file
        self.preferences = self.load_preferences()
    
    def load_preferences(self):
        """Charge les préférences depuis le fichier"""
        default_prefs = {
            "recognition_mode": "google",
            "wake_words": ["assistant", "réveille-toi", "hey assistant"],
            "sleep_words": ["dors", "va en veille", "arrête d'écouter"],
            "voice_speed": 160,
            "voice_volume": 1.0,
            "timeout_listen": 10,
            "phrase_time_limit": 10,
            "vad_sensitivity": 0.5,
            "language": "fr-FR",
            "auto_feedback": True,
            "confirm_destructive_actions": True
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_prefs = json.load(f)
                    # Fusionner avec les valeurs par défaut
                    default_prefs.update(loaded_prefs)
                    print("Préférences vocales chargées depuis le fichier")
            else:
                print("Fichier de préférences non trouvé, utilisation des valeurs par défaut")
                
        except Exception as e:
            print(f"Erreur chargement préférences: {e}")
        
        return default_prefs
    
    def get_preference(self, key):
        """Récupère une préférence"""
        return self.preferences.get(key)
    
    def set_preference(self, key, value):
        """Définit une préférence"""
        self.preferences[key] = value
        return self.save_preferences()
    
    def save_preferences(self):
        """Sauvegarde les préférences dans le fichier"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.preferences, f, ensure_ascii=False, indent=2)
            print("Préférences vocales sauvegardées")
            return True
        except Exception as e:
            print(f"Erreur sauvegarde préférences: {e}")
            return False
    
voice_prefs = VoicePreferences()