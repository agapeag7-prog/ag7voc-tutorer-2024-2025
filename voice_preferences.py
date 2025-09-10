import json
import os
import speech_recognition as sr

class VoicePreferences:
    def __init__(self, config_file="voice_preferences.json"):
        self.config_file = config_file
        self.preferences = self.load_preferences()
    
    def load_preferences(self):
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
                    default_prefs.update(loaded_prefs)
                    return default_prefs
        except Exception as e:
            print(f"Erreur chargement préférences: {e}")
        
        return default_prefs
    
    def save_preferences(self):
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.preferences, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Erreur sauvegarde préférences: {e}")
            return False
    
    def get_preference(self, key):
        return self.preferences.get(key)
    
    def set_preference(self, key, value):
        self.preferences[key] = value
        return self.save_preferences()
    
    def get_available_voices(self):
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        return [voice.id for voice in voices]
    
    def get_available_recognition_modes(self):
        modes = ["google"]
        try:
            import vosk
            modes.append("vosk")
        except ImportError:
            pass
        return modes
    
    def get_microphones_list(self):
        microphones = []
        try:
            mic_list = sr.Microphone.list_microphone_names()
            microphones = list(enumerate(mic_list))
        except Exception as e:
            print(f"Erreur liste micros: {e}")
        return microphones

voice_prefs = VoicePreferences()