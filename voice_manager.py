import pyttsx3
import threading
from PyQt5.QtCore import QObject, pyqtSignal

class VoiceManager(QObject):
    """Gestionnaire vocal centralisé"""
    
    # Signaux pour l'interface
    speech_started = pyqtSignal()
    speech_finished = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.tts_engine = None
        self.is_speaking = False
        self._init_tts()
    
    def _init_tts(self):
        """Initialise le moteur TTS"""
        try:
            self.tts_engine = pyttsx3.init()
            
            # Configuration par défaut
            self.tts_engine.setProperty('rate', 160)  # Vitesse de parole
            
            # Chercher une voix française
            voices = self.tts_engine.getProperty('voices')
            french_voice = None
            
            for voice in voices:
                if 'french' in voice.name.lower() or 'français' in voice.name.lower():
                    french_voice = voice.id
                    break
            
            if french_voice:
                self.tts_engine.setProperty('voice', french_voice)
            else:
                print("Aucune voix française trouvée, utilisation de la voix par défaut")
            
            # Connecter les événements
            self.tts_engine.connect('started-utterance', self._on_speech_start)
            self.tts_engine.connect('finished-utterance', self._on_speech_end)
            
        except Exception as e:
            print(f"Erreur initialisation TTS: {e}")
            self.tts_engine = None
    
    def _on_speech_start(self):
        """Début de la parole"""
        self.is_speaking = True
        self.speech_started.emit()
    
    def _on_speech_end(self):
        """Fin de la parole"""
        self.is_speaking = False
        self.speech_finished.emit()
    
    def set_rate(self, rate):
        """Définit la vitesse de parole"""
        if self.tts_engine and 50 <= rate <= 300:
            self.tts_engine.setProperty('rate', rate)
    
    def get_rate(self):
        """Retourne la vitesse de parole actuelle"""
        if self.tts_engine:
            return self.tts_engine.getProperty('rate')
        return 160
    
    def set_volume(self, volume):
        """Définit le volume (0.0 à 1.0)"""
        if self.tts_engine and 0.0 <= volume <= 1.0:
            self.tts_engine.setProperty('volume', volume)
    
    def get_volume(self):
        """Retourne le volume actuel"""
        if self.tts_engine:
            return self.tts_engine.getProperty('volume')
        return 1.0
    
    def speak(self, text):
        """Parle le texte (thread-safe)"""
        if not self.tts_engine or not text:
            return
        
        def _speak():
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"Erreur parole: {e}")
        
        # Lancer dans un thread séparé
        thread = threading.Thread(target=_speak)
        thread.daemon = True
        thread.start()
    
    def stop(self):
        """Arrête la parole en cours"""
        if self.tts_engine:
            try:
                self.tts_engine.stop()
            except:
                pass
    
    def get_available_voices(self):
        """Retourne la liste des voix disponibles"""
        if self.tts_engine:
            return self.tts_engine.getProperty('voices')
        return []

# Instance globale
voice_manager = VoiceManager()