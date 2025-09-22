# voice_manager.py - CORRIGER la classe VoiceManager

import speech_recognition as sr
import pyttsx3
import threading
import time
import queue
import json
import os
from voice_preferences import voice_prefs

class VoiceManager:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.tts_engine = pyttsx3.init()
        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.vosk_model = None
        self.tts_busy = False
        
        # CORRECTION: Appeler la bonne méthode
        self.setup_voice_settings()  # Cette méthode doit exister !
        self.load_vosk_model()
    
    def setup_voice_settings(self):
        """Configure les paramètres vocaux - MÉTHODE MANQUANTE AJOUTÉE"""
        try:
            # Vitesse de parole
            voice_speed = voice_prefs.get_preference("voice_speed")
            if voice_speed:
                self.tts_engine.setProperty('rate', voice_speed)
            else:
                self.tts_engine.setProperty('rate', 160)  # Valeur par défaut
            
            # Volume
            voice_volume = voice_prefs.get_preference("voice_volume")
            if voice_volume:
                self.tts_engine.setProperty('volume', voice_volume)
            else:
                self.tts_engine.setProperty('volume', 1.0)  # Valeur par défaut
            
            # Voix française
            voices = self.tts_engine.getProperty('voices')
            for voice in voices:
                if 'french' in voice.name.lower() or 'français' in voice.name.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    break
            
            print("Paramètres vocaux configurés avec succès")
            
        except Exception as e:
            print(f"Erreur configuration voix: {e}")
            # Valeurs par défaut en cas d'erreur
            self.tts_engine.setProperty('rate', 160)
            self.tts_engine.setProperty('volume', 1.0)
    
    def load_vosk_model(self):
        """Charge le modèle Vosk si disponible"""
        try:
            import vosk
            model_path = os.path.join(os.path.dirname(__file__), "vosk-model-fr")
            if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "model.conf")):
                self.vosk_model = vosk.Model(model_path)
                print(f"Modèle Vosk chargé: {model_path}")
            else:
                print(f"Modèle Vosk non trouvé: {model_path}")
        except ImportError:
            print("Vosk non installé. Utilisation du mode Google uniquement.")
        except Exception as e:
            print(f"Erreur chargement modèle Vosk: {e}")
    
    def speak(self, text, async_mode=True):
        """Version corrigée avec gestion de concurrence"""
        if async_mode:
            threading.Thread(target=self._speak_sync, args=(text,), daemon=True).start()
        else:
            self._speak_sync(text)
    
    def _speak_sync(self, text):
        """Version sécurisée de la synthèse vocale"""
        if self.tts_busy:
            print(f"VOIX (différé): {text}")
            return
            
        self.tts_busy = True
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except RuntimeError as e:
            if "run loop already started" in str(e):
                print("Recreation du moteur TTS...")
                self.reset_tts_engine()
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            else:
                print(f"Erreur synthèse vocale: {e}")
        except Exception as e:
            print(f"Erreur synthèse vocale: {e}")
        finally:
            self.tts_busy = False
    
    def reset_tts_engine(self):
        """Réinitialise complètement le moteur TTS"""
        try:
            if self.tts_engine:
                self.tts_engine.stop()
        except:
            pass
        
        self.tts_engine = pyttsx3.init()
        self.setup_voice_settings()  # Reconfigurer les paramètres
        self.tts_busy = False
        print("Moteur TTS réinitialisé")
    
    def listen(self):
        """Écoute et reconnaît la parole"""
        mode = voice_prefs.get_preference("recognition_mode") or "google"
        timeout = voice_prefs.get_preference("timeout_listen") or 5
        phrase_time = voice_prefs.get_preference("phrase_time_limit") or 5
        
        try:
            with sr.Microphone() as source:
                print("Microphone ouvert...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                if mode == "google":
                    print("Écoute (Google)...")
                    audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time)
                    text = self.recognizer.recognize_google(audio, language=voice_prefs.get_preference("language") or "fr-FR")
                    print(f"Reconnu: '{text}'")
                    return text
                
                elif mode == "vosk" and self.vosk_model:
                    print("Écoute (Vosk offline)...")
                    audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time)
                    
                    import vosk
                    recognizer = vosk.KaldiRecognizer(self.vosk_model, 16000)
                    
                    audio_data = audio.get_wav_data()
                    if recognizer.AcceptWaveform(audio_data):
                        result = json.loads(recognizer.Result())
                        text = result.get('text', '')
                        print(f"✅ Reconnu (Vosk): '{text}'")
                        return text
                    return ""
                    
        except sr.WaitTimeoutError:
            print("Timeout écoute")
            return ""
        except sr.UnknownValueError:
            print("Audio incompréhensible")
            return ""
        except sr.RequestError as e:
            print(f"Erreur service reconnaissance: {e}")
            return ""
        except Exception as e:
            print(f"Erreur écoute: {e}")
            return ""
    
    def start_continuous_listening(self, callback, wake_word_callback=None):
        """Démarre l'écoute continue"""
        self.is_listening = True
        
        def listen_loop():
            while self.is_listening:
                command = self.listen()
                if command:
                    if wake_word_callback and any(word in command.lower() for word in voice_prefs.get_preference("wake_words") or []):
                        wake_word_callback(command)
                    elif callback:
                        callback(command)
                time.sleep(0.1)
        
        threading.Thread(target=listen_loop, daemon=True).start()
        print("Écoute continue activée")
    
    def stop_continuous_listening(self):
        """Arrête l'écoute continue"""
        self.is_listening = False
        print("Écoute continue désactivée")

voice_manager = VoiceManager()