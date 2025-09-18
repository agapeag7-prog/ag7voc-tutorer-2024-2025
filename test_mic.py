import speech_recognition as sr
import sys

def test_microphone():
    print("Test du microphone...")
    
    try:
        r = sr.Recognizer()
        
        print("Microphones disponibles:")
        mics = sr.Microphone.list_microphone_names()
        for i, mic in enumerate(mics):
            print(f"  {i}: {mic}")
        
        with sr.Microphone() as source:
            print("Calibration du bruit ambiant...")
            r.adjust_for_ambient_noise(source, duration=2)
            
            print("Parlez maintenant (vous avez 3 secondes)...")
            audio = r.listen(source, timeout=5, phrase_time_limit=3)
            
            try:
                text = r.recognize_google(audio, language="fr-FR")
                print(f"Succès! J'ai entendu: '{text}'")
                return True
            except sr.UnknownValueError:
                print("Je n'ai rien compris")
                return False
            except sr.RequestError as e:
                print(f"Erreur service Google: {e}")
                return False
                
    except Exception as e:
        print(f"Erreur générale: {e}")
        return False

if __name__ == "__main__":
    if test_microphone():
        print("Microphone fonctionnel!")
        sys.exit(0)
    else:
        print("Probleme microphone!")
        sys.exit(1)