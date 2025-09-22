import sys
import os
import ctypes

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def check_dependencies():
    """Vérifie les dépendances essentielles"""
    try:
        import PyQt5
        import speech_recognition
        import pyttsx3
        import numpy
        return True
    except ImportError as e:
        print(f"Dépendance manquante: {e}")
        return False

def main():
    print("Lancement de l'assistant vocal AG7VOC")
    print("=" * 40)
    
    if not check_dependencies():
        print("Veuillez installer les dépendances manquantes")
        input("Appuyez sur Entrée pour quitter...")
        return
    
    try:
        from test_gui import VirtualAssistant
        from PyQt5.QtWidgets import QApplication
        
        app = QApplication(sys.argv)
        
        try:
            styles_path = os.path.join(current_dir, "styles.qss")
            if os.path.exists(styles_path):
                with open(styles_path, "r", encoding="utf-8") as f:
                    app.setStyleSheet(f.read())
        except:
            pass
        
        assistant = VirtualAssistant()
        assistant.show()
        
        print("Assistant lancé avec succès")
        print("Dites 'assistant' pour commencer")
        
        if sys.platform == "win32":
            ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)
        
        sys.exit(app.exec_())
    
    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()
        input("Appuyez sur Entrée pour quitter...")

if __name__ == "__main__":
    main()