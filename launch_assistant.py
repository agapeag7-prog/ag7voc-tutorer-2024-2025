import sys
import os
import ctypes

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_dependencies():
    """Vérifie les dépendances essentielles"""
    try:
        from PyQt5.QtWidgets import QApplication
        from tensorflow import keras
        import speech_recognition as sr
        return True
    except ImportError as e:
        print(f"Dépendance manquante: {e}")
        return False

def main():
    print("Lancement de l'assistant vocal AG7VOC")
    print("=" * 40)
    
    if not check_dependencies():
        print("Veuillez installer les dépendances: pip install -r requirements.txt")
        input("Appuyez sur Entrée pour quitter...")
        return
    
    try:
        from test_gui import VirtualAssistant
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtGui import QFont
        
        app = QApplication(sys.argv)
        
        font = QFont("Segoe UI", 10)
        app.setFont(font)
        
        try:
            with open("styles.qss", "r", encoding="utf-8") as f:
                app.setStyleSheet(f.read())
                print("Styles chargés")
        except FileNotFoundError:
            print("Fichier styles.qss non trouvé, utilisation des styles par défaut")
        except Exception as e:
            print(f"Erreur chargement styles: {e}")
        
        assistant = VirtualAssistant()
        assistant.show()
        
        print("Assistant lancé avec succès")
        print("Dites 'assistant' pour commencer")
        print("L'interface graphique est maintenant ouverte")
        
        if sys.platform == "win32":
            ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)
        
        sys.exit(app.exec_())
    
    except ImportError as e:
        print(f"Erreur d'importation: {e}")
        print("Lancez: pip install -r requirements.txt")
        input("Appuyez sur Entrée pour quitter...")
    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()
        input("Appuyez sur Entrée pour quitter...")

if __name__ == "__main__":
    main()