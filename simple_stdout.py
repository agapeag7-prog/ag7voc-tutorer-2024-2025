# Solution minimaliste pour stdout
import sys
import time
from PyQt5.QtCore import QTimer

class SimpleStdout:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.original_stdout = sys.stdout
        
    def write(self, text):
        if text.strip():
            # Méthode directe sans complexité
            QTimer.singleShot(0, lambda: self._append_simple(text.strip()))
        self.original_stdout.write(text)  # Garder la console aussi
        
    def _append_simple(self, text):
        try:
            timestamp = time.strftime('[%H:%M:%S]')
            self.text_widget.append(f"{timestamp} {text}")
        except:
            pass
            
    def flush(self):
        pass

import sys
import time
from PyQt5.QtCore import QTimer

class SimpleStdout:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.original_stdout = sys.stdout
        
    def write(self, text):
        if text.strip():
            # Méthode directe sans complexité
            QTimer.singleShot(0, lambda: self._append_simple(text.strip()))
        self.original_stdout.write(text)  # Garder la console aussi
        
    def _append_simple(self, text):
        try:
            timestamp = time.strftime('[%H:%M:%S]')
            self.text_widget.append(f"{timestamp} {text}")
        except:
            pass
            
    def flush(self):
        pass

# Dans test_gui.py, remplacer la redirection :
def redirect_stdout(self):
    """Redirection stdout simplifiée"""
    sys.stdout = SimpleStdout(self.output_text)