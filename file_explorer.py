import os
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QListWidget, QListWidgetItem
)
from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import QApplication

def open_file_explorer(mode="open", initial_path=None):
    app = QApplication.instance() or QApplication([])
    dialog = VoiceControlledFileExplorer(mode=mode, initial_path=initial_path)
    if dialog.exec_() == QDialog.Accepted:
        return dialog.selected_path
    return None

class VoiceControlledFileExplorer(QDialog):
    def __init__(self, parent=None, mode="open", initial_path=None):
        super().__init__(parent)
        self.mode = mode
        self.selected_path = None
        self.setWindowTitle("Explorateur de Fichiers - Contrôle Vocal")
        self.setMinimumSize(800, 600)
        self.initUI(initial_path)

    def initUI(self, initial_path):
        layout = QVBoxLayout(self)

        nav_layout = QHBoxLayout()
        self.back_btn = QPushButton("Retour")
        self.back_btn.clicked.connect(self.go_back)
        self.forward_btn = QPushButton("Suivant")
        self.forward_btn.setEnabled(False)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Chemin du dossier...")
        self.path_edit.returnPressed.connect(self.navigate_to_path)

        self.refresh_btn = QPushButton("Actualiser")
        self.refresh_btn.clicked.connect(self.refresh)

        nav_layout.addWidget(self.back_btn)
        nav_layout.addWidget(self.forward_btn)
        nav_layout.addWidget(self.path_edit)
        nav_layout.addWidget(self.refresh_btn)
        layout.addLayout(nav_layout)

        self.file_list = QListWidget()
        self.file_list.itemDoubleClicked.connect(self.on_item_double_clicked)
        self.file_list.setStyleSheet("""
            QListWidget {
                font-family: Consolas;
                font-size: 12px;
            }
            QListWidget::item {
                padding: 5px;
            }
            QListWidget::item:selected {
                background: #2196F3;
                color: white;
            }
        """)
        layout.addWidget(self.file_list)

        status_layout = QHBoxLayout()
        self.status_label = QLabel("Prêt")
        self.selected_label = QLabel("Aucune sélection")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        status_layout.addWidget(self.selected_label)
        layout.addLayout(status_layout)

        button_layout = QHBoxLayout()
        self.select_btn = QPushButton("Sélectionner")
        self.select_btn.clicked.connect(self.accept_selection)
        self.cancel_btn = QPushButton("Annuler")
        self.cancel_btn.clicked.connect(self.reject)

        button_layout.addWidget(self.select_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)

        self.history = []
        self.history_index = -1

        if initial_path and os.path.exists(initial_path):
            self.current_path = initial_path
        else:
            self.current_path = os.getcwd()

        self.navigate_to(self.current_path)

    def navigate_to(self, path):
        try:
            if os.path.exists(path):
                self.current_path = os.path.abspath(path)
                self.path_edit.setText(self.current_path)
                self.load_directory_contents()

                if not self.history or self.history[-1] != self.current_path:
                    self.history.append(self.current_path)
                    self.history_index = len(self.history) - 1

                self.update_navigation_buttons()
        except Exception as e:
            self.status_label.setText(f"Erreur: {e}")

    def load_directory_contents(self):
        self.file_list.clear()
        try:
            if self.current_path != os.path.dirname(self.current_path):
                parent_item = QListWidgetItem("../ (Dossier parent)")
                parent_item.setData(Qt.UserRole, os.path.dirname(self.current_path))
                self.file_list.addItem(parent_item)

            for item in sorted(os.listdir(self.current_path)):
                full_path = os.path.join(self.current_path, item)
                if os.path.isdir(full_path):
                    list_item = QListWidgetItem(f"{item}/")
                    list_item.setData(Qt.UserRole, full_path)
                    self.file_list.addItem(list_item)

            for item in sorted(os.listdir(self.current_path)):
                full_path = os.path.join(self.current_path, item)
                if os.path.isfile(full_path):
                    size = os.path.getsize(full_path)
                    size_str = self.format_file_size(size)
                    list_item = QListWidgetItem(f"📄 {item} ({size_str})")
                    list_item.setData(Qt.UserRole, full_path)
                    self.file_list.addItem(list_item)

            self.status_label.setText(f"{len(os.listdir(self.current_path))} éléments")
        except PermissionError:
            self.status_label.setText("Permission refusée")
        except Exception as e:
            self.status_label.setText(f"Erreur: {e}")

    def format_file_size(self, size):
        """Formate la taille du fichier de manière lisible"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"

    def on_item_double_clicked(self, item):
        path = item.data(Qt.UserRole)
        if os.path.isdir(path):
            self.navigate_to(path)
        else:
            self.selected_path = path
            self.selected_label.setText(f"Fichier sélectionné: {os.path.basename(path)}")

    def go_back(self):
        if self.history_index > 0:
            self.history_index -= 1
            self.navigate_to(self.history[self.history_index])

    def go_forward(self):
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.navigate_to(self.history[self.history_index])

    def update_navigation_buttons(self):
        self.back_btn.setEnabled(self.history_index > 0)
        self.forward_btn.setEnabled(self.history_index < len(self.history) - 1)

    def navigate_to_path(self):
        path = self.path_edit.text()
        if os.path.exists(path):
            self.navigate_to(path)
        else:
            self.status_label.setText("Chemin invalide")

    def refresh(self):
        self.navigate_to(self.current_path)

    def accept_selection(self):
        if self.file_list.currentItem():
            path = self.file_list.currentItem().data(Qt.UserRole)
            if os.path.isfile(path) and self.mode in ["open", "save"]:
                self.selected_path = path
                self.accept()
            elif os.path.isdir(path) and self.mode in ["select_folder", "select_drive"]:
                self.selected_path = path
                self.accept()
            else:
                self.status_label.setText("Sélection invalide pour ce mode")
        else:
            self.status_label.setText("Aucune sélection")
