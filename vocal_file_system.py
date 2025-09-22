# vocal_file_system.py - CORRECTION COMPLÈTE

import os
import shutil
import time
from pathlib import Path
import threading
from PyQt5.QtCore import QTimer, pyqtSignal, QObject

class VocalFileManager(QObject):
    """Gestionnaire complet des opérations fichiers/dossiers par commande vocale"""
    
    operation_complete = pyqtSignal(str, bool)  # message, succès
    confirmation_required = pyqtSignal(str, str)  # type_operation, details
    
    def __init__(self):
        super().__init__()
        self.current_context = None
        self.pending_operation = None
        self.operation_data = {}
        
        # États de navigation
        self.navigation_stack = []
        self.current_directory = Path.home()
        
    def navigate_to(self, target_path):
        """Navigation vers un dossier"""
        try:
            path = self.interpret_path(target_path)
            if path and path.exists() and path.is_dir():
                self.navigation_stack.append(self.current_directory)
                self.current_directory = path
                self.read_directory_content()
                return True
            else:
                self.speak(f"Dossier non trouvé : {target_path}")
                return False
        except Exception as e:
            self.speak(f"Erreur navigation : {str(e)}")
            return False
    
    def interpret_path(self, spoken_path):
        """Interprète un chemin parlé en chemin réel"""
        if not spoken_path:
            return None
            
        spoken_lower = spoken_path.lower()
        
        # Mapping des chemins courants
        special_paths = {
            "bureau": Path.home() / "Desktop",
            "documents": Path.home() / "Documents", 
            "téléchargements": Path.home() / "Downloads",
            "images": Path.home() / "Pictures",
            "musique": Path.home() / "Music",
            "vidéos": Path.home() / "Videos",
            "disque c": Path("C:/"),
            "disque d": Path("D:/"),
            "racine": Path("C:/"),
            "ici": self.current_directory,
            "parent": self.current_directory.parent,
        }
        
        # Recherche exacte
        for key, path in special_paths.items():
            if key in spoken_lower:
                return path
        
        # Recherche par nom de dossier
        if "dossier" in spoken_lower:
            folder_name = spoken_lower.replace("dossier", "").strip()
            return self.find_folder_by_name(folder_name)
        
        # Chemin direct
        potential_path = Path(spoken_path)
        if potential_path.exists():
            return potential_path
            
        return None
    
    def find_folder_by_name(self, folder_name):
        """Trouve un dossier par son nom"""
        search_locations = [
            self.current_directory,
            Path.home(),
            Path("C:/"),
        ]
        
        for location in search_locations:
            if not location.exists():
                continue
                
            try:
                for item in location.iterdir():
                    if item.is_dir() and folder_name.lower() in item.name.lower():
                        return item
            except PermissionError:
                continue
                
        return None
    
    def read_directory_content(self):
        """Lit et annonce le contenu du dossier courant"""
        try:
            items = list(self.current_directory.iterdir())
            folders = [item for item in items if item.is_dir()]
            files = [item for item in items if item.is_file()]
            
            message = f"Dossier {self.current_directory.name}. "
            message += f"{len(folders)} dossiers, {len(files)} fichiers."
            
            if folders:
                message += f" Dossiers : {', '.join([f.name for f in folders[:3]])}"
            if files:
                message += f" Fichiers : {', '.join([f.name for f in files[:3]])}"
                
            self.speak(message)
            return True
            
        except Exception as e:
            self.speak(f"Impossible de lire le dossier : {str(e)}")
            return False
    
    def create_folder(self, folder_name, parent_dir=None):
        """Crée un dossier"""
        try:
            parent = parent_dir or self.current_directory
            new_folder = parent / folder_name
            
            if new_folder.exists():
                self.speak(f"Le dossier {folder_name} existe déjà.")
                return False
                
            new_folder.mkdir(parents=True, exist_ok=True)
            self.speak(f"Dossier {folder_name} créé avec succès.")
            return True
            
        except Exception as e:
            self.speak(f"Erreur création dossier : {str(e)}")
            return False
    
    def delete_item(self, item_name):
        """Supprime un fichier ou dossier"""
        try:
            item_path = self.find_item(item_name)
            if not item_path:
                self.speak(f"{item_name} non trouvé.")
                return False
            
            if item_path.is_dir() and any(item_path.iterdir()):
                self.speak(f"Le dossier {item_name} n'est pas vide. Confirmez la suppression ?")
                self.pending_operation = ("delete", item_path)
                return True
                
            # Suppression simple
            if item_path.is_dir():
                shutil.rmtree(item_path)
            else:
                item_path.unlink()
                
            self.speak(f"{item_name} supprimé avec succès.")
            return True
            
        except Exception as e:
            self.speak(f"Erreur suppression : {str(e)}")
            return False
    
    def rename_item(self, old_name, new_name):
        """Renomme un fichier ou dossier"""
        try:
            old_path = self.find_item(old_name)
            if not old_path:
                self.speak(f"{old_name} non trouvé.")
                return False
                
            new_path = old_path.parent / new_name
            
            if new_path.exists():
                self.speak(f"Un élément nommé {new_name} existe déjà.")
                return False
                
            old_path.rename(new_path)
            self.speak(f"{old_name} renommé en {new_name}.")
            return True
            
        except Exception as e:
            self.speak(f"Erreur renommage : {str(e)}")
            return False
    
    def copy_item(self, item_name, destination):
        """Copie un fichier/dossier"""
        try:
            item_path = self.find_item(item_name)
            if not item_path:
                self.speak(f"{item_name} non trouvé.")
                return False
                
            dest_path = self.interpret_path(destination)
            if not dest_path or not dest_path.is_dir():
                self.speak(f"Destination {destination} invalide.")
                return False
                
            final_path = dest_path / item_path.name
            
            if item_path.is_dir():
                shutil.copytree(item_path, final_path)
            else:
                shutil.copy2(item_path, final_path)
                
            self.speak(f"{item_name} copié vers {destination}.")
            return True
            
        except Exception as e:
            self.speak(f"Erreur copie : {str(e)}")
            return False
    
    def move_item(self, item_name, destination):
        """Déplace un fichier/dossier"""
        try:
            item_path = self.find_item(item_name)
            if not item_path:
                self.speak(f"{item_name} non trouvé.")
                return False
                
            dest_path = self.interpret_path(destination)
            if not dest_path or not dest_path.is_dir():
                self.speak(f"Destination {destination} invalide.")
                return False
                
            final_path = dest_path / item_path.name
            shutil.move(str(item_path), str(final_path))
            
            self.speak(f"{item_name} déplacé vers {destination}.")
            return True
            
        except Exception as e:
            self.speak(f"Erreur déplacement : {str(e)}")
            return False
    
    def find_item(self, item_name):
        """Trouve un fichier/dossier par son nom"""
        # Recherche dans le dossier courant
        for item in self.current_directory.iterdir():
            if item_name.lower() in item.name.lower():
                return item
        
        # Recherche récursive limitée
        try:
            for item in self.current_directory.rglob("*"):
                if item_name.lower() in item.name.lower():
                    return item
                if len(list(self.current_directory.rglob("*"))) > 1000:
                    break
        except:
            pass
            
        return None
    
    def read_file_content(self, file_name, lines=5):
        """Lit le contenu d'un fichier"""
        try:
            file_path = self.find_item(file_name)
            if not file_path or not file_path.is_file():
                self.speak(f"Fichier {file_name} non trouvé.")
                return False
                
            if file_path.stat().st_size > 5 * 1024 * 1024:
                self.speak("Fichier trop volumineux pour la lecture vocale.")
                return False
                
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            lines_list = content.split('\n')[:lines]
            for i, line in enumerate(lines_list):
                if line.strip():
                    self.speak(f"Ligne {i+1} : {line.strip()}")
                    
            self.speak(f"Fin de la lecture de {file_name}.")
            return True
            
        except Exception as e:
            self.speak(f"Erreur lecture fichier : {str(e)}")
            return False
    
    def confirm_operation(self, response):
        """Traite les confirmations vocales"""
        if not self.pending_operation:
            return
            
        operation_type, data = self.pending_operation
        
        if "oui" in response.lower():
            if operation_type == "delete":
                try:
                    if data.is_dir():
                        shutil.rmtree(data)
                    else:
                        data.unlink()
                    self.speak("Élément supprimé avec succès.")
                except Exception as e:
                    self.speak(f"Erreur lors de la suppression : {str(e)}")
        else:
            self.speak("Opération annulée.")
            
        self.pending_operation = None
    
    def speak(self, text):
        """Synthèse vocale"""
        try:
            from voice_manager import voice_manager
            voice_manager.speak(text)
        except:
            print(f"🔊 VOIX: {text}")  # Fallback en mode texte
    
    def get_directory_info(self):
        """Donne des informations sur le dossier courant"""
        try:
            items = list(self.current_directory.iterdir())
            total_size = sum(f.stat().st_size for f in self.current_directory.rglob('*') if f.is_file())
            
            info = f"Dossier {self.current_directory.name}. "
            info += f"{len(items)} éléments. "
            info += f"Taille totale : {self.format_size(total_size)}."
            
            self.speak(info)
            return True
        except Exception as e:
            self.speak(f"Erreur informations dossier : {str(e)}")
            return False
    
    def format_size(self, size):
        """Formate la taille en unités lisibles"""
        for unit in ['o', 'Ko', 'Mo', 'Go']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} To"


class VocalFileCommandHandler:
    """Gestionnaire des commandes vocales fichiers/dossiers"""
    
    def __init__(self):
        self.file_manager = VocalFileManager()
        self.setup_voice_commands()
    
    def setup_voice_commands(self):
        """Configure les motifs de commandes vocales"""
        self.command_patterns = [
            # Pattern 1: "ouvre [le] [dossier] bureau"
            (r"(ouvre|ouvrir|affiche|montre)\s+(le\s+)?(dossier\s+)?(.+)", self.handle_navigation),
            
            # Pattern 2: "va dans [le] [dossier] documents"  
            (r"(va|aller|navigue)\s+(dans|vers|sur)\s+(le\s+)?(dossier\s+)?(.+)", self.handle_navigation),
            
            # Pattern 3: commandes simples vers dossiers spéciaux
            (r"(bureau|documents|téléchargements|images|musique|vidéos|disque c|disque d)", self.handle_special_folder),
            
            # Pattern 4: Création de dossiers
            (r"(crée|créer|nouveau|fais)\s+(un\s+)?(dossier\s+)?(.+)", self.handle_create_folder),
            
            # Pattern 5: Suppression
            (r"(supprime|efface|delete|enlève|retire)\s+(le\s+)?(dossier|fichier\s+)?(.+)", self.handle_delete),
            
            # Pattern 6: Lecture
            (r"(lis|affiche|montre)\s+(le\s+)?(contenu|fichier|texte\s+)?(.+)", self.handle_read),
        ]
    
    def handle_command(self, command):
        """Version améliorée avec meilleure reconnaissance"""
        command_lower = command.lower().strip()
        print(f"🔍 Analyse de la commande: '{command_lower}'")
        
        # Détection spéciale pour les dossiers courants
        special_folders = {
            'bureau': 'bureau',
            'documents': 'documents', 
            'téléchargements': 'téléchargements',
            'images': 'images',
            'musique': 'musique',
            'vidéos': 'vidéos',
            'disque c': 'disque c',
            'disque d': 'disque d'
        }
        
        # Vérifier les commandes simples vers dossiers spéciaux
        for folder_key, folder_name in special_folders.items():
            if folder_key in command_lower:
                print(f"🎯 Dossier spécial détecté: {folder_key}")
                return self.handle_special_folder(command, folder_name)
        
        # Vérifier les patterns complexes
        for pattern, handler in self.command_patterns:
            import re
            match = re.search(pattern, command_lower, re.IGNORECASE)
            if match:
                print(f"✅ Pattern matché: {pattern}")
                return handler(command, match.groups())
        
        print(f"❌ Aucun pattern matché pour: {command_lower}")
        return False
    
    def handle_special_folder(self, command, folder_name):
        """Gestion directe des dossiers spéciaux"""
        print(f"📁 Navigation vers dossier spécial: {folder_name}")
        return self.file_manager.navigate_to(folder_name)
    
    def handle_navigation(self, command, groups):
        """Gestion améliorée de la navigation"""
        print(f"🧭 Navigation avec groupes: {groups}")
        
        # Extraire le nom du dossier des groupes de capture
        target = ""
        if groups:
            # Prendre le dernier groupe non vide
            for group in reversed(groups):
                if group and group.strip():
                    target = group.strip()
                    break
        
        # Si pas de groupe, extraire de la commande
        if not target:
            target = self.extract_target_from_command(command)
        
        print(f"🎯 Cible extraite: '{target}'")
        
        if target:
            result = self.file_manager.navigate_to(target)
            print(f"📊 Résultat navigation: {result}")
            return result
        else:
            self.file_manager.speak("Quel dossier voulez-vous ouvrir ?")
            return True
    
    def handle_create_folder(self, command, groups):
        """Crée un nouveau dossier"""
        folder_name = ""
        if groups and len(groups) >= 4:
            folder_name = groups[3]  # Le 4ème groupe contient le nom
        
        if not folder_name:
            folder_name = self.extract_target_from_command(command)
        
        if folder_name:
            return self.file_manager.create_folder(folder_name)
        else:
            self.file_manager.speak("Comment voulez-vous nommer le nouveau dossier ?")
            return True
    
    def handle_delete(self, command, groups):
        """Supprime un fichier ou dossier"""
        item_name = ""
        if groups and len(groups) >= 4:
            item_name = groups[3]  # Le 4ème groupe contient le nom
        
        if not item_name:
            item_name = self.extract_target_from_command(command)
        
        if item_name:
            return self.file_manager.delete_item(item_name)
        else:
            self.file_manager.speak("Quel élément voulez-vous supprimer ?")
            return True
    
    def handle_read(self, command, groups):
        """Lit le contenu d'un fichier"""
        file_name = ""
        if groups and len(groups) >= 4:
            file_name = groups[3]  # Le 4ème groupe contient le nom
        
        if not file_name:
            file_name = self.extract_target_from_command(command)
        
        if file_name:
            return self.file_manager.read_file_content(file_name)
        else:
            self.file_manager.speak("Quel fichier voulez-vous lire ?")
            return True
    
    def extract_target_from_command(self, command):
        """Extraction améliorée de la cible"""
        command_lower = command.lower()
        
        # Supprimer les mots de commande
        command_words = ['ouvre', 'ouvrir', 'affiche', 'montre', 'va', 'aller', 'navigue', 
                        'dans', 'vers', 'sur', 'le', 'la', 'dossier', 'fichier', 'crée',
                        'créer', 'nouveau', 'fais', 'supprime', 'efface', 'delete', 'enlève',
                        'retire', 'lis', 'affiche', 'montre', 'contenu', 'texte']
        
        words = command_lower.split()
        target_words = [word for word in words if word not in command_words]
        
        target = ' '.join(target_words).strip()
        print(f"🎯 Cible extraite: '{target}'")
        return target


# ✅ CORRECTION: Créer l'instance APRÈS la définition des classes
vocal_file_handler = VocalFileCommandHandler()

# ✅ EXPORT EXPLICITE pour éviter les problèmes d'import
__all__ = ['VocalFileManager', 'VocalFileCommandHandler', 'vocal_file_handler']