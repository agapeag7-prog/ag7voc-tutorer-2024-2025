import os
import subprocess
import sys
import urllib.request
import zipfile

def install_requirements():
    """Installe les dépendances Python"""
    print("Installation des dépendances Python...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dépendances Python installées")
    except subprocess.CalledProcessError as e:
        print(f"Erreur installation dépendances: {e}")
        return False
    return True

def download_vosk_model():
    """Télécharge le modèle Vosk français"""
    print("Téléchargement du modèle Vosk français...")
    
    model_url = "https://alphacephei.com/vosk/models/vosk-model-fr-0.22.zip"
    model_dir = "vosk-model-fr"
    zip_path = "vosk-model-fr.zip"
    
    try:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        urllib.request.urlretrieve(model_url, zip_path)
        print("Modèle téléchargé")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(model_dir)
        print("Modèle extrait")
        
        os.remove(zip_path)
        print("Installation Vosk terminée")
        return True
        
    except Exception as e:
        print(f"Erreur téléchargement Vosk: {e}")
        return False

def download_spacy_model():
    """Télécharge le modèle spaCy français"""
    print("Installation du modèle spaCy français...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "fr_core_news_md"])
        print("Modèle spaCy installé")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Erreur installation spaCy: {e}")
        return False

def main():
    print("Installation de l'assistant vocal AG7VOC")
    print("=" * 50)
    
    success = True
    
    if not install_requirements():
        success = False
    
    # if not download_vosk_model():
    #     print("Modèle Vosk non installé, mode Google uniquement")
    
    if not download_spacy_model():
        print("Modèle spaCy non installé, certaines fonctionnalités limitées")
    
    if success:
        print("\nInstallation terminée avec succès!")
        print("Lancez: python launch_assistant.py")
    else:
        print("\nInstallation terminée avec des avertissements")
        print("Certaines fonctionnalités peuvent être limitées")

if __name__ == "__main__":
    main()