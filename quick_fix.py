# Solution d'urgence pour nlp
import sys
import os

# Ajouter le chemin actuel
sys.path.append(os.path.dirname(__file__))

def quick_nlp_fix():
    """Correction rapide pour nlp"""
    global nlp
    try:
        import spacy
        nlp = spacy.load("fr_core_news_sm")
        return nlp
    except:
        import spacy
        nlp = spacy.blank("fr")
        return nlp

# Appeler la correction
nlp = quick_nlp_fix()