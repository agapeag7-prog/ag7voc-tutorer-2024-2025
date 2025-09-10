import re
from collections import Counter
import numpy as np

def cosine_similarity(vec1, vec2):
    """Calcule la similarité cosinus entre deux vecteurs"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2) if norm1 and norm2 else 0

def text_to_vector(text):
    """Convertit un texte en vecteur de mots"""
    words = re.findall(r'\w+', text.lower())
    return Counter(words)

def improved_similarity(text1, text2):
    """Similarité améliorée avec cosinus et pondération"""
    vec1 = text_to_vector(text1)
    vec2 = text_to_vector(text2)
    
    # Créer l'union des mots
    all_words = set(vec1.keys()) | set(vec2.keys())
    
    # Créer les vecteurs
    v1 = np.array([vec1.get(word, 0) for word in all_words])
    v2 = np.array([vec2.get(word, 0) for word in all_words])
    
    return cosine_similarity(v1, v2)

# Dans ag7voc.py, remplacez la fonction similar() par :
def similar(a, b):
    """Calcule une similarité améliorée entre deux chaînes"""
    return improved_similarity(a, b)