"""
Preprocessing utilities for interview data
Handles conversation splitting and text cleaning
"""

import pandas as pd
import re

def split_conversation(text):
    """
    Découpe une conversation en tours de parole (Assistant/AI vs User).
    
    Parameters:
    -----------
    text : str
        Texte de la conversation complète
    
    Returns:
    --------
    pd.Series
        Série avec les tours de parole (AI_Turn_1, User_Turn_1, etc.)
    """
    # Découpage du texte selon les locuteurs
    parts = re.split(r'(Assistant:|AI:|User:)', text)

    # Nettoyage initial : on enlève les éléments vides
    cleaned_parts = [p.strip() for p in parts if p.strip()]

    turns = {}
    ai_count = 1
    user_count = 1

    for i in range(0, len(cleaned_parts) - 1, 2):
        speaker = cleaned_parts[i]
        content = cleaned_parts[i+1]

        # NETTOYAGE :
        # 1. Remplacer les sauts de ligne (\n, \r) par un espace
        # 2. Supprimer les doubles espaces créés par le remplacement
        # 3. Strip final pour les bords
        clean_content = re.sub(r'[\n\r]+', ' ', content)
        clean_content = re.sub(r'\s+', ' ', clean_content).strip()

        if "Assistant" in speaker or "AI" in speaker:
            turns[f'AI_Turn_{ai_count}'] = clean_content
            ai_count += 1
        elif "User" in speaker:
            turns[f'User_Turn_{user_count}'] = clean_content
            user_count += 1

    return pd.Series(turns)
