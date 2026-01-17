"""
Calcul des embeddings d'ancrage sémantiques pour filtrer les répliques
avant annotation LLM. Utilise des phrases complètes comme ancres.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# === CONSTANTES : PHRASES D'ANCRAGE ===
ANCHOR_SENTENCES = {
    'identity': [
        "What defines me as a professional is the core work I do myself",
        "I refuse to delegate tasks that are central to my professional identity",
        "The meaning of my work comes from doing these tasks personally",
        "Ce qui me définit professionnellement, c'est ce que je fais moi-même",
        "Je refuse de déléguer ce qui fait le cœur de mon métier"
    ],
    'adaptation': [
        "I am learning new skills to adapt to AI in my field",
        "I supervise AI systems rather than doing the work myself",
        "I use AI secretly without telling my colleagues",
        "I am considering changing careers because of AI",
        "J'apprends de nouvelles compétences pour m'adapter à l'IA",
        "Je supervise les systèmes IA plutôt que de faire le travail moi-même"
    ],
    'dissonance': [
        "I should maintain control but in practice AI often decides",
        "I say I keep boundaries but actually I delegate a lot",
        "Je dis que je garde le contrôle mais en réalité l'IA décide souvent"
    ],
    'stigma': [
        "My colleagues judge me for using AI tools",
        "I hide my AI usage because people think it's lazy",
        "I feel embarrassed about relying on AI",
        "Mes collègues me jugent pour mon utilisation de l'IA",
        "Je cache mon usage de l'IA car les gens pensent que c'est de la paresse"
    ],
    'inequality': [
        "AI gives me a competitive advantage and I produce much more",
        "I must adapt to survive because the market is saturated",
        "AI helps me stay ahead of competitors",
        "L'IA me donne un avantage concurrentiel",
        "Je dois m'adapter pour survivre car le marché est saturé"
    ]
}

def calculate_anchor_embeddings(model: SentenceTransformer) -> dict:
    """
    Calcule les embeddings moyens pour chaque catégorie d'ancres sémantiques.
    
    Parameters:
    -----------
    model : SentenceTransformer
        Modèle d'embeddings à utiliser
    
    Returns:
    --------
    dict
        Dictionnaire {category: embedding_moyen} pour chaque catégorie
    """
    anchor_embeddings = {}
    
    for category, sentences in ANCHOR_SENTENCES.items():
        anchor_embs = model.encode(sentences)
        anchor_embeddings[category] = np.mean(anchor_embs, axis=0)
    
    return anchor_embeddings

def add_similarity_columns(df_utterances, model: SentenceTransformer, anchor_embeddings: dict):
    """
    Ajoute les colonnes de similarité (sim_identity, sim_adaptation, etc.) à df_utterances.
    
    Parameters:
    -----------
    df_utterances : pd.DataFrame
        DataFrame avec une colonne 'embedding'
    model : SentenceTransformer
        Modèle d'embeddings (pour compatibilité, non utilisé ici)
    anchor_embeddings : dict
        Dictionnaire des embeddings d'ancrage calculés
    """
    print("Calcul des similarités avec les ancres...")
    
    # Convertir les embeddings en array numpy
    if isinstance(df_utterances['embedding'].iloc[0], list):
        utterance_embeddings_array = np.array(df_utterances['embedding'].tolist())
    else:
        utterance_embeddings_array = np.stack(df_utterances['embedding'].values)
    
    # Calculer la similarité cosine pour chaque catégorie
    for category, anchor_emb in anchor_embeddings.items():
        similarities = cosine_similarity(utterance_embeddings_array, [anchor_emb]).flatten()
        df_utterances[f'sim_{category}'] = similarities
    
    print(f"\n✓ Similarités calculées pour {len(anchor_embeddings)} catégories")
    print(f"  Colonnes ajoutées: {[f'sim_{cat}' for cat in anchor_embeddings.keys()]}")
    
    print("\n=== STATISTIQUES DES SIMILARITÉS ===")
    for category in anchor_embeddings.keys():
        col = f'sim_{category}'
        print(f"{category}: min={df_utterances[col].min():.3f}, max={df_utterances[col].max():.3f}, mean={df_utterances[col].mean():.3f}")
    
    return df_utterances

def setup_semantic_anchors(df_utterances, model: SentenceTransformer = None):
    """
    Fonction principale pour configurer les ancres sémantiques.
    Calcule les embeddings d'ancrage et ajoute les colonnes de similarité.
    
    Parameters:
    -----------
    df_utterances : pd.DataFrame
        DataFrame avec colonne 'embedding'
    model : SentenceTransformer, optional
        Modèle d'embeddings. Si None, utilise 'all-MiniLM-L6-v2'
    
    Returns:
    --------
    dict
        Dictionnaire des embeddings d'ancrage
    """
    print("=== CALCUL DES EMBEDDINGS D'ANCRAGE ===\n")
    
    if 'df_utterances' not in globals() or 'embedding' not in df_utterances.columns:
        print("⚠ df_utterances ou embeddings non disponibles. Vérifiez que la section 5.4 a été exécutée.")
        return {}
    
    print(f"✓ Utilisation des embeddings existants de df_utterances ({len(df_utterances)} répliques)")
    
    # Charger le modèle si nécessaire
    if model is None:
        print("Chargement du modèle sentence-transformers...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        USE_EMBEDDINGS = True
    else:
        USE_EMBEDDINGS = True
    
    # Calculer les embeddings d'ancrage
    print("Calcul des embeddings pour les ancres sémantiques...")
    anchor_embeddings = {}
    for category, sentences in ANCHOR_SENTENCES.items():
        if USE_EMBEDDINGS and model is not None:
            anchor_embs = model.encode(sentences)
            anchor_embeddings[category] = np.mean(anchor_embs, axis=0)
            print(f"  ✓ {category}: {len(sentences)} ancres → embedding moyen calculé")
        else:
            print(f"  ⚠ Modèle non disponible pour {category}")
    
    # Ajouter les colonnes de similarité
    add_similarity_columns(df_utterances, model, anchor_embeddings)
    
    print("\n✓ Embeddings d'ancrage configurés et similarités calculées")
    
    return anchor_embeddings
