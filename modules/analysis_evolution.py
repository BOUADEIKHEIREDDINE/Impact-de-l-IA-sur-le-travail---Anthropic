"""
6.1 Évolution des discours dans l'entretien
Analyse comment les positions évoluent entre le début et la fin de l'entretien.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def analyze_discourse_evolution(df_utterances, model: SentenceTransformer = None):
    """
    Analyse l'évolution des thèmes au cours de l'entretien.
    Divise chaque entretien en 3 segments et compare les scores de similarité.
    
    Parameters:
    -----------
    df_utterances : pd.DataFrame
        DataFrame avec colonnes 'parent_index', 'utterance', 'embedding'
    model : SentenceTransformer, optional
        Modèle d'embeddings pour calculer les similarités avec les thèmes
    
    Returns:
    --------
    pd.DataFrame
        DataFrame avec colonnes 'parent_index', 'position_relative', 'similarity_*' pour chaque thème
    """
    print("=== 6.1 ÉVOLUTION DES DISCOURS DANS L'ENTRETIEN ===\n")
    
    # Thèmes à analyser
    themes_evolution = {
        'Anxiété': ['anxiety', 'worry', 'fear', 'concern', 'threat', 'anxiété', 'inquiétude', 'peur'],
        'Optimisme': ['optimism', 'hope', 'positive', 'benefit', 'opportunity', 'optimisme', 'espoir', 'bénéfice'],
        'Contrôle': ['control', 'boundary', 'oversee', 'manage', 'contrôle', 'frontière', 'superviser'],
        'Dépendance': ['depend', 'rely', 'need', 'essential', 'dépendance', 'besoin', 'essentiel'],
        'Stratégie': ['strategy', 'adapt', 'learn', 'change', 'stratégie', 'adapter', 'apprendre']
    }
    
    # Charger le modèle si nécessaire
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Calculer les embeddings moyens pour chaque thème
    theme_embeddings = {}
    for theme, keywords in themes_evolution.items():
        theme_emb = model.encode(keywords)
        theme_embeddings[theme] = np.mean(theme_emb, axis=0)
    
    # Préparer les données d'évolution
    evolution_data = []
    
    for parent_id in df_utterances['parent_index'].unique():
        interview_utterances = df_utterances[df_utterances['parent_index'] == parent_id].copy()
        interview_utterances = interview_utterances.sort_index()  # Garder l'ordre original
        
        total_utterances = len(interview_utterances)
        if total_utterances == 0:
            continue
        
        # Diviser en 3 segments
        segment_size = total_utterances / 3
        interview_utterances['segment'] = pd.cut(
            range(total_utterances),
            bins=3,
            labels=['début', 'milieu', 'fin']
        )
        
        # Calculer la position relative (0.0 à 1.0)
        interview_utterances['position_relative'] = np.linspace(0, 1, total_utterances)
        
        # Calculer les similarités avec les thèmes
        if isinstance(interview_utterances['embedding'].iloc[0], list):
            utterance_embeddings = np.array(interview_utterances['embedding'].tolist())
        else:
            utterance_embeddings = np.stack(interview_utterances['embedding'].values)
        
        for theme, theme_emb in theme_embeddings.items():
            similarities = cosine_similarity(utterance_embeddings, [theme_emb]).flatten()
            interview_utterances[f'similarity_{theme}'] = similarities
        
        # Ajouter au DataFrame d'évolution
        evolution_data.append(interview_utterances[['parent_index', 'position_relative'] + 
                                                     [f'similarity_{t}' for t in themes_evolution.keys()]])
    
    df_evolution = pd.concat(evolution_data, ignore_index=True)
    
    print(f"✓ Données d'évolution préparées: {len(df_evolution)} répliques")
    print(f"  - {df_evolution['parent_index'].nunique()} entretiens analysés")
    
    return df_evolution, themes_evolution

def visualize_evolution(df_evolution, themes_evolution):
    """
    Visualise l'évolution des thèmes au cours de l'entretien avec des courbes.
    
    Parameters:
    -----------
    df_evolution : pd.DataFrame
        DataFrame avec colonnes 'position_relative' et 'similarity_*'
    themes_evolution : dict
        Dictionnaire des thèmes à visualiser
    """
    print("\n=== GÉNÉRATION DES COURBES D'ÉVOLUTION ===\n")
    
    # Créer des bins pour la position relative (0.0 à 1.0) - 20 points pour une courbe lisse
    df_evolution['position_bin'] = pd.cut(df_evolution['position_relative'], bins=20, labels=False) / 19.0
    
    # Calculer les moyennes par position pour chaque thème
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    for idx, theme in enumerate(themes_evolution.keys()):
        ax = axes[idx]
        
        # Grouper par position_bin et calculer la moyenne et l'écart-type
        evolution_by_position = df_evolution.groupby('position_bin')[f'similarity_{theme}'].agg(['mean', 'std', 'count'])
        evolution_by_position = evolution_by_position.reset_index()
        
        # Positions pour la courbe (0.0 à 1.0)
        positions = evolution_by_position['position_bin'].values
        means = evolution_by_position['mean'].values
        stds = evolution_by_position['std'].values
        
        # Tracer la courbe principale
        ax.plot(positions, means,
                color=colors[idx % len(colors)],
                linewidth=3,
                marker='o',
                markersize=6,
                label=f'Moyenne {theme}',
                zorder=3)
        
        # Ajouter une zone d'incertitude (écart-type)
        ax.fill_between(positions,
                         means - stds,
                         means + stds,
                         alpha=0.2,
                         color=colors[idx % len(colors)],
                         label='± 1 écart-type',
                         zorder=1)
        
        # Marquer les transitions entre segments (début/milieu/fin)
        ax.axvline(x=0.33, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=0.67, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Ajouter des annotations pour les segments
        y_max = ax.get_ylim()[1]
        ax.text(0.165, y_max * 0.95, 'Début', ha='center', fontsize=9, alpha=0.7,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        ax.text(0.5, y_max * 0.95, 'Milieu', ha='center', fontsize=9, alpha=0.7,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        ax.text(0.835, y_max * 0.95, 'Fin', ha='center', fontsize=9, alpha=0.7,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        ax.set_title(f'Évolution de {theme}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Position dans l\'entretien (0 = début, 1 = fin)', fontsize=10)
        ax.set_ylabel('Score de similarité moyen', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
        ax.legend(loc='best', fontsize=8)
        ax.set_xlim(-0.05, 1.05)
    
    plt.suptitle('Évolution des Thèmes au Cours de l\'Entretien (Courbes)',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()
    
    # Visualisation supplémentaire : Toutes les courbes sur un même graphique
    print("\n=== VUE D'ENSEMBLE : TOUTES LES COURBES ===")
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    for idx, theme in enumerate(themes_evolution.keys()):
        evolution_by_position = df_evolution.groupby('position_bin')[f'similarity_{theme}'].agg(['mean', 'std'])
        evolution_by_position = evolution_by_position.reset_index()
        
        positions = evolution_by_position['position_bin'].values
        means = evolution_by_position['mean'].values
        
        # Courbe principale
        ax.plot(positions, means,
                color=colors[idx % len(colors)],
                linewidth=2.5,
                marker='o',
                markersize=5,
                label=theme,
                alpha=0.8)
    
    # Marquer les transitions entre segments
    ax.axvline(x=0.33, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.axvline(x=0.67, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    y_max = ax.get_ylim()[1]
    ax.text(0.165, y_max * 0.98, 'Début', ha='center', fontsize=10, alpha=0.6,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    ax.text(0.5, y_max * 0.98, 'Milieu', ha='center', fontsize=10, alpha=0.6,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    ax.text(0.835, y_max * 0.98, 'Fin', ha='center', fontsize=10, alpha=0.6,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    ax.set_title('Évolution de Tous les Thèmes au Cours de l\'Entretien',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Position dans l\'entretien (0 = début, 1 = fin)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score de similarité moyen', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10, framealpha=0.9, ncol=2)
    ax.set_xlim(-0.05, 1.05)
    plt.tight_layout()
    plt.show()
    
    print("✓ Visualisations d'évolution générées")
