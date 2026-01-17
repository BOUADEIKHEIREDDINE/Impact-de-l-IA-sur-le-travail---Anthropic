"""
6.3 Stratégies d'adaptation à l'IA
Crée une typologie des façons de s'adapter à l'IA au travail.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from modules.llm_config import call_llm_json, USE_LLM

def analyze_adaptation_strategies(df_utterances, top_n=30):
    """
    Analyse les stratégies d'adaptation en utilisant embeddings + LLM.
    
    Parameters:
    -----------
    df_utterances : pd.DataFrame
        DataFrame avec colonnes 'parent_index', 'cluster', 'utterance', 'sim_adaptation'
    top_n : int
        Nombre de répliques à sélectionner par cluster selon sim_adaptation
    
    Returns:
    --------
    pd.DataFrame
        DataFrame avec colonnes 'parent_index', 'cluster', 'dominant_strategy'
    """
    print("=== 6.3 STRATÉGIES D'ADAPTATION À L'IA ===\n")
    
    # Étape 1: Filtrage sémantique
    if 'sim_adaptation' not in df_utterances.columns:
        print("⚠️ Colonne 'sim_adaptation' non disponible. Exécutez d'abord semantic_anchors.setup_semantic_anchors()")
        return pd.DataFrame()
    
    print("Étape 1: Filtrage sémantique...")
    selected_utterances = []
    for cluster_id in df_utterances['cluster'].dropna().unique():
        cluster_data = df_utterances[df_utterances['cluster'] == cluster_id].copy()
        top_repliques = cluster_data.nlargest(top_n, 'sim_adaptation')
        selected_utterances.append(top_repliques)
    
    df_selected = pd.concat(selected_utterances, ignore_index=True)
    print(f"✓ {len(df_selected)} répliques sélectionnées")
    
    # Étape 2: Annotation LLM
    if USE_LLM:
        print("\nÉtape 2: Annotation LLM...")
        excerpts = []
        excerpt_indices = []
        for idx, row in df_selected.iterrows():
            text = row['utterance']
            sentences = text.split('.')
            excerpt = '. '.join(sentences[:10])[:500]
            if len(excerpt) > 50:
                excerpts.append(excerpt)
                excerpt_indices.append(idx)
        
        if excerpts:
            print(f"  - {len(excerpts)} excerpts préparés pour annotation LLM")
            llm_annotations = call_llm_json(excerpts)
            
            # Ajouter les annotations au DataFrame
            for i, (orig_idx, annotation) in enumerate(zip(excerpt_indices, llm_annotations)):
                if orig_idx in df_selected.index:
                    df_selected.loc[orig_idx, 'adaptation_strategy'] = annotation.get('adaptation_strategy', 'none')
    else:
        print("⚠️ LLM non disponible, utilisation de valeurs par défaut")
        df_selected['adaptation_strategy'] = 'none'
    
    # Étape 3: Vote majoritaire par entretien
    print("\nÉtape 3: Vote majoritaire par entretien...")
    adaptation_profiles = []
    
    for parent_id in df_selected['parent_index'].unique():
        interview_data = df_selected[df_selected['parent_index'] == parent_id]
        
        if 'adaptation_strategy' in interview_data.columns:
            strategies = interview_data['adaptation_strategy'].value_counts()
            dominant_strategy = strategies.index[0] if len(strategies) > 0 else 'none'
        else:
            dominant_strategy = 'none'
        
        adaptation_profiles.append({
            'parent_index': parent_id,
            'cluster': interview_data['cluster'].iloc[0] if len(interview_data) > 0 else None,
            'dominant_strategy': dominant_strategy,
            'strategy_counts': dict(strategies) if 'adaptation_strategy' in interview_data.columns else {}
        })
    
    df_adaptation_profiles = pd.DataFrame(adaptation_profiles)
    
    print(f"✓ {len(df_adaptation_profiles)} profils d'adaptation créés")
    print("\n=== RÉPARTITION DES STRATÉGIES ===")
    if 'dominant_strategy' in df_adaptation_profiles.columns:
        strategy_counts = df_adaptation_profiles['dominant_strategy'].value_counts()
        for strategy, count in strategy_counts.items():
            print(f"  - {strategy}: {count} entretiens ({count/len(df_adaptation_profiles)*100:.1f}%)")
    
    return df_adaptation_profiles

def visualize_adaptation_strategies(df_adaptation_profiles):
    """
    Visualise la répartition des stratégies d'adaptation par cluster.
    
    Parameters:
    -----------
    df_adaptation_profiles : pd.DataFrame
        DataFrame avec colonnes 'cluster', 'dominant_strategy'
    """
    if len(df_adaptation_profiles) == 0:
        print("⚠️ Pas de données à visualiser")
        return
    
    print("\n=== VISUALISATION DES STRATÉGIES D'ADAPTATION ===\n")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Graphique 1: Répartition globale
    ax1 = axes[0]
    if 'dominant_strategy' in df_adaptation_profiles.columns:
        strategy_counts = df_adaptation_profiles['dominant_strategy'].value_counts()
        colors = plt.cm.Set3(range(len(strategy_counts)))
        ax1.bar(strategy_counts.index, strategy_counts.values, color=colors)
        ax1.set_title('Répartition Globale des Stratégies', fontweight='bold')
        ax1.set_xlabel('Stratégie', fontweight='bold')
        ax1.set_ylabel('Nombre d\'entretiens', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Graphique 2: Par cluster
    ax2 = axes[1]
    if 'cluster' in df_adaptation_profiles.columns and 'dominant_strategy' in df_adaptation_profiles.columns:
        cluster_strategy = pd.crosstab(df_adaptation_profiles['cluster'], df_adaptation_profiles['dominant_strategy'])
        cluster_strategy.plot(kind='bar', ax=ax2, stacked=True, colormap='Set3')
        ax2.set_title('Stratégies par Cluster', fontweight='bold')
        ax2.set_xlabel('Cluster', fontweight='bold')
        ax2.set_ylabel('Nombre d\'entretiens', fontweight='bold')
        ax2.legend(title='Stratégie', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.tick_params(axis='x', rotation=0)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.show()
    
    print("✓ Visualisations générées")
