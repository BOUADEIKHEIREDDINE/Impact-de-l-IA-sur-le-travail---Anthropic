"""
6.6 Inégalités et trajectoires professionnelles
Analyse comment l'IA crée ou renforce des inégalités professionnelles.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from modules.llm_config import call_llm_json, USE_LLM

def analyze_inequality(df_utterances, top_n=30):
    """
    Analyse les inégalités professionnelles en utilisant embeddings + LLM.
    
    Parameters:
    -----------
    df_utterances : pd.DataFrame
        DataFrame avec colonnes 'parent_index', 'cluster', 'utterance', 'sim_inequality'
    top_n : int
        Nombre de répliques à sélectionner par cluster selon sim_inequality
    
    Returns:
    --------
    pd.DataFrame
        DataFrame avec colonnes 'parent_index', 'cluster', 'inequality_label'
    """
    print("=== 6.6 INÉGALITÉS ET TRAJECTOIRES PROFESSIONNELLES ===\n")
    
    # Étape 1: Filtrage sémantique
    if 'sim_inequality' not in df_utterances.columns:
        print("⚠️ Colonne 'sim_inequality' non disponible. Exécutez d'abord semantic_anchors.setup_semantic_anchors()")
        return pd.DataFrame()
    
    print("Étape 1: Filtrage sémantique...")
    selected_utterances = []
    for cluster_id in df_utterances['cluster'].dropna().unique():
        cluster_data = df_utterances[df_utterances['cluster'] == cluster_id].copy()
        top_repliques = cluster_data.nlargest(top_n, 'sim_inequality')
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
                    df_selected.loc[orig_idx, 'inequality_positioning'] = annotation.get('inequality_positioning', 'neutral')
    else:
        print("⚠️ LLM non disponible, utilisation de valeurs par défaut")
        df_selected['inequality_positioning'] = 'neutral'
    
    # Étape 3: Vote majoritaire par entretien
    print("\nÉtape 3: Vote majoritaire par entretien...")
    inequality_profiles = []
    
    for parent_id in df_selected['parent_index'].unique():
        interview_data = df_selected[df_selected['parent_index'] == parent_id]
        
        if 'inequality_positioning' in interview_data.columns:
            positionings = interview_data['inequality_positioning'].value_counts()
            inequality_label = positionings.index[0] if len(positionings) > 0 else 'neutral'
        else:
            inequality_label = 'neutral'
        
        inequality_profiles.append({
            'parent_index': parent_id,
            'cluster': interview_data['cluster'].iloc[0] if len(interview_data) > 0 else None,
            'inequality_label': inequality_label,
            'positioning_counts': dict(positionings) if 'inequality_positioning' in interview_data.columns else {}
        })
    
    df_inequality_profiles = pd.DataFrame(inequality_profiles)
    
    print(f"✓ {len(df_inequality_profiles)} profils d'inégalité créés")
    print("\n=== RÉPARTITION DES POSITIONNEMENTS ===")
    if 'inequality_label' in df_inequality_profiles.columns:
        label_counts = df_inequality_profiles['inequality_label'].value_counts()
        for label, count in label_counts.items():
            print(f"  - {label}: {count} entretiens ({count/len(df_inequality_profiles)*100:.1f}%)")
    
    return df_inequality_profiles

def visualize_inequality(df_inequality_profiles):
    """
    Visualise la répartition des positionnements d'inégalité par cluster.
    
    Parameters:
    -----------
    df_inequality_profiles : pd.DataFrame
        DataFrame avec colonnes 'cluster', 'inequality_label'
    """
    if len(df_inequality_profiles) == 0:
        print("⚠️ Pas de données à visualiser")
        return
    
    print("\n=== VISUALISATION DES INÉGALITÉS ===\n")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Graphique 1: Répartition globale
    ax1 = axes[0]
    if 'inequality_label' in df_inequality_profiles.columns:
        label_counts = df_inequality_profiles['inequality_label'].value_counts()
        colors_map = {'advantage': '#4ECDC4', 'survival_pressure': '#FF6B6B', 'neutral': '#95A5A6'}
        colors = [colors_map.get(label, '#95A5A6') for label in label_counts.index]
        ax1.bar(label_counts.index, label_counts.values, color=colors)
        ax1.set_title('Répartition Globale des Positionnements', fontweight='bold')
        ax1.set_xlabel('Positionnement', fontweight='bold')
        ax1.set_ylabel('Nombre d\'entretiens', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Graphique 2: Par cluster
    ax2 = axes[1]
    if 'cluster' in df_inequality_profiles.columns and 'inequality_label' in df_inequality_profiles.columns:
        cluster_inequality = pd.crosstab(df_inequality_profiles['cluster'], df_inequality_profiles['inequality_label'])
        cluster_inequality.plot(kind='bar', ax=ax2, stacked=True, colormap='Set2')
        ax2.set_title('Positionnements par Cluster', fontweight='bold')
        ax2.set_xlabel('Cluster', fontweight='bold')
        ax2.set_ylabel('Nombre d\'entretiens', fontweight='bold')
        ax2.legend(title='Positionnement', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.tick_params(axis='x', rotation=0)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.show()
    
    print("✓ Visualisations générées")
