"""
6.4 Dissonance entre discours et pratiques
Détecte les contradictions entre ce que les participants disent et ce qu'ils font.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from llm_config import call_llm_json, USE_LLM

def analyze_dissonance(df_utterances, top_n=30):
    """
    Analyse la dissonance discours/pratiques en utilisant embeddings + LLM.
    
    Parameters:
    -----------
    df_utterances : pd.DataFrame
        DataFrame avec colonnes 'parent_index', 'cluster', 'utterance', 'sim_dissonance'
    top_n : int
        Nombre de répliques à sélectionner par cluster selon sim_dissonance
    
    Returns:
    --------
    pd.DataFrame
        DataFrame avec colonnes 'parent_index', 'cluster', 'sdc' (Self-Discourse Conflict)
    """
    print("=== 6.4 DISSONANCE ENTRE DISCOURS ET PRATIQUES ===\n")
    
    # Étape 1: Filtrage sémantique
    if 'sim_dissonance' not in df_utterances.columns:
        print("⚠️ Colonne 'sim_dissonance' non disponible. Exécutez d'abord semantic_anchors.setup_semantic_anchors()")
        return pd.DataFrame()
    
    print("Étape 1: Filtrage sémantique...")
    selected_utterances = []
    for cluster_id in df_utterances['cluster'].dropna().unique():
        cluster_data = df_utterances[df_utterances['cluster'] == cluster_id].copy()
        top_repliques = cluster_data.nlargest(top_n, 'sim_dissonance')
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
                    df_selected.loc[orig_idx, 'normative_discourse'] = annotation.get('normative_discourse', 'no')
                    df_selected.loc[orig_idx, 'descriptive_practice'] = annotation.get('descriptive_practice', 'no')
                    df_selected.loc[orig_idx, 'dissonance_signal'] = annotation.get('dissonance_signal', 'no')
    else:
        print("⚠️ LLM non disponible, utilisation de valeurs par défaut")
        df_selected['normative_discourse'] = 'no'
        df_selected['descriptive_practice'] = 'no'
        df_selected['dissonance_signal'] = 'no'
    
    # Étape 3: Calcul du SDC (Self-Discourse Conflict) par entretien
    print("\nÉtape 3: Calcul du SDC par entretien...")
    dissonance_profiles = []
    
    for parent_id in df_selected['parent_index'].unique():
        interview_data = df_selected[df_selected['parent_index'] == parent_id]
        
        if len(interview_data) > 0:
            # Calculer les proportions
            total = len(interview_data)
            pct_normative = (interview_data['normative_discourse'] == 'yes').sum() / total if total > 0 else 0
            pct_descriptive = (interview_data['descriptive_practice'] == 'yes').sum() / total if total > 0 else 0
            pct_dissonance = (interview_data['dissonance_signal'] == 'yes').sum() / total if total > 0 else 0
            
            # SDC = produit des proportions (plus il y a de dissonance, plus le score est élevé)
            sdc = pct_normative * pct_descriptive * pct_dissonance
        else:
            sdc = 0.0
            pct_normative = 0.0
            pct_descriptive = 0.0
            pct_dissonance = 0.0
        
        dissonance_profiles.append({
            'parent_index': parent_id,
            'cluster': interview_data['cluster'].iloc[0] if len(interview_data) > 0 else None,
            'sdc': sdc,
            'pct_normative': pct_normative,
            'pct_descriptive': pct_descriptive,
            'pct_dissonance': pct_dissonance
        })
    
    df_dissonance_profiles = pd.DataFrame(dissonance_profiles)
    
    print(f"✓ {len(df_dissonance_profiles)} profils de dissonance créés")
    print(f"\n=== STATISTIQUES SDC ===")
    print(f"  - Moyenne: {df_dissonance_profiles['sdc'].mean():.4f}")
    print(f"  - Médiane: {df_dissonance_profiles['sdc'].median():.4f}")
    print(f"  - Max: {df_dissonance_profiles['sdc'].max():.4f}")
    
    return df_dissonance_profiles

def visualize_dissonance(df_dissonance_profiles):
    """
    Visualise la distribution du SDC par cluster.
    
    Parameters:
    -----------
    df_dissonance_profiles : pd.DataFrame
        DataFrame avec colonnes 'cluster', 'sdc'
    """
    if len(df_dissonance_profiles) == 0:
        print("⚠️ Pas de données à visualiser")
        return
    
    print("\n=== VISUALISATION DE LA DISSONANCE ===\n")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Graphique 1: Distribution globale du SDC
    ax1 = axes[0]
    ax1.hist(df_dissonance_profiles['sdc'], bins=20, color='#FF6B6B', alpha=0.7, edgecolor='black')
    ax1.set_title('Distribution du SDC (Self-Discourse Conflict)', fontweight='bold')
    ax1.set_xlabel('Score SDC', fontweight='bold')
    ax1.set_ylabel('Nombre d\'entretiens', fontweight='bold')
    ax1.axvline(df_dissonance_profiles['sdc'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Moyenne: {df_dissonance_profiles["sdc"].mean():.3f}')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Graphique 2: SDC par cluster
    ax2 = axes[1]
    if 'cluster' in df_dissonance_profiles.columns:
        cluster_sdc = df_dissonance_profiles.groupby('cluster')['sdc'].mean().sort_index()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(cluster_sdc)]
        bars = ax2.bar([f'Cluster {int(c)}' for c in cluster_sdc.index], cluster_sdc.values, color=colors)
        
        # Ajouter les valeurs sur les barres
        for bar, value in zip(bars, cluster_sdc.values):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{value:.4f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax2.set_title('SDC Moyen par Cluster', fontweight='bold')
        ax2.set_xlabel('Cluster', fontweight='bold')
        ax2.set_ylabel('Score SDC moyen', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.show()
    
    print("✓ Visualisations générées")
