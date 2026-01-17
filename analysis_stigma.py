"""
6.5 Stigmatisation sociale et usage caché
Analyse la stigmatisation liée à l'utilisation de l'IA au travail.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from llm_config import call_llm_json, USE_LLM

def analyze_stigma(df_utterances, top_n=30):
    """
    Analyse la stigmatisation sociale en utilisant embeddings + LLM.
    
    Parameters:
    -----------
    df_utterances : pd.DataFrame
        DataFrame avec colonnes 'parent_index', 'cluster', 'utterance', 'sim_stigma'
    top_n : int
        Nombre de répliques à sélectionner par cluster selon sim_stigma
    
    Returns:
    --------
    pd.DataFrame
        DataFrame avec colonnes 'parent_index', 'cluster', 'stigma_intensity'
    """
    print("=== 6.5 STIGMATISATION SOCIALE ET USAGE CACHÉ ===\n")
    
    # Étape 1: Filtrage sémantique
    if 'sim_stigma' not in df_utterances.columns:
        print("⚠️ Colonne 'sim_stigma' non disponible. Exécutez d'abord semantic_anchors.setup_semantic_anchors()")
        return pd.DataFrame()
    
    print("Étape 1: Filtrage sémantique...")
    selected_utterances = []
    for cluster_id in df_utterances['cluster'].dropna().unique():
        cluster_data = df_utterances[df_utterances['cluster'] == cluster_id].copy()
        top_repliques = cluster_data.nlargest(top_n, 'sim_stigma')
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
                    df_selected.loc[orig_idx, 'stigma_signal'] = annotation.get('stigma_signal', 'no')
    else:
        print("⚠️ LLM non disponible, utilisation de valeurs par défaut")
        df_selected['stigma_signal'] = 'no'
    
    # Étape 3: Calcul de l'intensité de stigmatisation par entretien
    print("\nÉtape 3: Calcul de l'intensité de stigmatisation...")
    stigma_profiles = []
    
    for parent_id in df_selected['parent_index'].unique():
        interview_data = df_selected[df_selected['parent_index'] == parent_id]
        
        if len(interview_data) > 0:
            total = len(interview_data)
            stigma_count = (interview_data['stigma_signal'] == 'yes').sum()
            stigma_intensity = stigma_count / total if total > 0 else 0.0
        else:
            stigma_intensity = 0.0
            stigma_count = 0
        
        stigma_profiles.append({
            'parent_index': parent_id,
            'cluster': interview_data['cluster'].iloc[0] if len(interview_data) > 0 else None,
            'stigma_intensity': stigma_intensity,
            'stigma_count': stigma_count
        })
    
    df_stigma_profiles = pd.DataFrame(stigma_profiles)
    
    print(f"✓ {len(df_stigma_profiles)} profils de stigmatisation créés")
    print(f"\n=== STATISTIQUES DE STIGMATISATION ===")
    print(f"  - Moyenne: {df_stigma_profiles['stigma_intensity'].mean():.4f}")
    print(f"  - Médiane: {df_stigma_profiles['stigma_intensity'].median():.4f}")
    print(f"  - Entretiens avec stigmatisation (intensité > 0): {(df_stigma_profiles['stigma_intensity'] > 0).sum()}")
    
    return df_stigma_profiles

def visualize_stigma(df_stigma_profiles):
    """
    Visualise l'intensité de stigmatisation par cluster.
    
    Parameters:
    -----------
    df_stigma_profiles : pd.DataFrame
        DataFrame avec colonnes 'cluster', 'stigma_intensity'
    """
    if len(df_stigma_profiles) == 0:
        print("⚠️ Pas de données à visualiser")
        return
    
    print("\n=== VISUALISATION DE LA STIGMATISATION ===\n")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Graphique 1: Distribution globale
    ax1 = axes[0]
    ax1.hist(df_stigma_profiles['stigma_intensity'], bins=20, color='#FFA07A', alpha=0.7, edgecolor='black')
    ax1.set_title('Distribution de l\'Intensité de Stigmatisation', fontweight='bold')
    ax1.set_xlabel('Intensité de stigmatisation', fontweight='bold')
    ax1.set_ylabel('Nombre d\'entretiens', fontweight='bold')
    ax1.axvline(df_stigma_profiles['stigma_intensity'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Moyenne: {df_stigma_profiles["stigma_intensity"].mean():.3f}')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Graphique 2: Par cluster
    ax2 = axes[1]
    if 'cluster' in df_stigma_profiles.columns:
        cluster_stigma = df_stigma_profiles.groupby('cluster')['stigma_intensity'].mean().sort_index()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(cluster_stigma)]
        bars = ax2.bar([f'Cluster {int(c)}' for c in cluster_stigma.index], cluster_stigma.values, color=colors)
        
        # Ajouter les valeurs sur les barres
        for bar, value in zip(bars, cluster_stigma.values):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{value:.4f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax2.set_title('Intensité de Stigmatisation Moyenne par Cluster', fontweight='bold')
        ax2.set_xlabel('Cluster', fontweight='bold')
        ax2.set_ylabel('Intensité moyenne', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.show()
    
    print("✓ Visualisations générées")
