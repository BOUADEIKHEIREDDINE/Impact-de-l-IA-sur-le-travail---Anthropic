"""
6.7 Tableau récapitulatif des profils
Fusionne tous les profils d'analyse et crée un tableau récapitulatif.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def merge_profiles(df_identity_profiles, df_adaptation_profiles, df_dissonance_profiles,
                   df_stigma_profiles, df_inequality_profiles):
    """
    Fusionne tous les DataFrames de profils en un seul tableau récapitulatif.
    
    Parameters:
    -----------
    df_identity_profiles : pd.DataFrame
        DataFrame avec colonnes 'parent_index', 'ifi'
    df_adaptation_profiles : pd.DataFrame
        DataFrame avec colonnes 'parent_index', 'dominant_strategy'
    df_dissonance_profiles : pd.DataFrame
        DataFrame avec colonnes 'parent_index', 'sdc'
    df_stigma_profiles : pd.DataFrame
        DataFrame avec colonnes 'parent_index', 'stigma_intensity'
    df_inequality_profiles : pd.DataFrame
        DataFrame avec colonnes 'parent_index', 'inequality_label'
    
    Returns:
    --------
    pd.DataFrame
        DataFrame fusionné avec tous les profils
    """
    print("=== 6.7 TABLEAU RÉCAPITULATIF DES PROFILS ===\n")
    
    # Fusionner tous les profils sur 'parent_index'
    df_profiles = pd.DataFrame()
    
    # Commencer par df_identity_profiles (doit contenir 'cluster')
    if len(df_identity_profiles) > 0:
        df_profiles = df_identity_profiles[['parent_index', 'cluster', 'ifi']].copy()
    else:
        print("⚠️ df_identity_profiles vide, création d'un DataFrame de base")
        # Créer un DataFrame minimal si nécessaire
        all_parent_indices = set()
        if len(df_adaptation_profiles) > 0:
            all_parent_indices.update(df_adaptation_profiles['parent_index'].unique())
        if len(df_dissonance_profiles) > 0:
            all_parent_indices.update(df_dissonance_profiles['parent_index'].unique())
        if len(df_stigma_profiles) > 0:
            all_parent_indices.update(df_stigma_profiles['parent_index'].unique())
        if len(df_inequality_profiles) > 0:
            all_parent_indices.update(df_inequality_profiles['parent_index'].unique())
        
        df_profiles = pd.DataFrame({'parent_index': list(all_parent_indices)})
        df_profiles['cluster'] = None
        df_profiles['ifi'] = 0.0
    
    # Fusionner les autres profils
    if len(df_adaptation_profiles) > 0:
        df_profiles = df_profiles.merge(
            df_adaptation_profiles[['parent_index', 'dominant_strategy']],
            on='parent_index', how='outer'
        )
    
    if len(df_dissonance_profiles) > 0:
        df_profiles = df_profiles.merge(
            df_dissonance_profiles[['parent_index', 'sdc']],
            on='parent_index', how='outer'
        )
    
    if len(df_stigma_profiles) > 0:
        df_profiles = df_profiles.merge(
            df_stigma_profiles[['parent_index', 'stigma_intensity']],
            on='parent_index', how='outer'
        )
    
    if len(df_inequality_profiles) > 0:
        df_profiles = df_profiles.merge(
            df_inequality_profiles[['parent_index', 'inequality_label']],
            on='parent_index', how='outer'
        )
    
    # Remplir les valeurs manquantes
    df_profiles['ifi'] = df_profiles['ifi'].fillna(0.0)
    df_profiles['sdc'] = df_profiles.get('sdc', pd.Series(0.0)).fillna(0.0)
    df_profiles['stigma_intensity'] = df_profiles.get('stigma_intensity', pd.Series(0.0)).fillna(0.0)
    df_profiles['dominant_strategy'] = df_profiles.get('dominant_strategy', pd.Series('none')).fillna('none')
    df_profiles['inequality_label'] = df_profiles.get('inequality_label', pd.Series('neutral')).fillna('neutral')
    
    print(f"✓ {len(df_profiles)} profils fusionnés")
    print(f"\nColonnes: {list(df_profiles.columns)}")
    
    return df_profiles

def visualize_profiles_summary(df_profiles):
    """
    Visualise un résumé des profils par cluster.
    
    Parameters:
    -----------
    df_profiles : pd.DataFrame
        DataFrame avec tous les profils fusionnés
    """
    if len(df_profiles) == 0:
        print("⚠️ Pas de données à visualiser")
        return
    
    print("\n=== VISUALISATION DU RÉSUMÉ DES PROFILS ===\n")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Graphique 1: IFI moyen par cluster
    ax1 = axes[0, 0]
    if 'cluster' in df_profiles.columns and 'ifi' in df_profiles.columns:
        cluster_ifi = df_profiles.groupby('cluster')['ifi'].mean().sort_index()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(cluster_ifi)]
        bars = ax1.bar([f'Cluster {int(c)}' for c in cluster_ifi.index], cluster_ifi.values, color=colors)
        for bar, value in zip(bars, cluster_ifi.values):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        ax1.set_title('IFI Moyen par Cluster', fontweight='bold')
        ax1.set_ylabel('IFI moyen', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Graphique 2: SDC moyen par cluster
    ax2 = axes[0, 1]
    if 'cluster' in df_profiles.columns and 'sdc' in df_profiles.columns:
        cluster_sdc = df_profiles.groupby('cluster')['sdc'].mean().sort_index()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(cluster_sdc)]
        bars = ax2.bar([f'Cluster {int(c)}' for c in cluster_sdc.index], cluster_sdc.values, color=colors)
        for bar, value in zip(bars, cluster_sdc.values):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        ax2.set_title('SDC Moyen par Cluster', fontweight='bold')
        ax2.set_ylabel('SDC moyen', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Graphique 3: Intensité de stigmatisation par cluster
    ax3 = axes[1, 0]
    if 'cluster' in df_profiles.columns and 'stigma_intensity' in df_profiles.columns:
        cluster_stigma = df_profiles.groupby('cluster')['stigma_intensity'].mean().sort_index()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(cluster_stigma)]
        bars = ax3.bar([f'Cluster {int(c)}' for c in cluster_stigma.index], cluster_stigma.values, color=colors)
        for bar, value in zip(bars, cluster_stigma.values):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        ax3.set_title('Intensité de Stigmatisation Moyenne par Cluster', fontweight='bold')
        ax3.set_ylabel('Intensité moyenne', fontweight='bold')
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Graphique 4: Stratégies d'adaptation par cluster
    ax4 = axes[1, 1]
    if 'cluster' in df_profiles.columns and 'dominant_strategy' in df_profiles.columns:
        cluster_strategy = pd.crosstab(df_profiles['cluster'], df_profiles['dominant_strategy'])
        cluster_strategy.plot(kind='bar', ax=ax4, stacked=True, colormap='Set3')
        ax4.set_title('Stratégies d\'Adaptation par Cluster', fontweight='bold')
        ax4.set_xlabel('Cluster', fontweight='bold')
        ax4.set_ylabel('Nombre d\'entretiens', fontweight='bold')
        ax4.legend(title='Stratégie', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.tick_params(axis='x', rotation=0)
        ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.suptitle('Résumé des Profils par Cluster', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    print("✓ Visualisations du résumé générées")
