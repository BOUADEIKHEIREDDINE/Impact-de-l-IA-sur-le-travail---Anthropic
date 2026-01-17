"""
6.2 Identité professionnelle et tâches centrales
Analyse ce que les participants considèrent comme le "cœur" de leur métier.
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from modules.llm_config import call_llm_json, USE_LLM

def detect_identity_mentions(text):
    """
    Détecte les mentions d'identité professionnelle dans un texte.
    
    Parameters:
    -----------
    text : str
        Texte à analyser
    
    Returns:
    --------
    dict
        Dictionnaire avec les catégories détectées (booléens)
    """
    if pd.isna(text) or not isinstance(text, str):
        return {}
    
    text_lower = text.lower()
    results = {}
    
    identity_patterns = {
        'définition_professionnelle': [
            r'\b(what makes me|what defines me|who I am as|my identity as|my role as)\b',
            r'\b(ce qui me définit|mon identité|qui je suis en tant que)\b',
            r'\b(core of my|heart of my|essence of my)\b',
            r'\b(cœur de mon|essence de mon|fondamental)\b'
        ],
        'tâches_refusées': [
            r'\b(won\'t let|refuse to|never let|don\'t want.*to do)\b',
            r'\b(je refuse|je ne laisse pas|jamais laisser)\b',
            r'\b(keep.*myself|do.*myself|handle.*myself)\b',
            r'\b(garder.*moi|faire.*moi|gérer.*moi)\b'
        ],
        'menace_identité': [
            r'\b(threaten|threat|endanger|risk.*identity)\b',
            r'\b(menace|risque.*identité|menacer)\b',
            r'\b(lose.*identity|losing.*who I am)\b',
            r'\b(perdre.*identité|perte.*identité)\b'
        ],
        'redéfinition': [
            r'\b(redefine|reinvent|transform.*role|evolve.*role)\b',
            r'\b(redéfinir|réinventer|transformer.*rôle)\b',
            r'\b(new role|different role|changing role)\b',
            r'\b(nouveau rôle|rôle différent|changement de rôle)\b'
        ]
    }
    
    for category, patterns in identity_patterns.items():
        matches = []
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                matches.append(True)
        results[category] = len(matches) > 0
    
    return results

def analyze_identity(df_utterances, top_n=30):
    """
    Analyse l'identité professionnelle en utilisant embeddings + LLM.
    
    Parameters:
    -----------
    df_utterances : pd.DataFrame
        DataFrame avec colonnes 'parent_index', 'cluster', 'utterance', 'sim_identity'
    top_n : int
        Nombre de répliques à sélectionner par cluster selon sim_identity
    
    Returns:
    --------
    pd.DataFrame
        DataFrame avec colonnes 'parent_index', 'cluster', 'ifi' (Identity Fragility Index)
    """
    print("=== 6.2 IDENTITÉ PROFESSIONNELLE ET TÂCHES CENTRALES ===\n")
    
    # Étape 1: Détection par regex (baseline)
    print("Étape 1: Détection par patterns regex...")
    identity_analysis = []
    for idx, row in df_utterances.iterrows():
        utterance = row['utterance']
        mentions = detect_identity_mentions(utterance)
        
        identity_analysis.append({
            'parent_index': row.get('parent_index', idx),
            'cluster': row.get('cluster', None),
            'utterance': utterance,
            **mentions
        })
    
    df_identity = pd.DataFrame(identity_analysis)
    print(f"✓ {len(df_identity)} répliques analysées")
    
    # Étape 2: Filtrage sémantique + annotation LLM (si disponible)
    if USE_LLM and 'sim_identity' in df_utterances.columns:
        print("\nÉtape 2: Filtrage sémantique + annotation LLM...")
        
        # Sélectionner top-N répliques par cluster selon sim_identity
        selected_utterances = []
        for cluster_id in df_utterances['cluster'].dropna().unique():
            cluster_data = df_utterances[df_utterances['cluster'] == cluster_id].copy()
            top_repliques = cluster_data.nlargest(top_n, 'sim_identity')
            selected_utterances.append(top_repliques)
        
        df_selected = pd.concat(selected_utterances, ignore_index=True)
        
        # Préparer des excerpts (5-10 phrases max, limite 500 caractères)
        excerpts = []
        excerpt_indices = []
        for idx, row in df_selected.iterrows():
            text = row['utterance']
            sentences = text.split('.')
            excerpt = '. '.join(sentences[:10])[:500]
            if len(excerpt) > 50:  # Minimum de 50 caractères
                excerpts.append(excerpt)
                excerpt_indices.append(idx)
        
        if excerpts:
            print(f"  - {len(excerpts)} excerpts préparés pour annotation LLM")
            llm_annotations = call_llm_json(excerpts)
            
            # Fusionner les annotations avec df_selected
            for i, (orig_idx, annotation) in enumerate(zip(excerpt_indices, llm_annotations)):
                if orig_idx in df_identity.index:
                    for key, value in annotation.items():
                        df_identity.loc[orig_idx, f'llm_{key}'] = value
    
    # Étape 3: Calcul de l'IFI (Identity Fragility Index) par cluster
    print("\n=== CALCUL DE L'IFI PAR CLUSTER (POURCENTAGES) ===\n")
    
    fragility_results = []
    fragility_by_cluster = []
    cluster_labels = []
    
    for cluster_id in sorted(df_identity['cluster'].dropna().unique()):
        cluster_data = df_identity[df_identity['cluster'] == cluster_id]
        total_repliques = len(cluster_data)
        
        if total_repliques == 0:
            fragility_by_cluster.append(0.0)
            cluster_labels.append(f'Cluster {int(cluster_id)}')
            continue
        
        # Calculer les POURCENTAGES
        pct_definition = (cluster_data['définition_professionnelle'].sum() / total_repliques * 100) if total_repliques > 0 else 0
        pct_taches_refusees = (cluster_data['tâches_refusées'].sum() / total_repliques * 100) if total_repliques > 0 else 0
        pct_menace = (cluster_data['menace_identité'].sum() / total_repliques * 100) if total_repliques > 0 else 0
        pct_redefinition = (cluster_data['redéfinition'].sum() / total_repliques * 100) if total_repliques > 0 else 0
        
        # Calculer l'IFI basé sur les pourcentages
        fragilité = (
            pct_menace * 2 +           # Menace = +2
            pct_redefinition * 1.5 -     # Redéfinition = +1.5 (adaptation)
            pct_definition * 0.5        # Définition claire = -0.5 (protection)
        ) / 100.0
        
        fragility_results.append({
            'cluster': int(cluster_id),
            'total_repliques': total_repliques,
            'pct_definition': pct_definition,
            'pct_taches_refusees': pct_taches_refusees,
            'pct_menace': pct_menace,
            'pct_redefinition': pct_redefinition,
            'ifi': fragilité
        })
        
        fragility_by_cluster.append(fragilité)
        cluster_labels.append(f'Cluster {int(cluster_id)}')
        
        print(f"Cluster {int(cluster_id)} ({total_repliques} répliques):")
        print(f"  - Définition professionnelle: {pct_definition:.2f}%")
        print(f"  - Tâches refusées: {pct_taches_refusees:.2f}%")
        print(f"  - Menace identité: {pct_menace:.2f}%")
        print(f"  - Redéfinition: {pct_redefinition:.2f}%")
        print(f"  - IFI: {fragilité:.4f}")
        print()
    
    # Créer df_identity_per_user (agrégation par entretien)
    df_identity_per_user = df_identity.groupby('parent_index').agg({
        'définition_professionnelle': 'sum',
        'tâches_refusées': 'sum',
        'menace_identité': 'sum',
        'redéfinition': 'sum',
        'cluster': 'first'
    }).reset_index()
    
    # Calculer l'IFI par entretien
    def calculate_ifi_per_interview(row):
        total = row['définition_professionnelle'] + row['tâches_refusées'] + row['menace_identité'] + row['redéfinition']
        if total == 0:
            return 0.0
        pct_menace = (row['menace_identité'] / total * 100) if total > 0 else 0
        pct_redefinition = (row['redéfinition'] / total * 100) if total > 0 else 0
        pct_definition = (row['définition_professionnelle'] / total * 100) if total > 0 else 0
        ifi = (pct_menace * 2 + pct_redefinition * 1.5 - pct_definition * 0.5) / 100.0
        return ifi
    
    df_identity_per_user['ifi'] = df_identity_per_user.apply(calculate_ifi_per_interview, axis=1)
    
    print("✓ Analyse d'identité professionnelle terminée")
    
    return df_identity, df_identity_per_user, fragility_by_cluster, cluster_labels

def visualize_identity_fragility(fragility_by_cluster, cluster_labels):
    """
    Visualise l'indice de fragilité identitaire par cluster.
    
    Parameters:
    -----------
    fragility_by_cluster : list
        Liste des scores IFI par cluster
    cluster_labels : list
        Liste des labels de clusters
    """
    print("\n=== VISUALISATION DE LA FRAGILITÉ IDENTITAIRE ===\n")
    
    if not isinstance(fragility_by_cluster, list) or len(fragility_by_cluster) == 0:
        print("⚠️ fragility_by_cluster n'est pas une liste valide")
        return
    
    # Graphique unique : Barres avec indice de fragilité par cluster
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(cluster_labels)]
    bars = ax.bar(cluster_labels, fragility_by_cluster, color=colors)
    
    # Ajouter les valeurs sur les barres
    for bar, fragility in zip(bars, fragility_by_cluster):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{fragility:.4f}',
                ha='center', va='bottom' if fragility >= 0 else 'top', 
                fontweight='bold', fontsize=12)
    
    ax.set_title('Indice de Fragilité Identitaire par Cluster\n(basé sur pourcentages)', 
                    fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('Score IFI (normalisé)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.show()
    
    print("✓ Visualisation affichée")
    print(f"  - {len(fragility_by_cluster)} clusters analysés")
    print(f"  - Scores IFI: {[f'{f:.4f}' for f in fragility_by_cluster]}")
