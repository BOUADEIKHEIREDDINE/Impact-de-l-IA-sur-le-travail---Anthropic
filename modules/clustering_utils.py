"""
Clustering utilities for user segmentation
Handles keyword extraction and cluster analysis
"""

def get_top_keywords(tfidf, kmeans, n_clusters, n_terms):
    """
    Affiche les mots-clés les plus importants pour chaque cluster.
    
    Parameters:
    -----------
    tfidf : TfidfVectorizer
        Vectoriseur TF-IDF déjà ajusté
    kmeans : KMeans
        Modèle K-Means déjà ajusté
    n_clusters : int
        Nombre de clusters
    n_terms : int
        Nombre de termes à afficher par cluster
    
    Returns:
    --------
    dict
        Dictionnaire {cluster_id: [liste des mots-clés]}
    """
    terms = tfidf.get_feature_names_out()
    centroids = kmeans.cluster_centers_
    
    keywords_by_cluster = {}
    
    for i in range(n_clusters):
        print(f"\n--- CLUSTER {i} ---")
        top_indices = centroids[i].argsort()[-n_terms:][::-1]
        top_keywords = [terms[ind] for ind in top_indices]
        print(top_keywords)
        keywords_by_cluster[i] = top_keywords
    
    return keywords_by_cluster
