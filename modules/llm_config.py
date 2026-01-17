"""
Configuration LLM et fonctions utilitaires pour les analyses avancées
Gère l'API Gemini, le cache, et les appels LLM
"""

import os
import json
import hashlib
import time
import requests
from pathlib import Path
from typing import List, Dict, Optional

# === CONSTANTES ===
LLM_PROVIDER = "gemini"
BATCH_SIZE = 8
CACHE_DIR = Path("data/processed")
CACHE_FILE = CACHE_DIR / "llm_cache.json"

# Créer le répertoire de cache si nécessaire
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# === CHARGEMENT DE LA CLÉ API ===
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ python-dotenv disponible, chargement de .env si présent")
except ImportError:
    print("⚠ python-dotenv non disponible, utilisation des variables d'environnement système uniquement")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if not GEMINI_API_KEY or GEMINI_API_KEY.strip() == "":
    print("⚠ GEMINI_API_KEY non définie dans les variables d'environnement")
    print("   → Mode NO-LLM activé : analyses basées uniquement sur embeddings")
    USE_LLM = False
else:
    print("✓ GEMINI_API_KEY détectée")
    USE_LLM = True
    print(f"   Provider: {LLM_PROVIDER}")

print(f"\nMode d'exécution: {'LLM + Embeddings' if USE_LLM else 'Embeddings uniquement'}")

# === CHARGEMENT DU CACHE ===
llm_cache = {}
if CACHE_FILE.exists():
    try:
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            llm_cache = json.load(f)
        print(f"✓ Cache chargé: {len(llm_cache)} entrées")
    except Exception as e:
        print(f"⚠ Erreur lors du chargement du cache: {e}")
        llm_cache = {}
else:
    print("✓ Nouveau cache créé")

# === PROMPT D'ANNOTATION ===
FINAL_ANNOTATION_PROMPT = """You are a research assistant analyzing interview excerpts about AI in the workplace.

For each excerpt below, annotate it INDEPENDENTLY. Do NOT use external knowledge or make global inferences. 
Base your annotation ONLY on what is explicitly stated or clearly implied in that specific excerpt.

For each excerpt, return a JSON object with these exact fields:
- identity_signal: "yes" or "no" (mentions of professional identity, core job definition, refusal to delegate)
- adaptation_strategy: "resistance" or "hybridation" or "requalification" or "contournement" or "exit" or "none"
- normative_discourse: "yes" or "no" (statements about what should be done, moral boundaries)
- descriptive_practice: "yes" or "no" (statements about what is actually done in practice)
- dissonance_signal: "yes" or "no" (conflict between normative and descriptive, contradictions)
- stigma_signal: "yes" or "no" (mentions of judgment, hiding usage, peer criticism)
- inequality_positioning: "advantage" or "survival_pressure" or "neutral"
- trust_level_ai: "high" or "medium" or "low" or "not_applicable"
- justification: brief explanation (max 20 words)

Return ONLY a JSON array of objects, one per excerpt, in the same order. No other text.

Excerpts:
{excerpts}"""

# === FONCTIONS UTILITAIRES ===
def hash_excerpt(text: str) -> str:
    """Génère un hash SHA256 pour un excerpt (utilisé comme clé de cache)"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def validate_json_schema(output: List[Dict]) -> bool:
    """Valide que la sortie LLM contient tous les champs requis"""
    required_fields = [
        'identity_signal', 'adaptation_strategy', 'normative_discourse',
        'descriptive_practice', 'dissonance_signal', 'stigma_signal',
        'inequality_positioning', 'trust_level_ai', 'justification'
    ]
    
    if not isinstance(output, list):
        return False
    
    for item in output:
        if not isinstance(item, dict):
            return False
        for field in required_fields:
            if field not in item:
                return False
    
    return True

def call_llm_json(excerpts: List[str]) -> List[Dict]:
    """
    Appelle l'API LLM pour annoter des excerpts et retourne des annotations JSON structurées.
    Utilise le cache pour éviter les appels répétés.
    
    Parameters:
    -----------
    excerpts : List[str]
        Liste des excerpts à annoter (5-10 phrases max chacun)
    
    Returns:
    --------
    List[Dict]
        Liste de dictionnaires avec les annotations pour chaque excerpt
    """
    if not USE_LLM:
        # Mode fallback : retourner des annotations par défaut
        return [{
            'identity_signal': 'no', 'adaptation_strategy': 'none', 'normative_discourse': 'no',
            'descriptive_practice': 'no', 'dissonance_signal': 'no', 'stigma_signal': 'no',
            'inequality_positioning': 'neutral', 'trust_level_ai': 'not_applicable',
            'justification': 'LLM not available'
        } for _ in excerpts]
    
    results = []
    uncached_excerpts = []
    uncached_indices = []
    
    # Vérifier le cache pour chaque excerpt
    for idx, excerpt in enumerate(excerpts):
        cache_key = hash_excerpt(excerpt)
        if cache_key in llm_cache:
            results.append((idx, llm_cache[cache_key]))
        else:
            uncached_excerpts.append(excerpt)
            uncached_indices.append(idx)
    
    # Appeler l'API pour les excerpts non cachés
    if uncached_excerpts:
        if LLM_PROVIDER == "gemini":
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
                excerpts_text = "\n\n".join([f"Excerpt {i+1}: {ex}" for i, ex in enumerate(uncached_excerpts)])
                prompt_text = FINAL_ANNOTATION_PROMPT.format(excerpts=excerpts_text)
                
                payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
                response = requests.post(url, json=payload, timeout=60)
                response.raise_for_status()
                
                response_data = response.json()
                if 'candidates' in response_data and len(response_data['candidates']) > 0:
                    content = response_data['candidates'][0]['content']['parts'][0]['text']
                    content = content.strip()
                    
                    # Nettoyer le contenu (enlever markdown code blocks si présent)
                    if content.startswith("```"):
                        lines = content.split("\n")
                        content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
                    if content.startswith("```json"):
                        lines = content.split("\n")
                        content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
                    
                    try:
                        api_results = json.loads(content)
                        if not isinstance(api_results, list):
                            api_results = [api_results]
                    except json.JSONDecodeError:
                        # Retry avec un prompt plus strict
                        retry_prompt = prompt_text + "\n\nIMPORTANT: Return ONLY valid JSON array, no other text."
                        payload["contents"][0]["parts"][0]["text"] = retry_prompt
                        retry_response = requests.post(url, json=payload, timeout=60)
                        retry_data = retry_response.json()
                        if 'candidates' in retry_data:
                            retry_content = retry_data['candidates'][0]['content']['parts'][0]['text']
                            retry_content = retry_content.strip().replace("```json", "").replace("```", "").strip()
                            try:
                                api_results = json.loads(retry_content)
                                if not isinstance(api_results, list):
                                    api_results = [api_results]
                            except:
                                api_results = None
                        else:
                            api_results = None
                    
                    if api_results and validate_json_schema(api_results):
                        # Sauvegarder dans le cache
                        for i, (orig_idx, result) in enumerate(zip(uncached_indices, api_results)):
                            cache_key = hash_excerpt(uncached_excerpts[i])
                            llm_cache[cache_key] = result
                            results.append((orig_idx, result))
                    else:
                        print(f"⚠ Erreur de parsing JSON pour {len(uncached_excerpts)} excerpts, utilisation de valeurs par défaut")
                        for i, orig_idx in enumerate(uncached_indices):
                            default_result = {
                                'identity_signal': 'no', 'adaptation_strategy': 'none', 'normative_discourse': 'no',
                                'descriptive_practice': 'no', 'dissonance_signal': 'no', 'stigma_signal': 'no',
                                'inequality_positioning': 'neutral', 'trust_level_ai': 'not_applicable',
                                'justification': 'API parsing error'
                            }
                            cache_key = hash_excerpt(uncached_excerpts[i])
                            llm_cache[cache_key] = default_result
                            results.append((orig_idx, default_result))
                
                # Sauvegarder le cache
                with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                    json.dump(llm_cache, f, indent=2, ensure_ascii=False)
                
            except Exception as e:
                print(f"⚠ Erreur API Gemini: {e}")
                for i, orig_idx in enumerate(uncached_indices):
                    default_result = {
                        'identity_signal': 'no', 'adaptation_strategy': 'none', 'normative_discourse': 'no',
                        'descriptive_practice': 'no', 'dissonance_signal': 'no', 'stigma_signal': 'no',
                        'inequality_positioning': 'neutral', 'trust_level_ai': 'not_applicable',
                        'justification': f'API error: {str(e)[:30]}'
                    }
                    results.append((orig_idx, default_result))
        else:
            print(f"⚠ Provider {LLM_PROVIDER} non implémenté, utilisation de valeurs par défaut")
            for i, orig_idx in enumerate(uncached_indices):
                default_result = {
                    'identity_signal': 'no', 'adaptation_strategy': 'none', 'normative_discourse': 'no',
                    'descriptive_practice': 'no', 'dissonance_signal': 'no', 'stigma_signal': 'no',
                    'inequality_positioning': 'neutral', 'trust_level_ai': 'not_applicable',
                    'justification': 'Provider not implemented'
                }
                results.append((orig_idx, default_result))
    
    # Trier par index original et retourner seulement les résultats
    results.sort(key=lambda x: x[0])
    return [r[1] for r in results]

print(f"✓ Fonction call_llm_json configurée")
print(f"  - Cache: {len(llm_cache)} entrées")
print(f"  - Batch size: {BATCH_SIZE}")
print(f"  - Provider: {LLM_PROVIDER}")
