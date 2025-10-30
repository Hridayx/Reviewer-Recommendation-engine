#Re-ranking module: Apply boosts and penalties to RRF results
import pickle
from pathlib import Path

# Load author profiles (load once at module level)
AUTHOR_PROFILES_PATH = r"C:\Users\Hrida\OneDrive\Desktop\Applied AI\Assignment-2\Main\PKL_files\author_profiles.pkl"
try:
    with open(AUTHOR_PROFILES_PATH, 'rb') as f:
        AUTHOR_PROFILES = pickle.load(f)
    print(f"✓ Loaded {len(AUTHOR_PROFILES)} author profiles")
except Exception as e:
    print(f"✗ Error loading author profiles: {e}")
    AUTHOR_PROFILES = {}

# Indian premier institutions
PREMIER_INSTITUTIONS = ['IIT', 'IISc', 'IIIT', 'NIT', 'BITS', 'VIT']

def calculate_experience_boost(num_papers): #Calculate experience boost based on number of papers
    if num_papers >= 30:
        return 1.07
    elif num_papers >= 20:
        return 1.05
    elif num_papers >= 10:
        return 1.02
    elif num_papers >= 5:
        return 1.01
    else:
        return 1.00

def calculate_institution_boost(institution): #Calculate institution boost for Indian premier institutions
    if institution in PREMIER_INSTITUTIONS:
        return 1.02
    else:
        return 1.00

def calculate_recency_boost(recent_papers): #Calculate recency boost based on papers in last 3 years
    if recent_papers >= 3:
        return 1.015
    elif recent_papers >= 1:
        return 1.025
    else:
        return 1.04
def calculate_consistency_boost(avg_similarity): #Calculate consistency boost based on average similarity
    if avg_similarity >= 0.7:
        return 1.05
    elif avg_similarity >= 0.5:
        return 1.025
    else:
        return 1.00

def calculate_penalty(num_papers): #Calculate penalty for authors with very few papers
    if num_papers <= 2:
        return 0.95
    else:
        return 1.00


def get_author_info(author, bm25_rankings, st_rankings): #Get author information from profiles and rankings

    #Args: author: Author name,bm25_rankings: BM25 results list,st_rankings: Sentence Transformer results list

    # Get from author profiles (if available)
    profile = AUTHOR_PROFILES.get(author, {})
    
    # Initialize variables
    bm25_avg = None
    st_avg = None
    num_papers = profile.get('num_papers', 0)
    # Get BM25 avg score
    for auth, rank, max_score, avg_score, papers in bm25_rankings:
        if auth == author:
            bm25_avg = avg_score
            if num_papers == 0:  # Fallback if not in profiles
                num_papers = papers
            break
    # Get ST avg score
    for auth, rank, max_score, avg_score, papers in st_rankings:
        if auth == author:
            st_avg = avg_score
            if num_papers == 0:  # Fallback if not in profiles
                num_papers = papers
            break
    # Calculate average similarity (for consistency boost)
    if bm25_avg is not None and st_avg is not None:
        avg_similarity = (bm25_avg + st_avg) / 2
    elif bm25_avg is not None:
        avg_similarity = bm25_avg
    elif st_avg is not None:
        avg_similarity = st_avg
    else:
        avg_similarity = 0.0
    # Get profile data with safe defaults
    institution = profile.get('primary_institution', 'Other')
    recent_papers = profile.get('recent_papers', 0)
    latest_year = profile.get('latest_year', None)
    return {
        'num_papers': num_papers,
        'institution': institution,
        'recent_papers': recent_papers,
        'latest_year': latest_year,
        'avg_similarity': avg_similarity
    }


def assign_tier(rank): #Assign tier based on rank
    if rank <= 3:
        return "1. Highly Recommended"
    elif rank <= 7:
        return "2. Recommended"
    else:
        return "3. Consider"


def rerank_results(rrf_results, bm25_rankings, st_rankings, top_k=10): #Apply re-ranking with boosts and penalties    
    reranked = []
    
    for author, rrf_score, rrf_details in rrf_results:
        
        # Get author information
        info = get_author_info(author, bm25_rankings, st_rankings)
        
        # Calculate all boosts and penalties
        experience_boost = calculate_experience_boost(info['num_papers'])
        institution_boost = calculate_institution_boost(info['institution'])
        recency_boost = calculate_recency_boost(info['recent_papers'])
        consistency_boost = calculate_consistency_boost(info['avg_similarity'])
        penalty = calculate_penalty(info['num_papers'])
        
        # Calculate final score
        final_score = (rrf_score * 
                      experience_boost * 
                      institution_boost * 
                      recency_boost * 
                      consistency_boost * 
                      penalty)
        
        # Store result
        reranked.append({
            'author': author,
            'final_score': final_score,
            'rrf_score': rrf_score,
            'num_papers': info['num_papers'],
            'institution': info['institution'],
            'recent_papers': info['recent_papers'],
            'latest_year': info['latest_year'],
            'avg_similarity': info['avg_similarity'],
            'boosts': {
                'experience': experience_boost,
                'institution': institution_boost,
                'recency': recency_boost,
                'consistency': consistency_boost,
                'penalty': penalty
            }
        })
    
    # Sort by final score
    reranked.sort(key=lambda x: x['final_score'], reverse=True)
    
    # Take top K
    top_results = reranked[:top_k]
    
    # Normalize scores to 0-100
    if top_results:
        max_score = top_results[0]['final_score']
        
        for i, result in enumerate(top_results):
            # Normalize
            if max_score > 0:
                normalized_score = (result['final_score'] / max_score) * 100
            else:
                normalized_score = 0.0
            
            # Add rank and tier
            result['rank'] = i + 1
            result['tier'] = assign_tier(i + 1)
            result['score'] = round(normalized_score, 2)
            
            # Convert avg_similarity to percentage
            result['avg_similarity_pct'] = round(result['avg_similarity'] * 100, 1)
    
    return top_results

def get_reranked_recommendations(pdf_path, top_k=10): #Main function: Get re-ranked recommendations from PDF
    from RRF_Ensemble import rrf_ensemble
    from bm25_query import get_bm25_rankings
    from Sentence_Transformer import get_sentence_transformer_rankings
    
    print("\n" + "="*80)
    print("GETTING RE-RANKED RECOMMENDATIONS")
    print("="*80)
    
    # Step 1: Get RRF results
    print("\n[1/4] Running RRF ensemble...")
    rrf_results = rrf_ensemble(pdf_path, top_k=20, k=60)
    
    # Step 2: Get BM25 and ST rankings for avg scores
    print("[2/4] Getting BM25 rankings...")
    bm25_rankings = get_bm25_rankings(pdf_path, k=20)
    
    print("[3/4] Getting Sentence Transformer rankings...")
    st_rankings = get_sentence_transformer_rankings(pdf_path, top_k=20)
    
    # Step 3: Apply re-ranking
    print("[4/4] Applying re-ranking with boosts...")
    results = rerank_results(rrf_results, bm25_rankings, st_rankings, top_k=top_k)
    
    print(f"\n✓ Complete! Generated top {len(results)} recommendations\n")
    
    return results

def display_results(results): #Display re-ranked results in formatted manner
    # Group by tier
    tiers = {
        "1. Highly Recommended": [],
        "2. Recommended": [],
        "3. Consider": []
    }
    
    for result in results:
        tiers[result['tier']].append(result)
    
    print("TOP 10 EXPERT RECOMMENDATIONS")
    for tier_name, tier_results in tiers.items():
        if tier_results:
            print(f"\n🏆 {tier_name}")
            print("─"*80)
            for r in tier_results:
                print(f"\n{r['rank']}. {r['author']}")
                print(f"   Score: {r['score']}")
                print(f"   📄 Papers: {r['num_papers']} | 🏛️ {r['institution']} | "
                      f"📅 Latest: {r['latest_year'] if r['latest_year'] else 'N/A'} | "
                      f"🎯 Match: {r['avg_similarity_pct']}%")

if __name__ == "__main__":
    # Test the re-ranking system
    test_pdf = r"C:\Users\Hrida\OneDrive\Desktop\Applied AI\Assignment-2\Attention is all you need.pdf"
    
    # Get re-ranked results
    results = get_reranked_recommendations(test_pdf, top_k=10)
    
    # Display
    display_results(results)