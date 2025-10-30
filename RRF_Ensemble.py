#Reciprocal Rank Fusion (RRF) Ensemble:
import sys
from collections import defaultdict

# Import both methods
from bm25_query import get_bm25_rankings
from Sentence_Transformer import get_sentence_transformer_rankings

def compute_rrf_scores(rankings_dict, k=60): #Compute RRF scores from multiple ranking methods

    rrf_scores = defaultdict(float)
    # For each method's rankings
    for method_name, rankings in rankings_dict.items():
        for item in rankings:
            author = item[0]  # First element is author name
            rank = item[1]    # Second element is rank
            
            # RRF formula: 1 / (k + rank)
            rrf_scores[author] += 1.0 / (k + rank)
    
    return dict(rrf_scores)

def get_author_details(author, bm25_rankings, st_rankings): #Get detailed information for an author from both methods

    #Returns: Dict with BM25 and Sentence Transformer metrics  
    details = {
        'bm25_rank': None,
        'bm25_score': None,
        'st_rank': None,
        'st_score': None,
        'num_papers': None
    }
    # Find author in BM25 rankings
    for author_name, rank, max_score, avg_score, num_papers in bm25_rankings:
        if author_name == author:
            details['bm25_rank'] = rank
            details['bm25_score'] = max_score
            details['num_papers'] = num_papers
            break
    # Find author in ST rankings
    for author_name, rank, max_score, avg_score, num_papers in st_rankings:
        if author_name == author:
            details['st_rank'] = rank
            details['st_score'] = max_score
            if details['num_papers'] is None:  # Use ST count if BM25 not found
                details['num_papers'] = num_papers
            break  
    return details

def rrf_ensemble(pdf_input, top_k=10, k=60): #List of (author, rrf_score, details_dict) tuples
    
    print("Running RRF Ensemble\n")
    
    # Get rankings from both methods
    print("1/3 Getting BM25 rankings")
    bm25_rankings = get_bm25_rankings(pdf_input, k=20)  # Get top-20 from each
    
    print("2/3 Getting Sentence Transformer rankings")
    st_rankings = get_sentence_transformer_rankings(pdf_input, top_k=20)
    
    print("3/3 Computing RRF scores...\n")
    
    # Prepare rankings dict for RRF
    rankings_dict = {
        'BM25': bm25_rankings,
        'SentenceTransformer': st_rankings
    }
    # Compute RRF scores
    rrf_scores = compute_rrf_scores(rankings_dict, k=k)
    # Sort by RRF score
    ranked_authors = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Get detailed info for top-K authors
    results = []
    for author, rrf_score in ranked_authors[:top_k]:
        details = get_author_details(author, bm25_rankings, st_rankings)
        results.append((author, rrf_score, details))
    return results

def display_rrf_results(results): #Display RRF results
    print("TOP 10 RECOMMENDED REVIEWERS (RRF - Hybrid Ensemble)")
    for i, (author, rrf_score, details) in enumerate(results, 1):
        print(f"\n{i}. {author}")
        print(f"   RRF Score: {rrf_score:.6f}")
        
        # BM25 info
        if details['bm25_rank'] is not None:
            print(f"   BM25: Rank {details['bm25_rank']}, Score {details['bm25_score']:.4f}")
        else:
            print(f"   BM25: Not in top-20")
        # ST info
        if details['st_rank'] is not None:
            print(f"   Sentence Transformers: Rank {details['st_rank']}, Score {details['st_score']:.4f}")
        else:
            print(f"   Sentence Transformers: Not in top-20")
        # Papers count
        if details['num_papers'] is not None:
            print(f"   Papers: {details['num_papers']}")

def get_rrf_rankings(pdf_input, top_k=10): #Simple API function for RRF rankings

    #Args: pdf_path: Path to PDF file,top_k: Number of reviewers to return
    #Returns: List of (author, rank, rrf_score) tuples
    results = rrf_ensemble(pdf_input, top_k=top_k)
    # Format as (author, rank, score)
    rankings = [(author, i+1, rrf_score) 
                for i, (author, rrf_score, details) in enumerate(results)]
    return rankings

if __name__ == "__main__":
    # Test RRF ensemble
    pdf_path = r"C:\Users\karva\OneDrive\Desktop\Reviewer-Recommendation-engine\Clinical Validation of Deep Learning for Segmentation of.pdf"
    
    # Run RRF ensemble
    results = rrf_ensemble(pdf_path, top_k=10, k=60)
    
    # Display results
    display_rrf_results(results)