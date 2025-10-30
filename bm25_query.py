import pickle
from collections import defaultdict
import numpy as np
from pathlib import Path
import fitz  # PyMuPDF
import os

# import your existing cleaner
from preprocessing import clean_paper_text
import os
from pathlib import Path

# Get script directory
BASE_DIR = Path(__file__).parent
PKL_DIR = BASE_DIR / "PKL_files"

bm25 = pickle.load(open(PKL_DIR / "bm25_index.pkl", "rb"))
doc_authors = pickle.load(open(PKL_DIR / "bm25_doc_authors.pkl", "rb"))
doc_titles = pickle.load(open(PKL_DIR / "bm25_doc_titles.pkl", "rb"))

def extract_text_from_pdf(pdf_input):
    if hasattr(pdf_input, "read"):
        pdf_input.seek(0)  # reset pointer before reading
        pdf_bytes = pdf_input.read()
        if not pdf_bytes:
            raise ValueError("Uploaded file stream is empty after reset.")
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    elif isinstance(pdf_input, bytes):
        doc = fitz.open(stream=pdf_input, filetype="pdf")
    elif isinstance(pdf_input, (str, os.PathLike)):
        doc = fitz.open(pdf_input)
    else:
        raise ValueError(f"Unsupported input type for extract_text_from_pdf: {type(pdf_input)}")

    text = "\n".join([p.get_text("text") for p in doc])
    doc.close()
    return text





def bm25_scores_for_query_tokens(query_tokens): #Returns a list of scores aligned to the corpus docs
    return bm25.get_scores(query_tokens)

def aggregate_doc_scores_to_authors(doc_scores, agg="max"): # Aggregate per-document scores up to per-author scores.
    #Returns dict with max, avg, and count for each author.
    author_scores = defaultdict(list)
    for i, score in enumerate(doc_scores):
        author = doc_authors[i]
        author_scores[author].append(score)
    
    # Calculate both max and avg
    author_stats = {}
    for author, scores in author_scores.items():
        author_stats[author] = {
            'max': max(scores),
            'avg': sum(scores) / len(scores),
            'count': len(scores)
        }
    return author_stats
def normalize_scores(author_stats): #Min-max normalize both max and avg scores to [0, 1].
    # Extract max and avg scores
    max_scores = np.array([stats['max'] for stats in author_stats.values()])
    avg_scores = np.array([stats['avg'] for stats in author_stats.values()])
    
    if len(max_scores) == 0:
        return author_stats
    
    # Normalize max scores
    min_max, max_max = max_scores.min(), max_scores.max()
    if max_max == min_max:
        for author in author_stats:
            author_stats[author]['max_normalized'] = 1.0
    else:
        for author in author_stats:
            author_stats[author]['max_normalized'] = (
                (author_stats[author]['max'] - min_max) / (max_max - min_max)
            )
    
    # Normalize avg scores
    min_avg, max_avg = avg_scores.min(), avg_scores.max()
    if max_avg == min_avg:
        for author in author_stats:
            author_stats[author]['avg_normalized'] = 1.0
    else:
        for author in author_stats:
            author_stats[author]['avg_normalized'] = (
                (author_stats[author]['avg'] - min_avg) / (max_avg - min_avg)
            )
    return author_stats

def rank_authors_from_text(raw_text: str, k=10, agg="max"): #    Returns list of (author, rank, max_score, avg_score, num_papers) tuples
    cleaned = clean_paper_text(raw_text)
    query_tokens = cleaned.split()
    doc_scores = bm25_scores_for_query_tokens(query_tokens)
    author_stats = aggregate_doc_scores_to_authors(doc_scores, agg=agg)
    author_stats = normalize_scores(author_stats)
    
    # Sort by max_normalized score
    ranked = sorted(author_stats.items(), 
                   key=lambda x: x[1]['max_normalized'], 
                   reverse=True)[:k]
    
    # Format using NORMALIZED scores
    rankings = [(author, rank+1, stats['max_normalized'], stats['avg_normalized'], stats['count']) 
               for rank, (author, stats) in enumerate(ranked)] 
    return rankings
def rank_authors_from_pdf(pdf_path: str, k=10, agg="max"): #Rank authors from PDF file
    raw = extract_text_from_pdf(pdf_path)
    return rank_authors_from_text(raw, k=k, agg=agg)

def get_bm25_rankings(pdf_path, k=10): #    Returns: List of (author, rank, max_score, avg_score, num_papers) tuples
    raw = extract_text_from_pdf(pdf_path)
    rankings = rank_authors_from_text(raw, k=k, agg="max")
    return rankings

if __name__ == "__main__":
    # Query from a PDF
    pdf_path = r"C:\Users\Hrida\OneDrive\Desktop\Applied AI\Assignment-2\Attention is all you need.pdf"
    rankings = rank_authors_from_pdf(pdf_path, k=10, agg="max")
    
    print("\nTOP 10 RECOMMENDED REVIEWERS (BM25)\n")
    for author, rank, max_sim, avg_sim, num_papers in rankings:
        print(f"{rank}. {author}")
        print(f"   Max similarity: {max_sim:.4f}")
        print(f"   Avg similarity: {avg_sim:.4f}")
        print(f"   Papers: {num_papers}")