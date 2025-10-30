import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import re
import os
import io
from pathlib import Path

class ReviewerRecommender:  # Sentence Transformer based reviewer recommendation
    def __init__(self, embeddings_path=None):
        if embeddings_path is None:
            BASE_DIR = Path(__file__).parent
            embeddings_path = BASE_DIR / "PKL_files" / "sentence_transformer_embeddings.pkl"
        
        # Load saved embeddings
        with open(embeddings_path, 'rb') as f:
            saved_data = pickle.load(f)
        
        self.embeddings = saved_data['embeddings']
        self.all_paths = saved_data['all_paths']
        self.author_papers = saved_data['author_papers']
        self.model_name = saved_data['model_name']
        # Load sentence transformer model
        self.st_model = SentenceTransformer(self.model_name)
    
    def preprocess_text(self, raw_text): #Minimal preprocessing for transformer models
        text = raw_text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    def extract_text_from_pdf(self, pdf_input):
    

        if isinstance(pdf_input, (str, bytes, os.PathLike)):
            # Case 1: Path to file or raw bytes
            if isinstance(pdf_input, bytes):
                pdf_bytes = pdf_input
            else:
                with open(pdf_input, 'rb') as f:
                    pdf_bytes = f.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        elif hasattr(pdf_input, "read"):
            # Case 2: Streamlit UploadedFile object
            try:
                pdf_input.seek(0)  # Reset the file pointer before reading
            except Exception:
                pass
            pdf_bytes = pdf_input.read()
            if not pdf_bytes:
                raise ValueError("⚠️ Uploaded PDF stream is empty. Try re-uploading the file.")
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        else:
            raise ValueError(f"Unsupported input type for extract_text_from_pdf: {type(pdf_input)}")

        text = "\n".join(page.get_text("text") for page in doc)
        doc.close()
        return text


    def get_rankings(self, new_paper_text, top_k=10): #Get reviewer rankings for new paper(author, rank, max_score, avg_score, num_papers)

        # Truncate to 512 tokens
        tokens = new_paper_text.split()[:512]
        truncated_text = ' '.join(tokens)
        
        # Generate embedding
        new_embedding = self.st_model.encode(truncated_text, convert_to_numpy=True)
        new_embedding_2d = new_embedding.reshape(1, -1)
        
        # Compute similarities
        similarities = cosine_similarity(new_embedding_2d, self.embeddings)[0]
        paper_similarities = {self.all_paths[i]: similarities[i] 
                            for i in range(len(self.all_paths))}    
        
        # Aggregate by author (store both max and avg)
        author_scores = {}
        for author, papers in self.author_papers.items():
            scores = [paper_similarities[paper] for paper in papers 
                    if paper in paper_similarities]      
            
            if scores:
                author_scores[author] = {
                    'max': float(np.max(scores)),
                    'avg': float(np.mean(scores)),
                    'count': len(scores)
                }
        
        # Rank by maximum similarity
        ranked_authors = sorted(author_scores.items(), 
                            key=lambda x: x[1]['max'], 
                            reverse=True)
        
        # Return as (author, rank, max_score, avg_score, num_papers)
        rankings = [(author, rank+1, scores['max'], scores['avg'], scores['count']) 
                for rank, (author, scores) in enumerate(ranked_authors[:top_k])]
        
        return rankings
    
    def recommend_from_pdf(self, pdf_input, top_k=10):
        # Extract and preprocess
        raw_text = self.extract_text_from_pdf(pdf_input)
        processed_text = self.preprocess_text(raw_text)
        
        # Get rankings
        rankings = self.get_rankings(processed_text, top_k)
        
        return rankings
# Standalone function for RRF integration : rankings: List of (author, rank, score) tuples
def get_sentence_transformer_rankings(pdf_path, embeddings_path=None, top_k=10):
    if embeddings_path is None:
        BASE_DIR = Path(__file__).parent
        embeddings_path = BASE_DIR / "PKL_files" / "sentence_transformer_embeddings.pkl"
    
    recommender = ReviewerRecommender(embeddings_path)
    return recommender.recommend_from_pdf(pdf_path, top_k)
if __name__ == "__main__":
    # Initialize recommender
    recommender = ReviewerRecommender(r'C:\Users\Hrida\OneDrive\Desktop\Applied AI\Assignment-2\Main\PKL_files\sentence_transformer_embeddings.pkl')
    
    # Test PDF
    test_pdf = r"C:\Users\Hrida\OneDrive\Desktop\Applied AI\Assignment-2\Attention is all you need.pdf"

    # Get rankings
    print("Getting recommendations...\n")
    rankings = recommender.recommend_from_pdf(test_pdf, top_k=10)
    
    # Display results
    print("TOP 10 RECOMMENDED REVIEWERS (Sentence Transformers)\n")
    for author, rank, max_sim, avg_sim, num_papers in rankings:
        print(f"{rank}. {author}")
        print(f"   Max similarity: {max_sim:.4f}")
        print(f"   Avg similarity: {avg_sim:.4f}")
        print(f"   Papers: {num_papers}")