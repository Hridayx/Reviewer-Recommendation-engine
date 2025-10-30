# Reviewer-Recommendation-engine


Automated Reviewer Recommendation System for top 10 expert reviewers for research papers using hybrid retrieval and multi-factor re-ranking.

**Contributors:**
Hriday macha , Deekshitha karvan
---

## **Overview**

Combines BM25 (lexical) and Sentence Transformers (semantic) retrieval through RRF fusion, followed by re-ranking based on experience, institution, recency, and domain consistency.

**Architecture:**
```
PDF Input → BM25 + Sentence Transformers → RRF Fusion → 
Multi-Factor Re-Ranking → Top 10 (0-100 scores, 3 tiers)
```

---

## **Dataset**

- 637 research papers (Computer Science, AI/ML)
- 71 unique authors
- Pre-computed: BM25 index, ST embeddings, author profiles

---

## **Usage**
### **Web Interface**

```bash
streamlit run streamlit_app.py
```

Upload PDF → View top 10 recommendations with metrics

---

## **How It Works**

### **Stage 1: Hybrid Retrieval**

**BM25:** Exact keyword matching

**Sentence Transformers:** Semantic similarity 

Both return top 20 authors

### **Stage 2: RRF Fusion**

```
RRF_score = 1/(60 + BM25_rank) + 1/(60 + ST_rank)
```

Combines rankings without normalization, rewards consensus

### **Stage 3: Multi-Factor Re-Ranking**

**Five factors:**
1. Experience (publications): 1.00-1.07 boost
2. Institution (IIT/IIIT/NIT/BITS/VIT/IISc): 1.02 boost
3. Recency (last 3 years): 1.015-1.04 boost
4. Domain Consistency (avg similarity): 1.00-1.05 boost
5. Penalty (≤2 papers): 0.95

**Formula:**
```
Final = RRF × Experience × Institution × Recency × Consistency × Penalty
```

### **Stage 4: Normalization**

Scores normalized to 0-100 (top = 100), assigned to 3 tiers:
- Tier 1: Rank 1-3 (Highly Recommended)
- Tier 2: Rank 4-7 (Recommended)
- Tier 3: Rank 8-10 (Consider)

---

## **Output**

```json
{
  "rank": 1,
  "tier": "1. Highly Recommended",
  "author": "Author Name",
  "score": 100.0,
  "num_papers": 12,
  "institution": "IIT",
  "recent_papers": 3,
  "latest_year": 2022,
  "avg_similarity_pct": 78.5
}
```

---

## **Key Libraries**

- PyMuPDF - PDF extraction
- rank-bm25 - BM25 implementation
- sentence-transformers - Embeddings
- scikit-learn - Cosine similarity
- streamlit - Web interface

---

## **Project Structure**

```
├── preprocessing.py                 # Text cleaning
├── bm25_query.py                    # BM25 retrieval
├── Sentence_Transformer.py          # ST retrieval
├── RRF_ensemble.py                  # RRF fusion
├── build_author_profiles.py         # Author metadata
├── reranking.py                     # Re-ranking logic
├── streamlit_app.py                 # Web interface
├── PKL_files/                       # Pre-computed data
├── requirements.txt                 # Dependencies
└── README.md
```

---

## **Technical Contributions**

1. RRF fusion combines lexical and semantic retrieval
2. Domain consistency metric detects specialists
3. Hybrid year extraction (100% detection rate)
4. Multi-factor re-ranking balances relevance with qualifications
5. Interpretable 0-100 scoring with tiers

---

## **License**

MIT License

---
