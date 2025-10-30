import os, re, string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordfreq import zipf_frequency

# One-time downloads (safe to leave in your script)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)   # needed by recent NLTK
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
CUSTOM_STOPWORDS = {
    'discussion', 'conclusion', 'introduction', 'results', 'result',
    'figure', 'table', 'paper', 'manuscript', 'publication', 'author',
    'authors', 'profile', 'affiliation', 'section', 'study', 'studies',
    'work', 'works', 'analysis', 'statistic', 'statistics', 'dataset',
    'data', 'experiment', 'experiments', 'figure', 'caption', 'methodology',
    'keywords', 'abstract', 'index', 'terms', 'acknowledgement',
    'discussion', 'summary', 'appendix', 'reference', 'references',
    'corresponding', 'received', 'accepted', 'online', 'journal',
    'conference', 'vol', 'no', 'et', 'al'
}

STOP = set(stopwords.words('english')).union(CUSTOM_STOPWORDS)

LEMM = WordNetLemmatizer()



# ---------- Helpers ----------

SECTION_STARTS = [
    r'\babstract\b',
    r'\bintroduction\b',
    r'^\s*1[\.\)]?\s*introduction\b',
    r'^\s*i+\.\s*introduction\b',            # I. INTRODUCTION
    r'^\s*background\b'
]
SECTION_ENDS = [
    r'\breferences\b',
    r'\bbibliography\b',
    r'\backnowledg(e)?ments?\b',
    r'\bappendix\b'
]

AFFILIATION_HINTS = [
    'university','institute','department','school','laboratory','centre','center',
    'college','faculty','academy','csir','iit','iiit','nit','google','microsoft',
    'lab','research','science','engineering','technology','hospital'
]
EMAIL_ORCIDs = [r'\b@', r'\borcid\b', r'\b0000-\d{4}-\d{4}-\d{4}\b']

def normalize(text: str) -> str:
    # unify newlines/spaces, fix hyphenation at line breaks
    text = text.replace('\u00ad', '')           # soft hyphen
    text = re.sub(r'-\s*\n\s*', '', text)       # remove linebreak hyphenation
    text = re.sub(r'\n+', '\n', text)
    return text

def strip_front_matter(raw: str) -> str:
    """
    Remove author/affiliation block without requiring an abstract.
    Strategy:
      1) Find the first real section start (Abstract/Introduction/1. Introduction/I. INTRODUCTION).
         If found, keep from there.
      2) Else, drop the leading lines that look like affiliations/emails until we hit a 'normal' paragraph.
    """
    t = normalize(raw)
    # Extract first non-empty line as potential title
    lines = t.splitlines()
    title_line = ""
    for ln in lines:
        if ln.strip() and len(ln.strip().split()) > 2:   # must be a meaningful line
            title_line = ln.strip()
            break



    # Try #1: jump to first clear section start
    starts = [m.start() for pat in SECTION_STARTS for m in re.finditer(pat, t, flags=re.I|re.M)]
    if starts:
        t = t[min(starts):]   # cut before the earliest section marker
    else:
        # Try #2: heuristic line filtering at the top
        lines = t.splitlines()
        kept = []
        dropping = True
        for ln in lines:
            ln_strip = ln.strip()
            # heuristics that suggest "front matter"
            looks_affil = (
                any(h in ln_strip.lower() for h in AFFILIATION_HINTS) or
                any(re.search(p, ln_strip, flags=re.I) for p in EMAIL_ORCIDs) or
                (',' in ln_strip and len(ln_strip) < 140 and not ln_strip.endswith('.')) or
                re.search(r'\bindia\b|\busa\b|\buk\b|\bsingapore\b|\bchina\b|\baustralia\b', ln_strip, flags=re.I)
            )
            # once we encounter a "normal" paragraph, stop dropping
            longish_sentence = len(ln_strip) > 120 and ln_strip.endswith('.')
            sectionish = re.search(r'(abstract|introduction|keywords|index terms)\b', ln_strip, re.I)
            if dropping and (looks_affil or not longish_sentence) and not sectionish:
                continue
            else:
                dropping = False
                kept.append(ln)
        t = '\n'.join(kept)
    if title_line:
        t = title_line + "\n" + t

    return t

def strip_back_matter(text: str) -> str:
    # Cut at References/Bibliography/etc. if present
    ends = [m.start() for pat in SECTION_ENDS for m in re.finditer(pat, text, flags=re.I)]
    return text[:min(ends)] if ends else text

def remove_noise(text: str) -> str:
    # citations like [12], (Smith, 2020), figure/table captions noise, urls
    text = re.sub(r'\[[0-9,\s\-]+\]', ' ', text)                       # [12], [1, 2]
    text = re.sub(r'\(\s*[A-Z][A-Za-z\-]+,\s*\d{4}\s*\)', ' ', text)   # (Smith, 2020)
    text = re.sub(r'(figure|fig\.?|table)\s+\d+[:.\-]?', ' ', text, flags=re.I)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\d{1,4}\s*(%|°[CF]|km|mm|cm|m|hz|khz|mhz|ghz)\b', ' ', text, flags=re.I)
    return text

def basic_preprocess(text: str) -> str:
    # lower, drop punctuation/digits, tokenize, stopword-remove, simple lemma (no POS)
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)            # requires punkt + punkt_tab
    tokens = [w for w in tokens if w not in STOP and len(w) > 2]
    tokens = [LEMM.lemmatize(w) for w in tokens] # POS-agnostic lemmatization
    tokens = [w for w in tokens if w.isalpha() and len(w) > 2 and len(w) < 20]
    tokens = [w for w in tokens if w not in STOP]

    return ' '.join(tokens)

def clean_paper_text(raw_text: str) -> str:
    t = strip_front_matter(raw_text)     # drop authors/affiliations block
    t = strip_back_matter(t)             # drop references/appendix etc.
    t = remove_noise(t)                  # drop citations/urls/units
    t = normalize(t)
    t = basic_preprocess(t)              # simple lemma, no POS
    return t
# def process_corpus(base_dir: str, overwrite=True):
#     for author in os.listdir(base_dir):
#         ap = os.path.join(base_dir, author)
#         if not os.path.isdir(ap):
#             continue
#         for fn in os.listdir(ap):
#             if not fn.lower().endswith('.txt'):
#                 continue
#             p = os.path.join(ap, fn)
#             with open(p, 'r', encoding='utf-8', errors='ignore') as f:
#                 raw = f.read()
#             cleaned = clean_paper_text(raw)
#             outp = p if overwrite else p.replace('.txt', '.clean.txt')
#             with open(outp, 'w', encoding='utf-8') as f:
#                 f.write(cleaned)

# # Run this once:
# base_dir = r"extracted_text"
# process_corpus(base_dir)
