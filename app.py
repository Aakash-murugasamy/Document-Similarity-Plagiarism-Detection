# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import re
from config import SETTINGS

app = Flask(__name__)
CORS(app)

# Load SBERT model once
sbert_model = SentenceTransformer(SETTINGS['sbert_model_name'])


def clean_text(text):
    """Basic text cleaning: lowercasing, removing extra spaces."""
    return re.sub(r'\s+', ' ', text.strip().lower())


def tfidf_cosine(a, b):
    """Compute cosine similarity between TF-IDF vectors of two texts."""
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform([a, b])
    sim = cosine_similarity(tfidf[0:1], tfidf[1:2])
    return float(sim[0][0])


def sbert_cosine(a, b):
    """Compute cosine similarity between SBERT embeddings."""
    emb = sbert_model.encode([a, b], convert_to_numpy=True)
    a_emb, b_emb = emb[0], emb[1]
    num = np.dot(a_emb, b_emb)
    den = np.linalg.norm(a_emb) * np.linalg.norm(b_emb)
    if den == 0:
        return 0.0
    return float(num / den)


def top_matching_ngrams(a, b, n=3, min_len=3):
    """Find top overlapping n-grams (1 to n) between two documents."""
    a_tokens = [t for t in re.findall(r"\w+", a.lower()) if len(t) >= min_len]
    b_tokens = [t for t in re.findall(r"\w+", b.lower()) if len(t) >= min_len]
    matches = set()
    for k in range(1, n + 1):
        a_grams = set([' '.join(a_tokens[i:i + k]) for i in range(len(a_tokens) - k + 1)])
        b_grams = set([' '.join(b_tokens[i:i + k]) for i in range(len(b_tokens) - k + 1)])
        inter = a_grams.intersection(b_grams)
        for it in inter:
            matches.add((k, it))
    sorted_matches = sorted(list(matches), key=lambda x: (-x[0], x[1]))
    return [{'n': m[0], 'text': m[1]} for m in sorted_matches[:30]]


@app.route('/api/compare', methods=['POST'])
def compare_docs():
    """Main API endpoint to compare two documents."""
    payload = request.json
    a = clean_text(payload.get('a', ''))
    b = clean_text(payload.get('b', ''))

    if not a or not b:
        return jsonify({'error': 'Both documents (a and b) are required.'}), 400

    # Compute similarity scores
    tfidf_score = tfidf_cosine(a, b)
    sbert_score = sbert_cosine(a, b)
    ngram_matches = top_matching_ngrams(a, b, n=3)

    # Thresholds (allow user override)
    thresholds = SETTINGS['thresholds'].copy()
    user_thresh = payload.get('thresholds') or {}
    thresholds.update(user_thresh)

    tfidf_flag = tfidf_score >= thresholds['tfidf']
    sbert_flag = sbert_score >= thresholds['sbert']
    final_flag = tfidf_flag or sbert_flag  # OR logic; change to AND if needed

    response = {
        'tfidf_score': round(tfidf_score, 4),
        'sbert_score': round(sbert_score, 4),
        'tfidf_flag': tfidf_flag,
        'sbert_flag': sbert_flag,
        'final_flag': final_flag,
        'thresholds_used': thresholds,
        'ngram_matches': ngram_matches
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
