# Document Similarity & Plagiarism Detection

A semantic plagiarism detection system that uses SBERT embeddings and cosine similarity to identify paraphrased and contextually similar content across documents.

## Features

- **Semantic Similarity** — Detects paraphrasing and idea-level plagiarism, not just exact matches
- **SBERT Embeddings** — Generates high-quality sentence embeddings using Sentence-BERT
- **Cosine Similarity Scoring** — Measures document similarity on a 0–1 scale
- **Automated Scoring** — Reduces manual document review time significantly
- **Batch Processing** — Compare multiple documents in a single run

## Tech Stack

| Component | Technology |
|---|---|
| Embeddings | SBERT (sentence-transformers) |
| Similarity Metric | Cosine Similarity |
| Data Processing | NumPy, Pandas |
| Language | Python |

## How It Works

1. **Load Documents** — Input two or more text documents
2. **Embed** — Each document (or sentence) is encoded into a dense vector using SBERT
3. **Compare** — Cosine similarity is computed between embeddings
4. **Score** — A similarity score is returned; scores above a threshold flag potential plagiarism
5. **Report** — Output highlights the most similar sections across documents

```
Document A  ──►  SBERT Encoder  ──►  Vector A ──┐
                                                  ├──► Cosine Similarity ──► Score
Document B  ──►  SBERT Encoder  ──►  Vector B ──┘
```

## Getting Started

### Prerequisites

```bash
Python 3.9+
```

### Installation

```bash
git clone https://github.com/your-username/plagiarism-detection.git
cd plagiarism-detection
pip install -r requirements.txt
```

### Usage

```python
from detector import PlagiarismDetector

detector = PlagiarismDetector()
score = detector.compare("path/to/doc1.txt", "path/to/doc2.txt")
print(f"Similarity Score: {score:.2f}")
```

### Run via CLI

```bash
python detect.py --doc1 file1.txt --doc2 file2.txt
```

## Project Structure

```
plagiarism-detection/
├── detect.py            # CLI entry point
├── detector.py          # Core similarity logic
├── embedder.py          # SBERT embedding wrapper
├── utils.py             # File loading & preprocessing
├── requirements.txt
└── README.md
```

## Similarity Score Interpretation

| Score Range | Interpretation |
|---|---|
| 0.90 – 1.00 | Near-identical / direct copy |
| 0.75 – 0.89 | High similarity — likely paraphrased |
| 0.50 – 0.74 | Moderate similarity — possible overlap |
| Below 0.50  | Low similarity — likely original |

## Future Improvements

- Sentence-level granularity for exact passage highlighting
- Web interface for easy document upload
- Support for PDF and DOCX formats
- Multi-language plagiarism detection

