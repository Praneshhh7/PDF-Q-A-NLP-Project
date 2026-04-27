# NLP Document Retrieval & Semantic Search Pipeline

This project contains a comprehensive Natural Language Processing (NLP) pipeline designed for document-based Information Retrieval and Semantic Search. The core notebook (`NLP_FINAL.ipynb`) extracts text from a PDF document, applies rigorous text preprocessing, and implements **8 different text representation and embedding models** to retrieve the most contextually relevant sentences based on user queries using Cosine Similarity.

## ✨ Features

- **Automated Text Extraction:** Utilizes `PyMuPDF` (fitz) to seamlessly extract raw text from a given PDF document.
- **Rigorous Text Preprocessing:** Custom tokenization, stopword removal, lemmatization, and lowercasing implemented via the `NLTK` library.
- **Comparative Embedding Implementations:** Evaluates and compares classical ML structures against state-of-the-art localized and pre-trained Transformer embeddings:
  1. **Bag of Words (BoW)** (`sklearn`)
  2. **TF-IDF** (`sklearn`)
  3. **Word2Vec (CBOW)** (`gensim`)
  4. **Word2Vec (Skip-gram)** (`gensim`)
  5. **GloVe Representation** (`gensim`)
  6. **FastText** (`gensim`)
  7. **Sentence Transformers (SBERT)** (`all-MiniLM-L6-v2`)
  8. **BERT Base Uncased** (`transformers`, `torch`)
- **Interactive Prompt System:** An interactive command-line interface within the notebook that accepts user questions, vectorizes the query using the user-selected algorithm, computes the cosine similarity against the document corpus, and outputs the top matching sentences.

## 🛠️ Requirements

Ensure you have the following Python libraries installed before executing the Jupyter Notebook:

```bash
pip install pymupdf nltk scikit-learn gensim pandas numpy sentence-transformers transformers torch
```

*Note: The notebook also programmatically downloads necessary NLTK corpora (`punkt`, `stopwords`, `wordnet`, `averaged_perceptron_tagger`) upon execution.*

## 🚀 Usage

1. **Prepare Data:** Ensure your target PDF document is accessible by the notebook. The default path configured in the code expects a PDF to parse (e.g., `Downloads/NLP.pdf`).
2. **Run the Notebook:** Execute the cells sequentially in `NLP_FINAL.ipynb`. It will perform extraction, build the corpus, and train/load the designated embedding models.
3. **Query the System:** In the final interactive code block, you will be prompted to:
   - **Enter your question:** Provide a natural language query based on the document's contents.
   - **Choose Model:** Select an algorithm by entering a number from `1` to `8` corresponding to the embedding strategies above.
   - **Review Output:** The engine evaluates cosine similarities across the corpus and prints out the most relevant extraction texts.

## 🧠 Architecture Overview

- **Layer 1: Data Ingestion** - PDF parsing, chunking into workable sentence-level sizes.
- **Layer 2: Lexical Processing** - Cleaning text to remove grammatical noise and standardize formats.
- **Layer 3: Vectorization & Training** - Generating vector spaces using both frequency-based metrics (TF-IDF/BoW) and neural-network-based embeddings (Word2Vec/FastText/BERT).
- **Layer 4: Search & Retrieval Engine** - Projecting user queries into the established vector space and filtering results locally using Cosine Similarity metrics.

## 📂 Repository Structure

- `NLP_FINAL.ipynb` - The primary Jupyter Notebook containing the full implementation of the pipeline.
- `NLP.pdf` - *(Optional)* Place your source PDF document inside the directory alongside the Jupyter script.
