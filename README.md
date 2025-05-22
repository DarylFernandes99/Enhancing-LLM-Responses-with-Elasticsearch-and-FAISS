# Enhancing LLM Responses with Elasticsearch and FAISS

![Project Banner](./project-banner.png)

A sophisticated Retrieval-Augmented Generation (RAG) system that significantly improves Large Language Model responses by implementing a hybrid search architecture that combines the precision of Elasticsearch's keyword matching with the semantic understanding of FAISS vector search.

## 🎯 Project Overview

This project demonstrates how to enhance the factual accuracy, relevance, and contextual awareness of Large Language Model responses through a hybrid retrieval system. By combining two complementary search methodologies, the system retrieves the most relevant information to enrich LLM prompts, resulting in more accurate and context-aware responses.

### Key Innovation
The hybrid approach leverages:
- **Elasticsearch**: BM25 algorithm for precise keyword-based retrieval
- **FAISS**: Dense vector similarity search for semantic understanding
- **Intelligent Fusion**: Smart combination and deduplication of results
- **Prompt Engineering**: Structured context injection for optimal LLM performance

## ✨ Features

### Core Capabilities
- **🔍 Hybrid Search Architecture**: Seamlessly combines keyword and semantic search methodologies
- **⚡ High-Performance Retrieval**: Optimized for speed with Elasticsearch and FAISS
- **🧠 Semantic Understanding**: Uses state-of-the-art sentence transformers for embeddings
- **🔗 LLM Integration**: Ready-to-use integration with Google Gemini API
- **📊 Response Comparison**: Built-in demonstration of improvement in answer quality
- **🛠️ Modular Design**: Easy to extend and customize for different use cases

### Technical Features
- **Document Indexing**: Dual indexing system for both keyword and vector search
- **Query Processing**: Intelligent query handling for optimal retrieval
- **Result Fusion**: Advanced algorithms for combining and ranking results
- **Context Formatting**: Structured prompt enrichment for LLM consumption
- **Error Handling**: Robust error management and fallback mechanisms

## 🚀 Quick Start

### Prerequisites

Ensure you have the following installed:
- **Python 3.8+**
- **Java 8+** (for Elasticsearch)
- **Git** (for cloning the repository)

### Installation

1. **Install Python Dependencies**
   ```bash
   pip install elasticsearch==7.13.4 sentence-transformers faiss-cpu google-generativeai
   ```

2. **Set Up Elasticsearch**
   
   The project includes automated Elasticsearch setup:
   ```bash
   # Download and extract Elasticsearch
   wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-oss-7.9.2-linux-x86_64.tar.gz
   tar -xzf elasticsearch-oss-7.9.2-linux-x86_64.tar.gz
   
   # Start Elasticsearch
   sudo chown -R daemon:daemon elasticsearch-7.9.2/
   sudo -H -u daemon elasticsearch-7.9.2/bin/elasticsearch &
   ```

3. **Configure API Keys**
   
   Set up your Google Gemini API key:
   ```python
   API_KEY = 'your-google-gemini-api-key-here'
   ```

### Quick Example

```python
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai

# Initialize components
es = Elasticsearch(hosts=["http://localhost:9200"])
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
faiss_index = faiss.IndexFlatL2(384)

# Query the system
query = "What are the recent changes in global oil prices?"
results = hybrid_search(query)
enhanced_response = generate_response(enrich_prompt(query, results))
```

## 📖 Detailed Usage

### Running the Jupyter Notebook

1. **Start Jupyter Notebook**
   ```bash
   jupyter notebook Enhancing_LLM_Responses_with_Elasticsearch_and_FAISS.ipynb
   ```

2. **Follow the Notebook Sections**:
   - **Setup**: Install dependencies and configure Elasticsearch
   - **Indexing**: Load and index sample documents
   - **Querying**: Perform hybrid searches
   - **Enhancement**: Generate enhanced LLM responses
   - **Comparison**: Compare results with and without context

### Document Indexing

The system indexes documents into both search engines simultaneously:

```python
# Sample document structure
document = {
    "title": "Oil Prices Hit Three-Year Low",
    "content": "Global oil demand, led by a slowdown in China, has caused a sharp drop in oil prices..."
}

# Index into Elasticsearch (BM25)
es.index(index="data_index", id=doc_id, body=document)

# Generate embedding and add to FAISS
embedding = embedding_model.encode([document['title'], document['content']])[0]
faiss_index.add(np.array([embedding]))
```

### Hybrid Search Process

The search process combines both methodologies:

```python
def hybrid_search(query, top_k_bm25=2, top_k_semantic=2):
    # BM25 keyword search
    bm25_docs = search_elasticsearch(query, top_k=top_k_bm25)
    
    # Semantic vector search
    faiss_docs = search_faiss(query, top_k=top_k_semantic)
    
    # Combine and deduplicate results
    unique_docs = {doc['content']: doc for doc in bm25_docs + faiss_docs}.values()
    return list(unique_docs)
```

## 🏗️ Architecture Deep Dive

### System Components

```
┌─────────────────┐    ┌─────────────────┐
│   User Query    │    │   Documents     │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          v                      v
┌─────────────────┐    ┌─────────────────┐
│ Query Processing│    │Document Indexing│
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          v                      v
┌─────────────────┐    ┌─────────────────┐
│ Parallel Search │    │  Dual Storage   │
│ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │Elasticsearch│ │    │ │Elasticsearch│ │
│ │   (BM25)    │ │    │ │   Index     │ │
│ └─────────────┘ │    │ └─────────────┘ │
│ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │    FAISS    │ │    │ │    FAISS    │ │
│ │  (Semantic) │ │    │ │   Vectors   │ │
│ └─────────────┘ │    │ └─────────────┘ │
└─────────┬───────┘    └─────────────────┘
          │
          v
┌─────────────────┐
│ Result Fusion   │
└─────────┬───────┘
          │
          v
┌─────────────────┐
│Prompt Enrichment│
└─────────┬───────┘
          │
          v
┌─────────────────┐
│ LLM Generation  │
└─────────┬───────┘
          │
          v
┌─────────────────┐
│Enhanced Response│
└─────────────────┘
```

### Data Flow

1. **Document Preprocessing**: Clean and structure input documents
2. **Dual Indexing**: Simultaneously index in Elasticsearch and generate FAISS embeddings
3. **Query Analysis**: Process user queries for optimal search performance
4. **Parallel Retrieval**: Execute both keyword and semantic searches
5. **Result Fusion**: Combine, rank, and deduplicate retrieved documents
6. **Context Integration**: Format results for LLM consumption
7. **Response Generation**: Generate enhanced responses using enriched prompts

## 📊 Performance Comparison

### Example: Oil Price Query

**Query**: "Was there any changes in global oil prices this year?"

#### Standard LLM Response (Without Context)
```
I do not have access to real-time information, including constantly changing data like global oil prices.

To get the most up-to-date information on global oil prices, I recommend checking reputable financial news sources...
```

#### Enhanced LLM Response (With Context)
```
Yes, there have been significant changes in global oil prices this year.

The provided information states that oil prices have hit a three-year low, with Brent crude falling to $70 per barrel in early September 2024. This drop is attributed to a slowdown in global oil demand, particularly due to economic challenges in China.

Therefore, the answer is yes, there have been changes in global oil prices this year, with prices dropping to a three-year low.
```

### Key Improvements
- ✅ **Factual Accuracy**: Provides specific, verifiable information
- ✅ **Current Relevance**: Includes recent data and trends  
- ✅ **Source Attribution**: References specific information sources
- ✅ **Comprehensive Coverage**: Addresses all aspects of the query
- ✅ **Contextual Understanding**: Explains underlying causes

## 🔧 Configuration & Customization

### Elasticsearch Configuration

```python
# Custom Elasticsearch settings
es_config = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "custom_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "stop"]
                }
            }
        }
    }
}
```

### FAISS Index Options

```python
# Different FAISS index types for various use cases
index_flat = faiss.IndexFlatL2(dimension)           # Exact search
index_ivf = faiss.IndexIVFFlat(quantizer, dimension, nlist)  # Faster search
index_hnsw = faiss.IndexHNSWFlat(dimension, M)      # Graph-based search
```

### Embedding Model Alternatives

```python
# Various embedding models for different requirements
models = {
    'lightweight': 'all-MiniLM-L6-v2',           # Fast, good quality
    'balanced': 'all-mpnet-base-v2',             # Best quality/speed trade-off
    'high_quality': 'sentence-transformers/all-roberta-large-v1',  # Highest quality
    'multilingual': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
}
```

## 🎛️ Advanced Features

### Custom Scoring and Ranking

```python
def advanced_hybrid_search(query, alpha=0.7):
    """
    Advanced hybrid search with custom scoring
    alpha: weight for BM25 vs semantic scores
    """
    bm25_results = search_elasticsearch(query, return_scores=True)
    semantic_results = search_faiss(query, return_scores=True)
    
    # Normalize and combine scores
    combined_scores = {}
    for doc, score in bm25_results:
        combined_scores[doc['id']] = alpha * score
    
    for doc, score in semantic_results:
        if doc['id'] in combined_scores:
            combined_scores[doc['id']] += (1-alpha) * score
        else:
            combined_scores[doc['id']] = (1-alpha) * score
    
    # Return top-k results
    return sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
```

### Query Expansion

```python
def expand_query(original_query, model):
    """Expand queries using LLM for better retrieval"""
    expansion_prompt = f"""
    Given the query: "{original_query}"
    Generate 3 related queries that might help find relevant information:
    """
    expanded = model.generate_content(expansion_prompt)
    return [original_query] + parse_expanded_queries(expanded.text)
```

### Dynamic Context Weighting

```python
def dynamic_context_selection(retrieved_docs, query, max_context_length=1000):
    """Intelligently select most relevant context within token limits"""
    scored_docs = []
    for doc in retrieved_docs:
        relevance_score = calculate_relevance(doc, query)
        scored_docs.append((doc, relevance_score))
    
    # Select docs that fit within context window
    selected_docs = []
    current_length = 0
    
    for doc, score in sorted(scored_docs, key=lambda x: x[1], reverse=True):
        doc_length = len(doc['content'].split())
        if current_length + doc_length <= max_context_length:
            selected_docs.append(doc)
            current_length += doc_length
    
    return selected_docs
```

## 🧪 Testing & Evaluation

### Running Tests

```python
# Test retrieval quality
def evaluate_retrieval(test_queries, ground_truth):
    results = {}
    for query, expected_docs in test_queries.items():
        retrieved_docs = hybrid_search(query)
        precision, recall, f1 = calculate_metrics(retrieved_docs, expected_docs)
        results[query] = {'precision': precision, 'recall': recall, 'f1': f1}
    return results

# Test response quality
def evaluate_responses(test_queries):
    for query in test_queries:
        baseline_response = generate_response(query)
        enhanced_response = generate_enhanced_response(query)
        
        print(f"Query: {query}")
        print(f"Baseline: {baseline_response}")
        print(f"Enhanced: {enhanced_response}")
        print("-" * 50)
```

### Benchmark Datasets

The system can be evaluated against standard datasets:
- **MS MARCO**: Web search questions
- **Natural Questions**: Real Google search queries  
- **TREC**: Question answering benchmarks
- **Custom Domain**: Your specific use case data

## 🚀 Deployment Options

### Local Development
```bash
# Run everything locally
python -m jupyter notebook
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]
```

### Cloud Deployment
- **AWS**: EC2 with Elasticsearch Service
- **Google Cloud**: Compute Engine with Vertex AI
- **Azure**: Virtual Machines with Cognitive Search

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Areas for Contribution
- **Additional LLM Integrations**: OpenAI GPT, Anthropic Claude, etc.
- **Enhanced Retrieval Methods**: Graph-based search, learned sparse retrieval
- **Evaluation Frameworks**: Automated testing and benchmarking
- **UI Development**: Web interface for easier interaction
- **Documentation**: Tutorials, examples, and guides

### Development Setup
```bash
git clone https://github.com/yourusername/enhancing-llm-responses.git
cd enhancing-llm-responses
pip install -e .
pip install -r requirements-dev.txt
pre-commit install
```

### Submission Process
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🐛 Troubleshooting

### Common Issues

**Elasticsearch Connection Failed**
```bash
# Check if Elasticsearch is running
curl -X GET "localhost:9200/"

# Restart if needed
sudo systemctl restart elasticsearch
```

**FAISS Import Error**
```bash
# Install CPU version
pip uninstall faiss-gpu
pip install faiss-cpu

# Or GPU version (if CUDA available)
pip install faiss-gpu
```

**Memory Issues with Large Datasets**
```python
# Use batch processing
def batch_index_documents(documents, batch_size=100):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        process_batch(batch)
```

**API Rate Limiting**
```python
import time
from functools import wraps

def rate_limit(calls_per_second=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            time.sleep(1.0 / calls_per_second)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limit(0.5)  # 0.5 calls per second
def generate_response(prompt):
    return model.generate_content(prompt)
```

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

### What this means:
- ✅ **Commercial Use**: You can use this software commercially
- ✅ **Modification**: You can modify the source code
- ✅ **Distribution**: You can distribute the software
- ✅ **Private Use**: You can use it privately
- ❗ **License and Copyright Notice**: Must include license and copyright notice
- ❗ **State Changes**: Must document changes made to the code
- ❗ **Disclose Source**: Must make source code available when distributing

## 🙏 Acknowledgments

- **Elasticsearch Team**: For the powerful search engine
- **Facebook AI Research**: For FAISS vector search library
- **Sentence Transformers**: For high-quality embedding models
- **Google AI**: For the Gemini API
- **Open Source Community**: For countless contributions and feedback

---

**⭐ If you find this project helpful, please consider giving it a star!**

---

*Built with ❤️ for the AI and NLP community*
