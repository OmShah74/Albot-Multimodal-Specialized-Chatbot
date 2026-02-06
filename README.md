# Multimodal RAG System

Advanced multimodal retrieval-augmented generation system with:
- Multi-resolution knowledge graph
- Vector + Graph + BM25 hybrid retrieval
- Personalized PageRank
- Submodular diversity optimization
- Bayesian weight adaptation
- Multi-LLM fallback with user API keys

## Features

- **Multimodal Input**: Text, PDF, Images, Audio, Video, CSV, URLs, YouTube
- **Advanced Retrieval**: Vector similarity, graph traversal, BM25, structural scores
- **Novel Algorithms**: PPR, MMR, WEA, Thompson Sampling
- **ArangoDB**: Unified vector + graph storage
- **Docker-First**: Local data, scalable storage
- **Zero Infra Cost**: User-provided API keys

## Quick Start

```bash
# Clone repository
git clone <repo-url>
cd multimodal-rag-system

# Start with Docker Compose
docker-compose up -d

# Access at http://localhost:7860
```

## Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download models
python -m spacy download en_core_web_sm

# Start ArangoDB
docker run -p 8529:8529 -e ARANGO_ROOT_PASSWORD=rootpassword arangodb

# Start application
python frontend/app.py
```

## Architecture

- **FastAPI Backend**: Core retrieval engine
- **Gradio Frontend**: User interface
- **ArangoDB**: Graph + vector database
- **Multimodal Processing**: Whisper, CLIP, E5, spaCy

## API Keys

Add your API keys in the UI:
- OpenAI (GPT-4, GPT-3.5)
- Groq (Llama, Mixtral)
- Google Gemini
- Anthropic Claude
- OpenRouter

Multiple keys per provider supported with automatic fallback.

## Documentation

See `/docs` for detailed documentation on:
- Mathematical algorithms
- Graph construction
- Retrieval pipeline
- Configuration options