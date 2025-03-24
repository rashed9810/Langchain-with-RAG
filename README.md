# Multilingual RAG System

A powerful Retrieval-Augmented Generation system built with LangChain, FAISS vector store, and multilingual language models. This project specializes in providing accurate information retrieval and natural language generation with a focus on multilingual support, particularly for Japanese content.

## 📋 Overview

This system combines document processing, embedding generation, vector storage, and language model inference to create a comprehensive question-answering system. It can process PDF documents, break them into manageable chunks, convert them into vector embeddings, and then use those embeddings to retrieve relevant information when answering user queries.

## 🚀 Features

- **Document Processing**: Load and process PDF documents
- **Smart Text Chunking**: Intelligent text splitting that preserves context
- **Multilingual Support**: Uses multilingual embedding models for cross-language compatibility
- **Vector-based Retrieval**: FAISS-powered similarity search for fast and accurate information retrieval
- **Japanese Language Focus**: Specialized support for Japanese language through ELYZA's Japanese Llama-2 model
- **Chainlit Integration**: User-friendly web interface for interacting with the system

## 🛠️ Tech Stack

- **LangChain**: Framework for developing applications powered by language models
- **FAISS**: Efficient similarity search and clustering of dense vectors
- **HuggingFace Transformers**: Access to state-of-the-art NLP models
- **PyTorch**: Deep learning framework
- **Chainlit**: Streamlined UI for LLM-powered applications

## 📦 Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU recommended for optimal performance
- Required Python packages (see conda environment file)

## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Set up the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate langchain
   ```

3. Verify the installation:
   ```bash
   python -c "import langchain, torch; print(f'LangChain: {langchain.__version__}, PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
   ```

## 🚀 Quick Start

1. Place your PDF documents in the appropriate directory:
   ```bash
   mkdir -p /media/nsl3090-3/hdd1/hujaifa/Langchain_RAG/PDF/
   cp your-document.pdf /media/nsl3090-3/hdd1/hujaifa/Langchain_RAG/PDF/
   ```

2. Run the RAG system:
   ```bash
   python testing.py
   ```

3. Launch the Chainlit web interface:
   ```bash
   chainlit run app.py
   ```

## 🧩 How It Works

1. **Document Loading**: PDF documents are loaded using PyPDFLoader
2. **Text Splitting**: Documents are split into manageable chunks using RecursiveCharacterTextSplitter
3. **Embedding Generation**: Text chunks are converted to vector embeddings using the multilingual-e5-large model
4. **Vector Storage**: Embeddings are stored in a FAISS vector database for efficient retrieval
5. **Query Processing**: User queries are converted to embeddings and used to retrieve relevant document chunks
6. **Answer Generation**: Retrieved chunks are fed to the LLM (ELYZA-japanese-Llama-2-7b-fast-instruct) to generate comprehensive answers

## 🌐 Multilingual Support

This system is designed with multilingual capabilities:

- **Embedding Model**: Uses intfloat/multilingual-e5-large for cross-language understanding
- **Language Model**: Incorporates ELYZA's Japanese Llama-2 model for native Japanese language processing
- **UI Localization**: Chainlit interface supports both English and Portuguese localization

## 📝 Example Usage

```python
# Load a PDF document
pdf_loader = PyPDFLoader("path/to/your/document.pdf")
documents = pdf_loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20, separators='\n\n\n')
docs = text_splitter.split_documents(documents)

# Generate embeddings and create vector store
embeddings = HuggingFaceEmbeddings(
  model_name="intfloat/multilingual-e5-large",  
  model_kwargs={'device':'cuda:0'},
  encode_kwargs={'normalize_embeddings':False}
)
db = FAISS.from_documents(docs, embeddings)

# Ask a question
question = "人工知能関連の政策を議論する内閣府の「第2回AI戦略会議」の構成員は誰ですか？"
result = qa_chain({"query": question})
print(result["result"])
```

## 🖥️ Web Interface

The project includes a Chainlit-based web interface that allows users to:

- Upload PDF documents
- Ask questions in natural language
- View detailed answers with source citations
- Navigate conversation history
- Toggle between light and dark modes

## 🛡️ Environment Configuration

The system uses a `config.yml` file to configure the Chainlit interface:

- **Telemetry**: Enabled by default (no personal data collected)
- **Session Management**: 1-hour timeout for saved sessions
- **Security**: Configurable CORS settings for API access
- **Features**: Prompt playground, multi-modal uploads, and optional speech-to-text

## 📊 Performance Considerations

- **GPU Acceleration**: Uses CUDA for accelerated model inference
- **Chunk Size Optimization**: Configurable chunk sizes to balance context retention and processing efficiency
- **Model Loading**: Device mapping for optimal GPU memory utilization

## 🌐 Localization

The interface supports multiple languages, with pre-configured translations for:
- English
- Portuguese
- (Additional languages can be added in the UI configuration)

## 🤝 Contributing

Contributions to improve this RAG system are welcome. Please feel free to submit issues and pull requests.

