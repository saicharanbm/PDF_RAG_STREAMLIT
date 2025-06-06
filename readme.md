# 🤖 AskMyPDF

A powerful Streamlit application that allows you to upload PDF documents and chat with them using AI. Built with RAG (Retrieval-Augmented Generation) architecture using LangChain, Qdrant vector database, and Google's Gemini AI.

## ✨ Features

- 📤 **PDF Upload & Processing**: Upload and automatically index PDF documents
- 💬 **Interactive Chat**: Chat with your documents using natural language
- 🔍 **Smart Search**: Semantic search through document content
- 📄 **Source Citations**: View exact sources and page numbers for answers
- 🔄 **Batch Processing**: Handles large documents with intelligent batching
- 📊 **Progress Tracking**: Real-time progress indicators for document processing
- 🛡️ **Error Handling**: Robust error handling with detailed diagnostics

## 🏗️ Architecture

```
User Upload PDF → PyPDFLoader → Text Splitting → OpenAI Embeddings → Qdrant Vector Store

→ User Query → Similarity Search → Context Retrieval → Google Gemini → AI Response
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- Google AI API key
- Qdrant Cloud account (or local instance)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/saicharanbm/PDF_RAG_STREAMLIT
cd PDF_RAG_STREAMLIT
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
   Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your-openai-api-key-here
GOOGLE_API_KEY=your-google-api-key-here
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key-here
```

4. **Run the application**

```bash
streamlit run main.py
```

## 🔧 Configuration

### Environment Variables

| Variable         | Description                   | Required |
| ---------------- | ----------------------------- | -------- |
| `OPENAI_API_KEY` | OpenAI API key for embeddings | ✅       |
| `GOOGLE_API_KEY` | Google AI API key for Gemini  | ✅       |
| `QDRANT_URL`     | Qdrant cluster URL            | ✅       |
| `QDRANT_API_KEY` | Qdrant API key                | ✅       |

### Configurable Parameters

Edit these in the code as needed:

```python
CHUNK_SIZE = 1000          # Text chunk size for splitting
CHUNK_OVERLAP = 200        # Overlap between chunks
BATCH_SIZE = 10           # Documents per batch
QDRANT_TIMEOUT = 120      # Timeout in seconds
MAX_CONTEXT_CHUNKS = 5    # Max chunks for context
```

## 🎯 Usage

### 1. Upload Documents

1. Select "📤 Upload Document" mode
2. Choose a PDF file (max 50MB)
3. Click "🚀 Process Document"
4. Wait for processing to complete

### 2. Chat with Documents

1. Select "💬 Chat with Document" mode
2. Choose a document from the dropdown
3. Type your question in the chat input
4. View AI responses with source citations

### 3. Managing Collections

- Documents are automatically indexed as collections
- Collection names are based on PDF filenames
- Duplicate names get timestamp suffixes

## 📁 Project Structure

```
pdf-qa-system/
├── main.py              # Main Streamlit application
├── helper.py            # Has all the required Helper function
├── configuration.py     # Has all the configurations
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (create this)
├── README.md            # This file
└── .gitignore           # Git ignore file
```

## 🚀 Deployment

### Local Development

```bash
streamlit run app.py
```

## 📚 Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [Google AI Documentation](https://ai.google.dev/)
