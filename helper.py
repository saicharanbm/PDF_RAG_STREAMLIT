import os
import tempfile
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from qdrant_client import QdrantClient
from configuration import QDRANT_TIMEOUT,BATCH_SIZE,CHUNK_SIZE,CHUNK_OVERLAP,MAX_CONTEXT_CHUNKS
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage


def initialize_clients():
    """Initialize Qdrant client and embedding model with caching"""
    try:
        qdrant_host = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Validate environment variables
        missing_vars = []
        if not qdrant_host:
            missing_vars.append("QDRANT_URL")
        if not qdrant_api_key:
            missing_vars.append("QDRANT_API_KEY")
        if not openai_api_key:
            missing_vars.append("OPENAI_API_KEY")
        
        if missing_vars:
            st.error(f"Missing environment variables: {', '.join(missing_vars)}")
            st.info("Please check your .env file or environment configuration")
            st.stop()
        
        # Test Qdrant connection
        qdrant_client = QdrantClient(
            url=qdrant_host, 
            api_key=qdrant_api_key,
            timeout=QDRANT_TIMEOUT,
            prefer_grpc=False
        )
        
        # Verify connection with a simple operation
        try:
            collections = qdrant_client.get_collections()
            # st.success(f"✅ Connected to Qdrant successfully!")
        except Exception as e:
            st.error(f"❌ Qdrant connection failed: {str(e)}")
            st.info("Please check your QDRANT_URL and QDRANT_API_KEY")
            
            # Debug info
            with st.expander("🔍 Debug Information"):
                st.code(f"""
Qdrant URL: {qdrant_host}
API Key: {'*' * (len(qdrant_api_key) - 4) + qdrant_api_key[-4:] if len(qdrant_api_key) > 4 else '****'}
Error: {str(e)}
                """)
            st.stop()
        
        # Initialize OpenAI embeddings
        try:
            embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
            # Test embedding creation
            test_embedding = embedding_model.embed_query("test")
            # if len(test_embedding) > 0:
                # st.success("✅ OpenAI embeddings initialized successfully!")
        except Exception as e:
            st.error(f"❌ OpenAI embeddings initialization failed: {str(e)}")
            st.info("Please check your OPENAI_API_KEY")
            st.stop()
        
        return qdrant_client, embedding_model, qdrant_host, qdrant_api_key
        
    except Exception as e:
        st.error(f"Failed to initialize clients: {str(e)}")
        st.stop()


def validate_pdf(uploaded_file):
    """Validate uploaded PDF file"""
    if uploaded_file.size > 50 * 1024 * 1024:  # 50MB limit
        st.error("File size exceeds 50MB limit")
        return False
    return True


def create_vector_store_batch(documents, qdrant_host, qdrant_api_key, collection_name, embedding_model, batch_size=BATCH_SIZE):
    """Create vector store with batch processing to handle timeouts"""
    try:
        total_docs = len(documents)
        st.info(f"Processing {total_docs} documents in batches of {batch_size}")
        
        # Create empty collection first
        from qdrant_client.models import Distance, VectorParams
        
        # Initialize client with timeout
        client = QdrantClient(
            url=qdrant_host,
            api_key=qdrant_api_key,
            timeout=QDRANT_TIMEOUT
        )
        
        # Get embedding dimension
        sample_embedding = embedding_model.embed_query("test")
        vector_size = len(sample_embedding)
        
        # Create collection
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            st.success(f"✅ Created collection '{collection_name}' with vector size {vector_size}")
        except Exception as e:
            if "already exists" in str(e).lower():
                st.warning(f"Collection '{collection_name}' already exists, will add documents to it")
            else:
                raise e
        
        # Process documents in batches
        processed = 0
        progress_bar = st.progress(0)
        
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_docs + batch_size - 1) // batch_size
            
            st.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
            
            try:
                # Create vector store for this batch
                QdrantVectorStore.from_documents(
                    documents=batch,
                    url=qdrant_host,
                    api_key=qdrant_api_key,
                    collection_name=collection_name,
                    embedding=embedding_model
                )
                
                processed += len(batch)
                progress = processed / total_docs
                progress_bar.progress(progress)
                
                st.success(f"✅ Batch {batch_num} processed successfully ({processed}/{total_docs} documents)")
                
            except Exception as batch_error:
                st.error(f"❌ Error processing batch {batch_num}: {str(batch_error)}")
                
                # Try with smaller batch size
                if len(batch) > 1:
                    st.info(f"Retrying batch {batch_num} with individual documents...")
                    for doc in batch:
                        try:
                            QdrantVectorStore.from_documents(
                                documents=[doc],
                                url=qdrant_host,
                                api_key=qdrant_api_key,
                                collection_name=collection_name,
                                embedding=embedding_model
                            )
                            processed += 1
                            progress = processed / total_docs
                            progress_bar.progress(progress)
                        except Exception as doc_error:
                            st.error(f"❌ Failed to process individual document: {str(doc_error)}")
                else:
                    st.error(f"❌ Failed to process single document in batch {batch_num}")
        
        progress_bar.progress(1.0)
        st.success(f"🎉 Successfully processed {processed}/{total_docs} documents!")
        
        return True, processed
        
    except Exception as e:
        st.error(f"❌ Vector store creation failed: {str(e)}")
        return False, 0

def generate_collection_name(base_name, existing_names):
    """Generate unique collection name"""
    collection_name = base_name.lower().replace(" ", "_").replace("-", "_")
    
    if collection_name in existing_names:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        collection_name = f"{collection_name}_{timestamp}"
    
    return collection_name


def process_document(uploaded_file, qdrant_client, embedding_model, qdrant_host, qdrant_api_key):
    """Process and index the uploaded document"""
    try:
        # Generate collection name
        base_name = uploaded_file.name.split(".pdf")[0]
        
        # Test Qdrant connection before proceeding
        try:
            existing_collections = qdrant_client.get_collections()
            existing_names = [col.name for col in existing_collections.collections]
        except Exception as e:
            st.error(f"❌ Failed to connect to Qdrant: {str(e)}")
            st.info("This usually indicates an authentication issue. Please verify:")
            st.markdown("""
            - Your QDRANT_URL is correct (including https://)
            - Your QDRANT_API_KEY is valid and active
            - Your Qdrant cluster is running and accessible
            """)
            return None
        
        collection_name = generate_collection_name(base_name, existing_names)
        
        # Save PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Load and process document
        with st.spinner("Loading PDF..."):
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
        
        # Clean up temp file
        os.remove(tmp_file_path)
        
        if not documents:
            st.error("Failed to extract content from PDF")
            return None
        
        # Split documents with size check
        with st.spinner("Splitting document into chunks..."):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, 
                chunk_overlap=CHUNK_OVERLAP
            )
            split_docs = text_splitter.split_documents(documents)
        
        total_chunks = len(split_docs)
        st.info(f"Document split into {total_chunks} chunks")
        
       
        
        # Create vector store with batch processing
        with st.spinner("Creating embeddings and indexing (this may take a while)..."):
            success, processed_count = create_vector_store_batch(
                documents=split_docs,
                qdrant_host=qdrant_host,
                qdrant_api_key=qdrant_api_key,
                collection_name=collection_name,
                embedding_model=embedding_model,
                batch_size=min(BATCH_SIZE, max(1, total_chunks // 10))  # Dynamic batch size
            )
            
            if not success:
                st.error("❌ Failed to create vector store completely")
                st.info("Common causes of timeouts:")
                st.markdown("""
                - Large documents taking too long to process
                - Network connectivity issues  
                - Qdrant cluster resource limits
                - API rate limiting
                """)
                return None
            
            # Verify the collection was created
            try:
                updated_collections = qdrant_client.get_collections()
                if collection_name in [col.name for col in updated_collections.collections]:
                    st.success(f"✅ Collection '{collection_name}' verified successfully")
                else:
                    st.warning("⚠️ Collection verification failed but processing completed")
            except Exception as verify_error:
                st.warning(f"⚠️ Could not verify collection: {str(verify_error)}")
        
        return collection_name, processed_count
        
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        
        # Provide detailed error information
        with st.expander("🔍 Detailed Error Information"):
            import traceback
            st.code(traceback.format_exc())
        
        return None
    

def get_chat_response(query, collection_name, vector_store):
    """Generate chat response based on query and context"""
    try:
        # Retrieve relevant documents
        results = vector_store.similarity_search(query=query, k=MAX_CONTEXT_CHUNKS)
        
        if not results:
            return "I couldn't find relevant information in the document to answer your question.", []
        
        # Prepare context
        context_parts = []
        for i, res in enumerate(results, 1):
            page_num = res.metadata.get('page', res.metadata.get('page_label', 'Unknown'))
            context_parts.append(f"**Chunk {i} (Page {page_num}):**\n{res.page_content}")
        
        context = "\n\n".join(context_parts)
        
        # Enhanced system prompt
        SYSTEM_PROMPT = f"""You are a helpful AI assistant that answers questions based on the provided PDF document context.

IMPORTANT INSTRUCTIONS:
1. FIRST, check if the answer exists in the provided context below
2. If the answer IS in the context: Answer based on the context and include page references
3. If the answer is NOT in the context: Clearly state that the information is not in the document, then provide a short answer using your general knowledge
4. If the query is regarding programming always try giving an example for better understanding.

Guidelines:
- Be concise but comprehensive
- If the query is about programming, always provide examples
- If you're uncertain about your general knowledge, acknowledge it

Context from the document:
{context}

Remember: You MUST answer the user's question. If it's not in the document, use your general knowledge!
"""
        
        # Generate response
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.5,  # Lower temperature for more focused answers
             max_tokens=1000,
            timeout=30,
            max_retries=2,
        )
        
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=query)
        ]
        
        response = llm.invoke(messages)
        
        return response.content, results
        
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "Sorry, I encountered an error while processing your question.", []
