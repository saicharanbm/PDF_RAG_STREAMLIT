import streamlit as st
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from helper import initialize_clients, validate_pdf,process_document,get_chat_response

load_dotenv()


@st.cache_resource
def load_clients():
    return initialize_clients()

qdrant_client, embedding_model, qdrant_host, qdrant_api_key = load_clients()


# --- Main Interface ---
st.title("🤖 Intelligent PDF Q&A System")

# --- Custom CSS for footer positioning ---
st.markdown("""
<style>
/* Add padding to the main container to make room for footer */
.main > div {
    padding-bottom: 120px !important;
}

/* Style the chat input container */
.stChatInput {
    margin-bottom: 30px !important;
    
}

.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: rgba(38, 39, 48, 0.95);
    color: #fafafa;
    text-align: center;
    padding: 15px 0;
    z-index: 999;
    border-top: 1px solid #333;
    backdrop-filter: blur(10px);
}
.footer a {
    color: #58a6ff;
    text-decoration: none;
}
.footer a:hover {
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)

# Footer with fixed positioning
st.markdown(
    '<div class="footer">Built with ❤️ by <a href="https://github.com/saicharanbm/pdf_rag_streamlit" target="_blank">Sai Charan B M</a></div>',
    unsafe_allow_html=True
)

# Mode selection
mode = st.selectbox("Choose an option", ["📤 Upload Document", "💬 Chat with Document"])

# --- UPLOAD INTERFACE ---
if mode == "📤 Upload Document":
    st.header("Upload and Index a PDF")
    st.markdown("Upload a PDF document to create a searchable knowledge base.")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file", 
        type="pdf",
        help="Maximum file size: 50MB"
    )

    if uploaded_file is not None:
        if validate_pdf(uploaded_file):
            # Show file info
            st.info(f"**File:** {uploaded_file.name} ({uploaded_file.size / 1024 / 1024:.2f} MB)")
            
            if st.button("🚀 Process Document", type="primary"):
                result = process_document(uploaded_file, qdrant_client, embedding_model, qdrant_host, qdrant_api_key)
                
                if result:
                    collection_name, num_chunks = result
                    st.success(f"✅ Document successfully indexed as `{collection_name}` with {num_chunks} chunks")
                    st.balloons()

# --- CHAT INTERFACE ---
elif mode == "💬 Chat with Document":
    st.header("Chat with Your Documents")
    
    # Fetch available collections
    try:
        collections_resp = qdrant_client.get_collections()
        collections = [col.name for col in collections_resp.collections]
    except Exception as e:
        st.error(f"Error fetching collections: {str(e)}")
        collections = []

    if not collections:
        st.warning("📋 No documents found. Please upload a document first.")
        st.markdown("👆 Switch to the **Upload Document** tab to get started.")
    else:
        # Collection selection
        selected_collection = st.selectbox(
            "📖 Select a document to chat with:", 
            collections,
            help="Choose from your uploaded documents"
        )
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about the document..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Initialize vector store
                        vector_store = QdrantVectorStore.from_existing_collection(
                            url=qdrant_host,
                            api_key=qdrant_api_key,
                            collection_name=selected_collection,
                            embedding=embedding_model
                        )
                        
                        # Get response
                        response, sources = get_chat_response(prompt, selected_collection, vector_store)
                        st.markdown(response)
                        
                        # Show sources in expander
                        if sources:
                            with st.expander("📄 View Sources"):
                                for i, source in enumerate(sources, 1):
                                    page_num = source.metadata.get('page', source.metadata.get('page_label'))
                                    st.markdown(f"**Source {i} (Page {page_num}):**")
                                    st.markdown(f"```\n{source.page_content}\n```")
                        
                        # Add assistant message to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        # Clear chat button
        if st.session_state.messages:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                st.rerun()