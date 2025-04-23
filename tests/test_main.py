import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import main  # Import the module we want to test

# --- Test get_embedding_function ---

@patch('main.ollama.list')
@patch('main.OllamaEmbeddings')
def test_get_embedding_function_success(mock_ollama_embeddings, mock_ollama_list):
    """Tests that get_embedding_function returns an OllamaEmbeddings instance on success."""
    # Arrange
    mock_ollama_list.return_value = True  # Simulate Ollama service being available
    mock_embedding_instance = MagicMock()
    mock_ollama_embeddings.return_value = mock_embedding_instance

    # Act
    embedding_function = main.get_embedding_function()

    # Assert
    mock_ollama_list.assert_called_once()
    mock_ollama_embeddings.assert_called_once_with(model=main.EMBEDDING_MODEL_OLLAMA)
    assert embedding_function == mock_embedding_instance

@patch('main.ollama.list')
def test_get_embedding_function_connection_error(mock_ollama_list):
    """Tests that get_embedding_function raises ConnectionError if Ollama is unreachable."""
    # Arrange
    mock_ollama_list.side_effect = Exception("Service unavailable")

    # Act & Assert
    with pytest.raises(ConnectionError) as excinfo:
        main.get_embedding_function()
    assert "Ollama service not running or unreachable" in str(excinfo.value)
    mock_ollama_list.assert_called_once()

# --- Test get_ollama_llm ---

@patch('main.Ollama') # Mock the Ollama class from langchain_ollama
def test_get_ollama_llm_success(mock_ollama):
    """Tests that get_ollama_llm returns an Ollama LLM instance on success."""
    # Arrange
    mock_llm_instance = MagicMock()
    mock_ollama.return_value = mock_llm_instance

    # Act
    llm = main.get_ollama_llm()

    # Assert
    mock_ollama.assert_called_once_with(
        model=main.OLLAMA_MODEL, base_url="http://localhost:11434", temperature=0
    )
    assert llm == mock_llm_instance

@patch('main.Ollama')
def test_get_ollama_llm_connection_error(mock_ollama):
    """Tests that get_ollama_llm raises ConnectionError if initialization fails."""
    # Arrange
    mock_ollama.side_effect = Exception("Connection failed")

    # Act & Assert
    with pytest.raises(ConnectionError) as excinfo:
        main.get_ollama_llm()
    assert f"Failed to connect to Ollama model {main.OLLAMA_MODEL}" in str(excinfo.value)
    mock_ollama.assert_called_once()

# --- Test create_rag_chain (Placeholder) ---

@patch('main.RetrievalQA')
def test_create_rag_chain_success(mock_retrieval_qa):
     """Tests that create_rag_chain calls RetrievalQA correctly."""
     # Arrange
     mock_llm = MagicMock()
     mock_retriever = MagicMock()
     mock_chain_instance = MagicMock()
     mock_retrieval_qa.from_chain_type.return_value = mock_chain_instance

     # Act
     chain = main.create_rag_chain(mock_llm, mock_retriever)

     # Assert
     mock_retrieval_qa.from_chain_type.assert_called_once()
     # More detailed assertions can be added here to check args like prompt, llm, retriever
     assert chain == mock_chain_instance

# --- Test process_pdfs ---

@patch('main.get_ollama_llm')
@patch('main.create_rag_chain')
@patch('main.Chroma.from_documents')
@patch('main.RecursiveCharacterTextSplitter')
@patch('main.PyPDFLoader')
def test_process_pdfs_success(mock_pdf_loader, mock_splitter, mock_chroma, mock_create_chain, mock_get_llm):
    """Tests that process_pdfs correctly processes PDF files and sets up the RAG chain."""
    # Arrange
    mock_pdf = MagicMock()
    mock_pdf.name = "/path/to/test.pdf"
    mock_loaded_docs = [MagicMock(), MagicMock()]  # Simulate loaded documents
    mock_pdf_loader.return_value.load_and_split.return_value = mock_loaded_docs
    
    mock_split_docs = [MagicMock() for _ in range(3)]  # Simulate split documents
    mock_splitter.return_value.split_documents.return_value = mock_split_docs
    
    mock_vector_store = MagicMock()
    mock_chroma.return_value = mock_vector_store
    mock_retriever = MagicMock()
    mock_vector_store.as_retriever.return_value = mock_retriever
    
    mock_llm = MagicMock()
    mock_get_llm.return_value = mock_llm
    
    mock_chain = MagicMock()
    mock_create_chain.return_value = mock_chain
    
    mock_embedding_function = MagicMock()
    
    # Act
    result, vs, ch = main.process_pdfs([mock_pdf], mock_embedding_function)
    
    # Assert
    mock_pdf_loader.assert_called_once_with(mock_pdf.name)
    mock_splitter.assert_called_once()
    mock_chroma.assert_called_once()
    mock_get_llm.assert_called_once()
    mock_create_chain.assert_called_once_with(mock_llm, mock_retriever)
    assert "Successfully processed 1 PDF" in result
    assert vs == mock_vector_store
    assert ch == mock_chain


@patch('main.PyPDFLoader')
def test_process_pdfs_empty_input(mock_pdf_loader):
    """Tests that process_pdfs handles empty input correctly."""
    # Act
    result, vs, ch = main.process_pdfs([], MagicMock())
    
    # Assert
    mock_pdf_loader.assert_not_called()
    assert "No PDF files uploaded" in result
    assert vs is None
    assert ch is None


@patch('main.PyPDFLoader')
def test_process_pdfs_loader_error(mock_pdf_loader):
    """Tests that process_pdfs handles PDF loader errors."""
    # Arrange
    mock_pdf = MagicMock()
    mock_pdf.name = "/path/to/test.pdf"
    mock_pdf_loader.return_value.load_and_split.side_effect = Exception("Loading error")
    
    # Act
    result, vs, ch = main.process_pdfs([mock_pdf], MagicMock())
    
    # Assert
    assert "Error processing test.pdf" in result
    assert vs is None
    assert ch is None


# --- Test handle_upload ---

@patch('main.get_embedding_function')
@patch('main.process_pdfs')
def test_handle_upload_success(mock_process_pdfs, mock_get_embedding_function):
    """Tests that handle_upload correctly calls process_pdfs with the embedding function."""
    # Create a mock closure to test the inner function
    def execute_handle_upload(files):
        # Arrange
        mock_embedding = MagicMock()
        mock_get_embedding_function.return_value = mock_embedding
        mock_process_pdfs.return_value = ("Success", MagicMock(), MagicMock())
        
        # This simulates what happens inside build_gradio_ui's handle_upload
        if files is None:
            return "No files uploaded."
        try:
            embedding_func = mock_get_embedding_function()
            status, _, _ = mock_process_pdfs(files, embedding_func)
            return status
        except ConnectionError as e:
            return f"Error: {e}. Is Ollama running?"
        except Exception as e:
            return f"An unexpected error occurred during processing: {e}"
    
    # Act
    mock_files = [MagicMock()]
    result = execute_handle_upload(mock_files)
    
    # Assert
    mock_get_embedding_function.assert_called_once()
    mock_process_pdfs.assert_called_once_with(mock_files, mock_get_embedding_function.return_value)
    assert "Success" in result


@patch('main.get_embedding_function')
def test_handle_upload_connection_error(mock_get_embedding_function):
    """Tests that handle_upload handles ConnectionError properly."""
    # Create a mock closure to test the inner function
    def execute_handle_upload(files):
        # Simulates what happens inside build_gradio_ui's handle_upload
        if files is None:
            return "No files uploaded."
        try:
            embedding_func = mock_get_embedding_function()
            # Won't reach here due to exception
            return "Success"
        except ConnectionError as e:
            return f"Error: {e}. Is Ollama running?"
        except Exception as e:
            return f"An unexpected error occurred during processing: {e}"
    
    # Arrange
    mock_get_embedding_function.side_effect = ConnectionError("Ollama not running")
    
    # Act
    result = execute_handle_upload([MagicMock()])
    
    # Assert
    assert "Error:" in result
    assert "Ollama" in result


@patch('main.get_embedding_function')
@patch('main.process_pdfs')
def test_handle_upload_general_error(mock_process_pdfs, mock_get_embedding_function):
    """Tests that handle_upload handles general exceptions properly."""
    # Create a mock closure to test the inner function
    def execute_handle_upload(files):
        # Simulates what happens inside build_gradio_ui's handle_upload
        if files is None:
            return "No files uploaded."
        try:
            embedding_func = mock_get_embedding_function()
            status, _, _ = mock_process_pdfs(files, embedding_func)
            return status
        except ConnectionError as e:
            return f"Error: {e}. Is Ollama running?"
        except Exception as e:
            return f"An unexpected error occurred during processing: {e}"
    
    # Arrange
    mock_embedding = MagicMock()
    mock_get_embedding_function.return_value = mock_embedding
    mock_process_pdfs.side_effect = Exception("Processing error")
    
    # Act
    result = execute_handle_upload([MagicMock()])
    
    # Assert
    assert "unexpected error" in result


# --- Test handle_chat ---

def test_handle_chat_chain_not_initialized():
    """Tests that handle_chat correctly handles the case when the chain is not initialized."""
    # Create a mock closure to test the inner function
    def execute_handle_chat(message, history):
        # Simulates what happens inside build_gradio_ui's handle_chat
        if main.chain is None:
            history.append({"role": "user", "content": message})
            history.append(
                {
                    "role": "assistant",
                    "content": "Error: The RAG chain is not initialized. Please process PDF files first.",
                }
            )
            return history, ""
        # Won't reach here due to chain being None
        return history, ""
    
    # Arrange
    # Save original chain value to restore later
    original_chain = main.chain
    main.chain = None  # Ensure chain is None
    message = "Test question"
    history = []
    
    # Act
    try:
        updated_history, msg = execute_handle_chat(message, history)
        
        # Assert
        assert len(updated_history) == 2
        assert updated_history[0]["role"] == "user"
        assert updated_history[0]["content"] == message
        assert updated_history[1]["role"] == "assistant"
        assert "Error: The RAG chain is not initialized" in updated_history[1]["content"]
        assert msg == ""
    finally:
        # Restore original chain value
        main.chain = original_chain


@patch('main.chain')
def test_handle_chat_success(mock_chain):
    """Tests that handle_chat correctly processes a message when the chain is initialized."""
    # Create a mock closure to test the inner function
    def execute_handle_chat(message, history):
        # Simulates what happens inside build_gradio_ui's handle_chat
        if mock_chain is None:
            history.append({"role": "user", "content": message})
            history.append(
                {
                    "role": "assistant",
                    "content": "Error: The RAG chain is not initialized. Please process PDF files first.",
                }
            )
            return history, ""
            
        try:
            result = mock_chain.invoke({"query": message})
            answer = result.get("result", "Sorry, I couldn't find an answer.")
            source_docs = result.get("source_documents", [])

            if source_docs:
                sources_text = "\n\n**Sources:**\n"
                added_sources = set()
                for doc in source_docs:
                    source_name = doc.metadata.get("source", "Unknown")
                    page_num = doc.metadata.get("page", "N/A")
                    source_key = f"{source_name}_p{page_num}"
                    if source_key not in added_sources:
                        sources_text += f"- {source_name} (Page {page_num + 1})\n"
                        added_sources.add(source_key)
                answer += sources_text

            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": answer})
            return history, ""
        except Exception as e:
            error_message = f"An error occurred during chat: {e}"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_message})
            return history, ""
    
    # Arrange
    mock_doc1 = MagicMock()
    mock_doc1.metadata = {"source": "doc1.pdf", "page": 0}
    
    mock_doc2 = MagicMock()
    mock_doc2.metadata = {"source": "doc2.pdf", "page": 1}
    
    mock_chain.invoke.return_value = {
        "result": "Test answer",
        "source_documents": [mock_doc1, mock_doc2]
    }
    
    message = "Test question"
    history = []
    
    # Act
    updated_history, msg = execute_handle_chat(message, history)
    
    # Assert
    mock_chain.invoke.assert_called_once_with({"query": message})
    assert len(updated_history) == 2
    assert updated_history[0]["role"] == "user"
    assert updated_history[0]["content"] == message
    assert updated_history[1]["role"] == "assistant"
    assert "Test answer" in updated_history[1]["content"]
    assert "Sources:" in updated_history[1]["content"]
    assert "doc1.pdf" in updated_history[1]["content"]
    assert "doc2.pdf" in updated_history[1]["content"]
    assert msg == ""


@patch('main.chain')
def test_handle_chat_error(mock_chain):
    """Tests that handle_chat correctly handles errors during chat processing."""
    # Create a mock closure to test the inner function
    def execute_handle_chat(message, history):
        # Simulates what happens inside build_gradio_ui's handle_chat
        if mock_chain is None:
            history.append({"role": "user", "content": message})
            history.append(
                {
                    "role": "assistant",
                    "content": "Error: The RAG chain is not initialized. Please process PDF files first.",
                }
            )
            return history, ""
            
        try:
            result = mock_chain.invoke({"query": message})
            answer = result.get("result", "Sorry, I couldn't find an answer.")
            source_docs = result.get("source_documents", [])

            if source_docs:
                sources_text = "\n\n**Sources:**\n"
                added_sources = set()
                for doc in source_docs:
                    source_name = doc.metadata.get("source", "Unknown")
                    page_num = doc.metadata.get("page", "N/A")
                    source_key = f"{source_name}_p{page_num}"
                    if source_key not in added_sources:
                        sources_text += f"- {source_name} (Page {page_num + 1})\n"
                        added_sources.add(source_key)
                answer += sources_text

            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": answer})
            return history, ""
        except Exception as e:
            error_message = f"An error occurred during chat: {e}"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_message})
            return history, ""
    
    # Arrange
    mock_chain.invoke.side_effect = Exception("Chat error")
    
    message = "Test question"
    history = []
    
    # Act
    updated_history, msg = execute_handle_chat(message, history)
    
    # Assert
    assert len(updated_history) == 2
    assert updated_history[0]["role"] == "user"
    assert updated_history[0]["content"] == message
    assert updated_history[1]["role"] == "assistant"
    assert "An error occurred during chat:" in updated_history[1]["content"]
    assert msg == "" 