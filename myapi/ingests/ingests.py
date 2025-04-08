from pathlib import Path
from typing import List, Union, Type, Dict
from llama_index.core.readers.base import BaseReader
from llama_index.core import SimpleDirectoryReader as LlamaDirectoryReader
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.schema import Document as LlamaDocument
import uuid  # For generating unique IDs

class Document:
    def __init__(self, content: str, file_path: Union[str, Path]):
        self.content = content
        self.file_path = file_path
        self.doc_id = str(uuid.uuid4())  # Assign unique ID if not provided

class BaseComponent:
    def log_progress(self, progress_key: str, **kwargs):
        """Logs the progress of the operation."""
        print(f"Logging progress: {progress_key} with details {kwargs}")

class DocumentIngestor(BaseComponent):
    """Ingest PDF documents into Document for indexing."""
    
    # Set PDF mode to 'normal' for standard PDF reading (can adjust as needed)
    pdf_mode: str = "normal"

    # Using TokenSplitter from llama_index for chunking text into manageable sizes
    text_splitter = TokenTextSplitter(
        chunk_size=1024,
        chunk_overlap=20,
        separator=" ",
    )

    # Initialize SentenceWindowNodeParser for sentence-level chunking
    sentence_window_parser = SentenceWindowNodeParser(
        chunk_size=1024,  # Define max chunk size for sentences
        chunk_overlap=20,  # Overlap between sentence windows
    )

    def _get_reader(self, input_files: List[Union[str, Path]]) -> LlamaDirectoryReader:
        """Get appropriate reader for PDF files."""
        # Initialize the PDF reader
        file_extractors = {
            ".pdf": PDFReader(),
        }

        # Initialize the main reader to handle PDF files
        main_reader = LlamaDirectoryReader(
            input_files=[str(file) for file in input_files],
            file_extractor=file_extractors,
        )

        return main_reader

    def run(self, file_paths: Union[List[Union[str, Path]], Union[str, Path]]) -> List[Document]:
        """Ingest the PDF file paths into Document."""

        # Ensure file_paths is a list even if it's a single file path
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]

        # Convert file paths to Path objects for consistency
        file_paths = [Path(file) for file in file_paths]

        # Validate file paths
        for file in file_paths:
            if not file.exists():
                raise FileNotFoundError(f"File {file} not found.")
            if not file.is_file():
                raise ValueError(f"{file} is not a valid file.")

        # Read documents using the appropriate PDF reader
        try:
            main_reader = self._get_reader(input_files=file_paths)
            documents = main_reader.load_data()
            print(f"Read {len(file_paths)} PDF files into {len(documents)} documents.")
        except Exception as e:
            raise RuntimeError(f"Error reading files: {e}")

        # Split documents into nodes using both TokenSplitter and SentenceWindowNodeParser
        nodes = []
        for document in documents:
    # Skip strings that don't have the required attributes
            if isinstance(document, str): 
                print("Skipping string document...")
                continue  # Skip strings that don't have the 'id_' attribute

            # Handle expected types
            if isinstance(document, dict):
                # Extract 'text' and 'metadata' (file_name) from the dictionary
                content = document.get("text", "")
                file_name = document.get("metadata", {}).get("file_name", "")
                file_path = Path(file_name) if file_name else None  # Use file_name from metadata to construct file path
            elif isinstance(document, LlamaDocument):
                content = document.text
                file_path = None  # No file path assigned for LlamaDocument in this case
            elif isinstance(document, Document):
                content = document.content
                file_path = document.file_path
            else:
                print(f"Warning: Unexpected document type: {type(document)}. Skipping this document.")
                continue  # Skip this document

            # Proceed with content extraction
            token_chunks = self.text_splitter.split_text(content)
            sentence_chunks = self.sentence_window_parser.get_nodes_from_documents(content)

            # Combine both chunking methods into nodes
            for chunk in token_chunks + sentence_chunks:
                nodes.append({
                    "content": chunk,
                    "file_path": file_path
                })

        print(f"Processed {len(documents)} documents and generated {len(nodes)} nodes.")
        return nodes

