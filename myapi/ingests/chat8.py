import os
import json
import boto3
import requests
from pathlib import Path
from typing import List, Dict, Any, Tuple
from urllib.parse import urlparse
import numpy as np
import traceback
import logging
from scipy.spatial.distance import cosine
from .docx_loader import docx_reader
from .txt_loader import load_data
from .html_loader import parse_html
from .excel_loader import load_excel_data
from .xml_loader import xml_data
from .ocr_loader import extract_images_and_text_from_pdf
import dotenv
import ollama
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import SentenceTransformersTokenTextSplitter, RecursiveCharacterTextSplitter
import hashlib
import uuid

# Django specific imports
from django.db import models
from django.utils import timezone
from django.db.models import Q
import django

# Initialize Django settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "your_project.settings")
django.setup()

# Import the Chunk model
from myapi.models import Chunk


class DocumentQueryAssistant:
    def __init__(self, 
                 chunk_size: int = 500, 
                 chunk_overlap: int = 50, 
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 ollama_model: str = "mistral:latest"):
        
        # Configure logging
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Initialize text splitters
        self.token_splitter = SentenceTransformersTokenTextSplitter(
            model_name=embedding_model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.backup_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Ollama model for response generation
        self.ollama_model = ollama_model
        
        # AWS S3 client
        try:
            self.s3_client = boto3.client('s3')
        except Exception as e:
            self.logger.error(f"AWS S3 client initialization failed: {e}")
            self.s3_client = None

    def load_existing_embeddings(self, document_id):
        """Load existing embeddings from database for a specific document"""
        try:
            # Query the database for chunks with the document_id in metadata
            chunks = Chunk.objects.filter(metadata__document_id=document_id)
            
            if chunks.exists():
                # Convert Django QuerySet to list of dictionaries
                chunks_data = []
                for chunk in chunks:
                    chunk_dict = {
                        "chunk_id": chunk.chunk_id,
                        "chunk_text": chunk.chunk_text,
                        "embedding": chunk.embedding,
                        "metadata": chunk.metadata,
                        "document_id": document_id
                    }
                    chunks_data.append(chunk_dict)
                return chunks_data
            return None
        except Exception as e:
            self.logger.error(f"Error loading embeddings for document {document_id}: {e}")
            return None
    
    def save_embeddings(self, document_id, embeddings_data):
        """Save embeddings for a specific document to database"""
        try:
            # Create or update chunks in the database
            for chunk_data in embeddings_data:
                # Set document_id in metadata if not already there
                if not chunk_data.get("metadata"):
                    chunk_data["metadata"] = {}
                
                chunk_data["metadata"]["document_id"] = document_id
                
                # Create new chunk in database
                Chunk.objects.create(
                    file_name=chunk_data["metadata"].get("file_name", ""),
                    file_path=chunk_data["metadata"].get("file_path", ""),
                    chunk_text=chunk_data["chunk_text"],
                    page_number=chunk_data["metadata"].get("page_number"),
                    embedding=chunk_data["embedding"],
                    metadata=chunk_data["metadata"]
                )
                
            self.logger.info(f"Saved {len(embeddings_data)} chunks for document {document_id}")
        except Exception as e:
            self.logger.error(f"Error saving embeddings for document {document_id}: {e}")
            self.logger.error(traceback.format_exc())

    def download_file(self, url: str, local_path: Path):
        """Download file from S3 or HTTP/HTTPS URL"""
        parsed_url = urlparse(url)
        
        if parsed_url.scheme == 's3':
            return self.download_file_from_s3(url, local_path)
        elif parsed_url.scheme in ['http', 'https']:
            return self.download_file_from_http(url, local_path)
        else:
            self.logger.error(f"Unsupported URL scheme: {parsed_url.scheme}")
            return None

   
    def download_file_from_s3(self, s3_url: str, folder_name: str = "docs"):
        """Download file from S3"""
        try:
            parsed_url = urlparse(s3_url)
            filename = os.path.basename(parsed_url.path)

            folder_path = Path(folder_name)
            folder_path.mkdir(parents=True, exist_ok=True)

            file_path = folder_path / filename

            response = requests.get(s3_url, stream=True)
            response.raise_for_status()

            with open(file_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            print(f"File downloaded successfully: {file_path}")
            return file_path
        except Exception as e:
            print(f"Error downloading file: {e}")
            return None
            
    def download_file_from_http(self, url: str, local_path: Path):
        """Download file from HTTP/HTTPS URL"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, 'wb') as f:
                f.write(response.content)
            return local_path
        except Exception as e:
            self.logger.error(f"Error downloading file from URL: {e}")
            return None

    def generate_document_id(self, file_name: str, document_url: str) -> str:
        """Generate a unique document ID"""
        unique_string = f"{file_name}|{document_url}"
        return hashlib.md5(unique_string.encode()).hexdigest()

    def process_document_from_url(self, url: str, download_dir: Path) -> List[Dict]:
        """Download and process a document based on URL"""
        parsed_url = urlparse(url)
        file_name = os.path.basename(parsed_url.path)
        file_path = download_dir / file_name
        
        downloaded_file = self.download_file(url, file_path)
        if not downloaded_file:
            self.logger.error("Failed to download file")
            return []
        
        return self.process_document(downloaded_file)

    def process_document(self, file_path: Path) -> List[Dict]:
        """Process document based on file type"""
        file_extension = file_path.suffix.lower()
        file_metadata = {"file_name": file_path.name, "file_path": str(file_path)}
        all_chunks = []

        try:
            if file_extension == ".pdf":
                ocr_results = extract_images_and_text_from_pdf(file_path)
                for page in ocr_results:
                    if page.get("text"):
                        page_metadata = {
                            **file_metadata,
                            "page_number": page.get("page_number"),
                            "source_type": "pdf"
                        }
                        chunks = self.create_chunks(page["text"], page_metadata)
                        for chunk in chunks:
                            chunk["metadata"].update({
                                "images": page.get("images", []),
                                "ocr_tables": page.get("ocr_tables", [])
                            })
                            all_chunks.append(chunk)

            elif file_extension == ".txt":
                text_data = load_data(file_path)
                for item in text_data:
                    if item.get("text"):
                        txt_metadata = {
                            **file_metadata,
                            **item.get("metadata", {}),
                            "source_type": "txt"
                        }
                        chunks = self.create_chunks(item["text"], txt_metadata)
                        all_chunks.extend(chunks)

            # Add similar processing for other file types like HTML, DOCX, XML, etc.
            elif file_extension == ".html":
                html_data = parse_html(file_path)
                for item in html_data:
                    if item.get("text"):
                        html_metadata = {
                            **file_metadata,
                            **item.get("metadata", {}),
                            "source_type": "html"
                        }
                        chunks = self.create_chunks(item["text"], html_metadata)
                        all_chunks.extend(chunks)

            elif file_extension in [".xls", ".xlsx"]:
                excel_data = load_excel_data(file_path)
                for item in excel_data:
                    if item.get("text"):
                        excel_metadata = {
                            **file_metadata,
                            **item.get("metadata", {}),
                            "source_type": "excel"
                        }
                        chunks = self.create_chunks(item["text"], excel_metadata)
                        all_chunks.extend(chunks)

            elif file_extension == ".docx":
                docx_data = docx_reader(file_path, extra_info=file_metadata)
                for item in docx_data:
                    if item.get("text"):
                        docx_metadata = {
                            **file_metadata, 
                            **item.get("metadata", {}), 
                            "source_type": "docx"
                        }
                        chunks = self.create_chunks(item["text"], docx_metadata)
                        all_chunks.extend(chunks)

            elif file_extension == ".xml":
                xml_datas = xml_data(file_path, extra_info=file_metadata)
                for item in xml_datas:
                    if item.get("text"):
                        xml_metadata = {
                            **file_metadata, 
                            **item.get("metadata", {}), 
                            "source_type": "xml"
                        }
                        chunks = self.create_chunks(item["text"], xml_metadata)
                        all_chunks.extend(chunks)

        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            print(traceback.format_exc())

        return all_chunks

    def create_chunks(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create text chunks with metadata"""
        try:
            chunks = self.token_splitter.create_documents(
                texts=[text],
                metadatas=[metadata]
            )
        except Exception as e:
            print(f"Falling back to recursive splitter due to: {e}")
            chunks = self.backup_splitter.create_documents(
                texts=[text],
                metadatas=[metadata]
            )

        return [{
            "chunk_text": chunk.page_content,
            "metadata": chunk.metadata
        } for chunk in chunks]

    def generate_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """Generate embeddings for text chunks"""
        for chunk in chunks:
            chunk['embedding'] = self.embedding_model.encode(
                chunk['chunk_text'], 
                convert_to_numpy=True
            ).tolist()
        return chunks

    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        return 1 - cosine(vec1, vec2)

    def select_relevant_chunks(self, query: str, embeddings_data: List[Dict], top_k: int = 5):
        """Select top relevant chunks based on query"""
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        similarities = []

        for chunk in embeddings_data:
            chunk_embedding = np.array(chunk.get("embedding", []))
            if chunk_embedding.size == 0:
                continue

            sim = self.cosine_similarity(query_embedding, chunk_embedding)
            similarities.append((chunk, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [(chunk, score) for chunk, score in similarities[:top_k]]
    

    def rerank_results(self, query_embedding, retrieved_chunks):
        """Ranks retrieved chunks based on similarity and length."""
        # Unpack chunks and scores if they're provided as tuples
        if isinstance(retrieved_chunks[0], tuple):
            chunks_only = [chunk for chunk, _ in retrieved_chunks]
            initial_scores = [score for _, score in retrieved_chunks]
        else:
            chunks_only = retrieved_chunks
            initial_scores = []
        
        retrieved_embeddings = [np.array(chunk["embedding"]) for chunk in chunks_only if "embedding" in chunk]
        similarities = [self.cosine_similarity(query_embedding, emb) for emb in retrieved_embeddings]
        chunk_lengths = [len(chunk["chunk_text"]) for chunk in chunks_only]

        # Normalize scores safely
        max_sim, min_sim = max(similarities, default=1), min(similarities, default=0)
        similarities_normalized = [(sim - min_sim) / (max_sim - min_sim + 1e-9) for sim in similarities]

        max_length, min_length = max(chunk_lengths, default=1), min(chunk_lengths, default=0)
        chunk_lengths_normalized = [(l - min_length) / (max_length - min_length + 1e-9) for l in chunk_lengths]

        combined_scores = [(sim + l) / 2 for sim, l in zip(similarities_normalized, chunk_lengths_normalized)]

        # Sort by score
        sorted_indices = np.argsort(combined_scores)[::-1]
        sorted_chunks = [chunks_only[i] for i in sorted_indices]
        sorted_scores = [combined_scores[i] for i in sorted_indices]
        sorted_similarities = [similarities[i] for i in sorted_indices]

        # Return best chunk, all sorted chunks, and all score information
        return sorted_chunks[0], sorted_chunks, {
            'combined_scores': sorted_scores,
            'similarity_scores': sorted_similarities,
            'initial_scores': initial_scores if initial_scores else None
        }

    def generate_response(self, query: str, best_chunk: Dict):
        """Generate response using Ollama"""
        best_text = best_chunk.get("chunk_text", "") if isinstance(best_chunk, dict) else str(best_chunk)
        input_text = f"Query: {query}\nRelevant Context: {best_text}\nAnswer:"

        try:
            response = ollama.chat(
                model=self.ollama_model, 
                messages=[{"role": "user", "content": input_text}]
            )
            
            return response['message']['content']
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return "Unable to generate a response at this time."

    def process_and_query(self, document_url: str, query: str, document_id: str = None, base_dir: Path = Path('./document_processing')):
        base_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Generate document_id if not provided
            if not document_id and document_url:
                file_name = os.path.basename(urlparse(document_url).path)
                document_id = self.generate_document_id(file_name, document_url)
            
            if not document_id:
                self.logger.error("No document_id provided and couldn't generate one from URL.")
                return None
            
            # Try to load existing embeddings from database
            embeddings_with_metadata = self.load_existing_embeddings(document_id)
            
            # If embeddings don't exist in database, process the document and store in database
            if embeddings_with_metadata:
                self.logger.info(f"Using existing embeddings for document_id: {document_id}")
            elif document_url:
                file_name = os.path.basename(urlparse(document_url).path)
                local_file_path = base_dir / file_name
                local_file = self.download_file(document_url, local_file_path)
                
                if not local_file:
                    self.logger.error("Document download failed.")
                    return None
                
                print(f"File downloaded successfully: {local_file}")
                
                chunks = self.process_document(local_file)
                if not chunks:
                    self.logger.warning("No chunks created from the document.")
                    return None
                
                embeddings_with_metadata = self.generate_embeddings(chunks)
                
                # Save embeddings to database
                self.save_embeddings(document_id, embeddings_with_metadata)
            else:
                self.logger.error("No existing embeddings found and no document_url provided to download.")
                return None
        
            # Select relevant chunks for the query
            chunks_with_scores = self.select_relevant_chunks(query, embeddings_with_metadata)
            relevant_chunks = [chunk for chunk, _ in chunks_with_scores]
            similarity_scores = [score for _, score in chunks_with_scores]
            
            # Validate chunk content
            for chunk in relevant_chunks:
                if "chunk_text" not in chunk or not chunk["chunk_text"]:
                    self.logger.warning(f"Missing chunk_text in chunk: {chunk.get('chunk_id', 'unknown')}")
            
            # Rerank results
            query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
            best_chunk, reranked_chunks, score_details = self.rerank_results(query_embedding, chunks_with_scores)
            
            # Generate response
            response = self.generate_response(query, best_chunk) if best_chunk else "No relevant information found."
        
            # Prepare output data
            output_data = {
                'response': response, 
                'relevant_chunks': relevant_chunks,
                'similarity_scores': similarity_scores,
                'reranked_chunks': reranked_chunks,
                'score_details': score_details,
                'document_id': document_id
            }
            
            # Save output to file (optional, could be removed in production)
            output_file = base_dir / "output.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)

            print(f"Output saved to {output_file}")
            return output_data
        
        except Exception as e:
            self.logger.error(f"Error in document processing and querying: {e}")
            self.logger.error(traceback.format_exc())
            return None

if __name__ == "__main__":
        assistant = DocumentQueryAssistant()
        result = assistant.process_and_query(
            document_url="",
            query="Tell me about about Interest Payments ?",
            document_id="e57d7a745e68382c99e4a742b82bbf1b"
        )
        
        if result:
            print("\nResponse:", result['response'])
            print("\nRelevant Chunks:")
            for i, (chunk, score) in enumerate(zip(result['relevant_chunks'], result['similarity_scores'])):
                print(f"Chunk {i+1} (Similarity Score: {score:.4f}) ---")
                print("\nChunk Text:", chunk.get('chunk_text', 'No text available'))
                print("\nMetadata:")
                metadata = chunk.get('metadata', {})
                for key, value in metadata.items():
                    print(f"  {key}: {value}")