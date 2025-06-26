import os
import json
import boto3
import requests
from pymongo import MongoClient
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
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
from PIL import Image
import pytesseract
import tempfile
import shutil
import contextlib
import threading
import functools
import time

# Django specific imports
from django.db import models
from django.utils import timezone
from django.db.models import Q
import django
import openai

# Initialize Django settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "your_project.settings")
django.setup()

# Import the Chunk model
from myapi.models import Chunk


class ModelCache:
    """Singleton class to cache models and avoid reloading"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._models = {}
            self._initialized = True
    
    def get_embedding_model(self, model_name: str) -> SentenceTransformer:
        """Get cached embedding model or load if not exists"""
        if model_name not in self._models:
            print(f"Loading embedding model: {model_name}")
            self._models[model_name] = SentenceTransformer(model_name)
        return self._models[model_name]


def timing_decorator(func):
    """Decorator to measure function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper


class DocumentQueryAssistant:
    def __init__(self, 
                 chunk_size: int = 500, 
                 chunk_overlap: int = 50, 
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 ollama_model: str = "mistral:7b-instruct-q2_k",
                 max_cache_size: int = 1000):
        
        # Configure logging
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Use cached model instance
        self.model_cache = ModelCache()
        self.embedding_model_name = embedding_model_name
        
        # Lazy load embedding model (only when needed)
        self._embedding_model = None
        
        # Initialize text splitters (lightweight)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._token_splitter = None
        self._backup_splitter = None
        
        # Ollama model for response generation
        self.ollama_model = ollama_model
        
        # MongoDB connection with connection pooling
        self.mongo_client = MongoClient(
            "mongodb://140.10.2.95:27017/",
            maxPoolSize=10,
            minPoolSize=1,
            maxIdleTimeMS=30000,
            waitQueueTimeoutMS=5000
        )
        self.mongo_db = self.mongo_client["mio_db"]
        self.mongo_collection = self.mongo_db["embeddings"]
        
        # Create index for faster queries
        try:
            self.mongo_collection.create_index("file_name")
        except Exception as e:
            self.logger.warning(f"Index creation failed (might already exist): {e}")
        
        # In-memory cache for embeddings
        self.embeddings_cache = {}
        self.max_cache_size = max_cache_size
        
        # AWS S3 client
        try:
            self.s3_client = boto3.client('s3')
        except Exception as e:
            self.logger.error(f"AWS S3 client initialization failed: {e}")
            self.s3_client = None

    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy load embedding model"""
        if self._embedding_model is None:
            self._embedding_model = self.model_cache.get_embedding_model(self.embedding_model_name)
        return self._embedding_model
    
    @property
    def token_splitter(self):
        """Lazy load token splitter"""
        if self._token_splitter is None:
            self._token_splitter = SentenceTransformersTokenTextSplitter(
                model_name=self.embedding_model_name,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        return self._token_splitter
    
    @property
    def backup_splitter(self):
        """Lazy load backup splitter"""
        if self._backup_splitter is None:
            self._backup_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        return self._backup_splitter

    @timing_decorator
    def load_existing_embeddings(self, file_name: str) -> Optional[List[Dict]]:
        """Load embeddings from cache or MongoDB using file_name"""
        # Check in-memory cache first
        if file_name in self.embeddings_cache:
            self.logger.info(f"Loading embeddings from cache for: {file_name}")
            return self.embeddings_cache[file_name]
        
        try:
            # Use projection to only get required fields
            cursor = self.mongo_collection.find(
                {"file_name": file_name},
                {"chunk_text": 1, "embedding": 1, "metadata": 1, "_id": 0}
            )
            results = list(cursor)
            
            if results:
                # Cache the results
                self._update_cache(file_name, results)
                self.logger.info(f"Loaded {len(results)} embeddings from MongoDB for: {file_name}")
                return results
            return None
        except Exception as e:
            self.logger.error(f"Error loading embeddings from MongoDB: {e}")
            return None

    def _update_cache(self, file_name: str, embeddings: List[Dict]):
        """Update in-memory cache with size limit"""
        if len(self.embeddings_cache) >= self.max_cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.embeddings_cache))
            del self.embeddings_cache[oldest_key]
        
        self.embeddings_cache[file_name] = embeddings

    @timing_decorator
    def save_embeddings(self, file_name: str, embeddings_data: List[Dict]):
        """Save embeddings into MongoDB using file_name as unique key"""
        try:
            # Remove existing entries with the same file_name
            self.mongo_collection.delete_many({"file_name": file_name})

            # Prepare documents for bulk insert
            documents = []
            for chunk in embeddings_data:
                doc = {
                    "file_name": file_name,
                    "chunk_text": chunk["chunk_text"],
                    "embedding": chunk["embedding"],
                    "metadata": chunk.get("metadata", {})
                }
                documents.append(doc)

            # Bulk insert for better performance
            if documents:
                self.mongo_collection.insert_many(documents, ordered=False)
                # Update cache
                self._update_cache(file_name, embeddings_data)
                self.logger.info(f"Saved {len(embeddings_data)} chunks for file_name: {file_name} into MongoDB")

        except Exception as e:
            self.logger.error(f"Error saving embeddings to MongoDB: {e}")
            self.logger.error(traceback.format_exc())

    @contextlib.contextmanager
    def temporary_directory(self):
        """Context manager for creating and cleaning up temporary directory"""
        temp_dir = tempfile.mkdtemp(prefix="document_processing_")
        try:
            yield Path(temp_dir)
        finally:
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                self.logger.error(f"Error cleaning up temporary directory {temp_dir}: {e}")

    @timing_decorator
    def download_file_from_url(self, url: str, temp_dir: Path) -> Path:
        """Download file from URL to temporary directory"""
        parsed_url = urlparse(url)
        
        if parsed_url.scheme == 's3':
            return self._download_from_s3_url(url, temp_dir)
        elif parsed_url.scheme in ['http', 'https']:
            return self._download_from_http(url, temp_dir)
        else:
            raise ValueError(f"Unsupported URL scheme: {parsed_url.scheme}")

    def _download_from_s3_url(self, s3_url: str, temp_dir: Path) -> Path:
        """Download file from S3 URL to temporary directory"""
        try:
            parsed_url = urlparse(s3_url)
            filename = os.path.basename(parsed_url.path)
            
            if not filename:
                filename = f"downloaded_file_{uuid.uuid4().hex[:8]}"
            
            file_path = temp_dir / filename
            
            response = requests.get(s3_url, stream=True)
            response.raise_for_status()
            
            with open(file_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
            
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error downloading from S3 URL {s3_url}: {e}")
            raise

    def _download_from_http(self, url: str, temp_dir: Path) -> Path:
        """Download file from HTTP/HTTPS URL to temporary directory"""
        try:
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            
            if not filename or '.' not in filename:
                filename = f"downloaded_file_{uuid.uuid4().hex[:8]}"
                
                response_head = requests.head(url, allow_redirects=True)
                content_disposition = response_head.headers.get('Content-Disposition', '')
                if 'filename=' in content_disposition:
                    filename = content_disposition.split('filename=')[1].strip('"\'')
            
            file_path = temp_dir / filename
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error downloading from HTTP URL {url}: {e}")
            raise

    def download_file_from_s3_direct(self, bucket: str, key: str, temp_dir: Path) -> Path:
        """Download file directly from S3 using boto3"""
        try:
            if not self.s3_client:
                raise Exception("S3 client not initialized")
            
            filename = os.path.basename(key)
            if not filename:
                filename = f"s3_file_{uuid.uuid4().hex[:8]}"
            
            file_path = temp_dir / filename
            
            self.s3_client.download_file(bucket, key, str(file_path))
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error downloading from S3 bucket {bucket}, key {key}: {e}")
            raise

    def generate_document_id(self, file_name: str, document_url: str) -> str:
        """Generate a unique document ID"""
        unique_string = f"{file_name}|{document_url}"
        return hashlib.md5(unique_string.encode()).hexdigest()

    @timing_decorator
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

            elif file_extension.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']:
                image = Image.open(file_path)
                text_data = pytesseract.image_to_string(image)

                if text_data.strip():
                    txt_metadata = {
                        **file_metadata,
                        "source_type": "ocr"
                    }
                    chunks = self.create_chunks(text_data, txt_metadata)
                    all_chunks.extend(chunks)

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

            elif file_extension in [".html", ".htm"]:
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
            else:
                self.logger.warning(f"Unsupported file extension: {file_extension}")

        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            self.logger.error(traceback.format_exc())

        return all_chunks

    def create_chunks(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create text chunks with metadata"""
        try:
            chunks = self.token_splitter.create_documents(
                texts=[text],
                metadatas=[metadata]
            )
        except Exception as e:
            self.logger.warning(f"Falling back to recursive splitter due to: {e}")
            chunks = self.backup_splitter.create_documents(
                texts=[text],
                metadatas=[metadata]
            )

        return [{
            "chunk_text": chunk.page_content,
            "metadata": chunk.metadata
        } for chunk in chunks]

    @timing_decorator
    def generate_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """Generate embeddings for text chunks with batch processing"""
        if not chunks:
            return chunks
            
        # Extract texts for batch processing
        texts = [chunk['chunk_text'] for chunk in chunks]
        
        # Generate embeddings in batch (much faster than one-by-one)
        embeddings = self.embedding_model.encode(
            texts, 
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32  # Adjust based on your memory
        )
        
        # Assign embeddings back to chunks
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i].tolist()
            
        return chunks

    @timing_decorator
    def cosine_similarity_batch(self, query_embedding: np.ndarray, chunk_embeddings: np.ndarray) -> np.ndarray:
        """Vectorized cosine similarity calculation for batch processing"""
        # Normalize vectors for efficient cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        chunk_norms = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
        
        # Compute cosine similarity using dot product
        similarities = np.dot(chunk_norms, query_norm)
        return similarities

    @timing_decorator
    def select_relevant_chunks(self, query: str, embeddings_data: List[Dict], top_k: int = 5):
        """Optimized chunk selection using vectorized operations"""
        if not embeddings_data:
            return []
            
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        
        # Extract embeddings and create numpy array for vectorized operations
        valid_chunks = []
        chunk_embeddings_list = []
        
        for chunk in embeddings_data:
            if "embedding" in chunk and chunk["embedding"]:
                valid_chunks.append(chunk)
                chunk_embeddings_list.append(np.array(chunk["embedding"]))
        
        if not chunk_embeddings_list:
            return []
        
        # Stack embeddings for batch processing
        chunk_embeddings = np.stack(chunk_embeddings_list)
        
        # Compute similarities in batch
        similarities = self.cosine_similarity_batch(query_embedding, chunk_embeddings)
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return top chunks with scores
        return [(valid_chunks[i], similarities[i]) for i in top_indices]

    def rerank_results(self, query_embedding: np.ndarray, retrieved_chunks: List[Tuple]):
        """Optimized reranking with vectorized operations"""
        if not retrieved_chunks:
            return None, [], {}
            
        # Unpack chunks and scores
        chunks_only = [chunk for chunk, _ in retrieved_chunks]
        initial_scores = [score for _, score in retrieved_chunks]
        
        # Extract embeddings and similarities
        embeddings = np.array([chunk["embedding"] for chunk in chunks_only])
        similarities = self.cosine_similarity_batch(query_embedding, embeddings)
        
        chunk_lengths = np.array([len(chunk["chunk_text"]) for chunk in chunks_only])

        # Vectorized normalization
        similarities_normalized = (similarities - similarities.min()) / (similarities.max() - similarities.min() + 1e-9)
        lengths_normalized = (chunk_lengths - chunk_lengths.min()) / (chunk_lengths.max() - chunk_lengths.min() + 1e-9)
        
        combined_scores = (similarities_normalized + lengths_normalized) / 2

        # Sort by combined scores
        sorted_indices = np.argsort(combined_scores)[::-1]
        sorted_chunks = [chunks_only[i] for i in sorted_indices]
        
        return sorted_chunks[0], sorted_chunks, {
            'combined_scores': combined_scores[sorted_indices].tolist(),
            'similarity_scores': similarities[sorted_indices].tolist(),
            'initial_scores': initial_scores
        }

    def generate_response_stream(self, query: str, best_chunk: dict):
        best_text = best_chunk.get("chunk_text", "") if isinstance(best_chunk, dict) else str(best_chunk)
        input_text = f"Query: {query}\nRelevant Context: {best_text}\nAnswer:"

        try:
            for chunk in ollama.chat(
                model=self.ollama_model,
                messages=[{"role": "user", "content": input_text}],
                stream=True
            ):
                yield chunk["message"]["content"]
        except Exception as e:
            self.logger.error(f"Error streaming response: {e}")
            yield "[Error generating response]"

    def process_and_query(self, document_url: str, query: str, document_id: str = None,
                          output_dir: Path = None, base_dir: Path = None,
                          save_output_file: bool = False, top_k: int = 5, stream: bool = False):

        if base_dir is not None:
            output_dir = base_dir
        elif output_dir is None:
            output_dir = Path('./document_processing')

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            file_name = os.path.basename(urlparse(document_url).path)
            if not file_name:
                return iter(["Document name is missing."]) if stream else {"response": "Document name is missing."}

            embeddings_with_metadata = self.load_existing_embeddings(file_name)

            if not embeddings_with_metadata:
                with self.temporary_directory() as temp_dir:
                    local_file_path = self.download_file_from_url(document_url, temp_dir)
                    chunks = self.process_document(local_file_path)
                    if not chunks:
                        return iter(["No chunks created."]) if stream else {"response": "No chunks created."}

                    embeddings_with_metadata = self.generate_embeddings(chunks)
                    self.save_embeddings(file_name, embeddings_with_metadata)

            chunks_with_scores = self.select_relevant_chunks(query, embeddings_with_metadata, top_k)
            if not chunks_with_scores:
                return iter(["No relevant information found."]) if stream else {
                    'response': "No relevant information found.",
                    'relevant_chunks': [], 'similarity_scores': [],
                    'reranked_chunks': [], 'score_details': {}, 'document_id': document_id
                }

            relevant_chunks = [chunk for chunk, _ in chunks_with_scores]
            similarity_scores = [float(score) for _, score in chunks_with_scores]

            query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
            best_chunk, reranked_chunks, score_details = self.rerank_results(query_embedding, chunks_with_scores)

            if stream:
                return self.generate_response_stream(query, best_chunk) if best_chunk else iter(["No relevant information found."])

            response = self.generate_response(query, best_chunk) if best_chunk else "No relevant information found."
            output_data = {
                'response': response,
                'relevant_chunks': relevant_chunks,
                'similarity_scores': similarity_scores,
                'reranked_chunks': reranked_chunks,
                'score_details': score_details,
                'document_id': document_id
            }

            if save_output_file:
                output_file = output_dir / "output.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=4, ensure_ascii=False, default=str)

            return output_data

        except Exception as e:
            self.logger.error(f"Error in document processing and querying: {e}")
            self.logger.error(traceback.format_exc())
            return iter(["[Error occurred during processing]"]) if stream else {
                'response': "An error occurred during processing.",
                'document_id': document_id
            }

    def clear_cache(self):
        """Clear in-memory cache"""
        self.embeddings_cache.clear()
        self.logger.info("Cache cleared")

    def get_cache_info(self):
        """Get cache statistics"""
        return {
            "cache_size": len(self.embeddings_cache),
            "cached_files": list(self.embeddings_cache.keys())
        }



    