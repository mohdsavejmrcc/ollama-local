import os
import json
from pathlib import Path
from typing import List, Dict, Any
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from ocr_loader import extract_images_and_text_from_pdf 
from txt_loader import load_data
from html_loader import parse_html, parse_mhtml
from docx_loader import docx_reader
from excel_loader import load_pandas_excel_data, load_excel_data
from  xml_loader import xml_data
from open import embedding_manager
class TextChunker:
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.token_splitter = SentenceTransformersTokenTextSplitter(
            model_name="all-MiniLM-L6-v2",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.backup_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )


    def create_chunks(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            # Try using the transformer-based splitter first
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


def doc_chunks(directory_path: Path, output_json_path: Path, chunks_json_path: Path):
    """Reads all files in a directory, processes them, and saves both original content and chunks."""
    results = []
    all_chunks = []
    chunker = TextChunker()

    if not directory_path.exists() or not directory_path.is_dir():
        raise ValueError(f"The provided path {directory_path} is not a valid directory.")

    for file in directory_path.iterdir():
        file_extension = file.suffix.lower()
        file_metadata = {"file_name": file.name, "file_path": str(file)}

        if file_extension == ".pdf":
            print(f"Processing PDF file: {file.name}")
            ocr_results = extract_images_and_text_from_pdf(file)
            result = {
                "file_name": file.name,
                "ocr_results": ocr_results
            }
            
            # Process text from OCR results
            for page in ocr_results:
                    page_metadata = {
                        **file_metadata,
                        "page_number": page.get("page_number"),
                        "source_type": "pdf"
                    }
                    if page.get("text"):

                        chunks = chunker.create_chunks(page["text"], page_metadata)
                        for chunk in chunks:
                            chunk["metadata"].update({
                                "images": page.get("images", []),
                                "ocr_tables": page.get("ocr_tables", [])
                            })
                            all_chunks.append(chunk)
                    

        elif file_extension == ".txt":
            print(f"Processing TXT file: {file.name}")
            text_data = load_data(file)
            result = {
                "file_name": file.name,
                "text_data": text_data
            }
            
            for item in text_data:
                if item.get("text"):
                    txt_metadata = {
                        **file_metadata,
                        **item.get("metadata", {}),
                        "source_type": "txt"
                    }
                    chunks = chunker.create_chunks(item["text"], txt_metadata)
                    all_chunks.extend(chunks)

        elif file_extension == ".html":
            print(f"Processing HTML file: {file.name}")
            html_data = parse_html(file)
            result = {
                "file_name": file.name,
                "html_data": html_data
            }
            
            for item in html_data:
                if item.get("text"):
                    html_metadata = {
                        **file_metadata,
                        **item.get("metadata", {}),
                        "source_type": "html"
                    }
                    chunks = chunker.create_chunks(item["text"], html_metadata)
                    all_chunks.extend(chunks)

        elif file_extension in [".xls", ".xlsx"]:
            print(f"Processing Excel file: {file.name}")
            excel_data = load_excel_data(file)
            result = {
                "file_name": file.name,
                "excel_data": excel_data
            }
            
            for item in excel_data:
                if item.get("text"):
                    excel_metadata = {
                        **file_metadata,
                        **item.get("metadata", {}),
                        "source_type": "excel"
                    }
                    chunks = chunker.create_chunks(item["text"], excel_metadata)
                    all_chunks.extend(chunks)
        elif file_extension == ".docx":
            print("Processing DOCX file :{file.name}")
            docx_data=docx_reader(file,extra_info=file_metadata)
            result = {"file_name":file.name,"docx_data":docx_data}
            for item in docx_data:
                if item.get("text"):
                   docx_metadata = {**file_metadata, **item.get("metadata", {}), "source_type": "docx"}
                   chunks = chunker.create_chunks(item["text"], docx_metadata)
                   all_chunks.extend(chunks)
                   
        elif file_extension=='.xml':
            xml_datas=xml_data(file,extra_info=file_metadata)
            result={"fil_name":file.name,"xml_data":xml_datas}
            for item in xml_datas:
                if item.get("text"):
                    xml_metadata = {**file_metadata, **item.get("metadata", {}), "source_type": "xml"}
                    chunks = chunker.create_chunks(item["text"], xml_metadata)
                    all_chunks.extend(chunks)
        else:
            print(f"Unsupported file type: {file.name}")
            continue

        results.append(result)

    # Ensure output directories exist
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_json_path.parent.mkdir(parents=True, exist_ok=True)

    # Save original document data
    with open(output_json_path, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, indent=4, ensure_ascii=False)
    print(f"Original document results saved to {output_json_path}")

    # Save chunks data
    with open(chunks_json_path, "w", encoding="utf-8") as json_file:
        json.dump(all_chunks, json_file, indent=4, ensure_ascii=False)
    print(f"Chunks saved to {chunks_json_path}")

if __name__ == "__main__":
    directory_path = Path("C:/Users/savej/Desktop/model1/dir")
    output_json_path = Path("C:/Users/savej/Desktop/model1/document_chunks.json")
    chunks_json_path = Path("C:/Users/savej/Desktop/model1/text_chunks1.json")
    doc_chunks(directory_path, output_json_path, chunks_json_path)
    embedding_manager(chunks_json_path, output_file_path="C:/Users/savej/Desktop/model1/embeddings.json")
    print("Embeddings have been generated and saved.")