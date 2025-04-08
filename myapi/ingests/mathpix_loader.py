import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
from langchain.utils import get_from_dict_or_env

def process_and_upload_pdf(file_path: Union[str, Path], processed_file_format: str = "md", max_wait_time_seconds: int = 900, should_clean_pdf: bool = True, extra_info: Optional[Dict] = None, **kwargs: Any) -> List[Dict]:
    """Process and upload a PDF using the Mathpix API."""
    
    # Get API credentials
    mathpix_api_key = get_from_dict_or_env(kwargs, "mathpix_api_key", "MATHPIX_API_KEY", default="empty")
    mathpix_api_id = get_from_dict_or_env(kwargs, "mathpix_api_id", "MATHPIX_API_ID", default="empty")
    
    def _mathpix_headers() -> Dict[str, str]:
        return {"app_id": mathpix_api_id, "app_key": mathpix_api_key}
    
    def send_pdf(file_path: Path) -> str:
        with open(file_path, "rb") as f:
            files = {"file": f}
            response = requests.post(
                "https://api.mathpix.com/v3/pdf", headers=_mathpix_headers(), files=files, data={"options_json": json.dumps({"conversion_formats": {processed_file_format: True}, "enable_tables_fallback": True})}
            )
        response_data = response.json()
        if "pdf_id" in response_data:
            return response_data["pdf_id"]
        else:
            raise ValueError("Unable to send PDF to Mathpix.")
    
    def wait_for_processing(pdf_id: str) -> None:
        url = f"https://api.mathpix.com/v3/pdf/{pdf_id}"
        for _ in range(0, max_wait_time_seconds, 5):
            response = requests.get(url, headers=_mathpix_headers())
            response_data = response.json()
            status = response_data.get("status", None)
            print(f"Processing status: {status}, Progress: {response_data.get('percent_done', 0)}%")
            if status == "completed":
                return
            elif status == "error":
                raise ValueError(f"Mathpix processing error: {response_data}")
            time.sleep(5)
        raise TimeoutError(f"Processing did not complete within {max_wait_time_seconds} seconds")
    
    def get_processed_pdf(pdf_id: str) -> str:
        wait_for_processing(pdf_id)
        url = f"https://api.mathpix.com/v3/pdf/{pdf_id}.{processed_file_format}"
        response = requests.get(url, headers=_mathpix_headers())
        if response.status_code != 200:
            raise ValueError(f"Failed to get processed PDF: {response.text}")
        return response.content.decode("utf-8")
    
    def clean_pdf(contents: str) -> str:
        """Clean the PDF contents by removing unwanted characters."""
        contents = "\n".join([line for line in contents.split("\n") if not line.startswith("![]")])
        contents = contents.replace("\\section{", "# ").replace("}", "").replace("{", "").replace("\\text", "").replace("\\mathrm", "")
        return contents
    
    def parse_markdown_text_to_tables(content: str) -> tuple[List[tuple[int, str]], List[tuple[int, str]]]:
        """Parse markdown content to extract tables and text."""
        pages = re.split(r"(?m)^# Page \d+\n", content)
        tables = []
        texts = []
        for page_num, page_content in enumerate(pages, 1):
            if not page_content.strip():
                continue
            table_matches = re.findall(r"(\|[^\n]+\|(?:\n\|[^\n]+\|)*)", page_content)
            for table in table_matches:
                tables.append((page_num, table.strip()))
            page_content = re.sub(r"(\|[^\n]+\|(?:\n\|[^\n]+\|)*)", "", page_content)
            chunks = re.split(r"\n\s*\n", page_content)
            for chunk in chunks:
                if chunk.strip():
                    texts.append((page_num, chunk.strip()))
        return tables, texts
    
    # Load and process PDF
    pdf_id = send_pdf(Path(file_path))
    content = get_processed_pdf(pdf_id)
    
    if should_clean_pdf:
        content = clean_pdf(content)
    
    tables, texts = parse_markdown_text_to_tables(content)
    
    # Prepare documents
    documents = []
    for page_num, table_content in tables:
        text = table_content.strip()
        metadata = {"table_origin": table_content, "type": "table", "page_label": page_num, "page_number": page_num}
        if extra_info:
            metadata.update(extra_info)
        documents.append({"text": text, "metadata": metadata})
    
    for page_num, text_content in texts:
        if text_content.strip():
            metadata = {"source": str(file_path), "type": "text", "page_label": page_num, "page_number": page_num}
            if extra_info:
                metadata.update(extra_info)
            documents.append({"text": text_content, "metadata": metadata})
    
    if not documents and content.strip():
        metadata = {"source": str(file_path), "type": "text", "page_label": 1, "page_number": 1}
        if extra_info:
            metadata.update(extra_info)
        documents.append({"text": content.strip(), "metadata": metadata})
    
    return documents
