import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

from fsspec import AbstractFileSystem
from llama_index.readers.file import PDFReader
from PIL import Image

def get_page_thumbnails(
    file_path: Path, pages: List[int], dpi: int = 80
) -> List[str]:
    """Get image thumbnails of the pages in the PDF file.

    Args:
        file_path (Path): path to the PDF file
        pages (List[int]): list of page numbers to extract

    Returns:
        List[str]: list of base64-encoded page thumbnails
    """
    # Ensure the file is a PDF
    suffix = file_path.suffix.lower()
    assert suffix == ".pdf", "This function only supports PDF files."

    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("Please install PyMuPDF: 'pip install PyMuPDF'")

    # Open the PDF document
    doc = fitz.open(file_path)
    output_imgs = []

    # Process each page number
    for page_number in pages:
        # Load the page and generate its pixmap
        page = doc.load_page(page_number)
        pixmap = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
        # Convert the image to base64
        output_imgs.append(convert_image_to_base64(img))

    return output_imgs

def convert_image_to_base64(img: Image.Image) -> str:
    """Convert an image to base64 encoding."""
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    img_base64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_base64}"

class PDFThumbnailReader(PDFReader):
    """PDF parser with thumbnail for each page."""

    def __init__(self) -> None:
        """Initialize PDFReader."""
        super().__init__(return_full_document=False)

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Dict]:
        """Parse file and add thumbnails."""
        # Parse the PDF file using the parent class's load_data method
        documents = super().load_data(file, extra_info, fs)

        # Prepare to extract page numbers and generate thumbnails
        page_numbers_str = []
        filtered_docs = []
        is_int_page_number: Dict[str, bool] = {}

        # Filter documents based on page labels (metadata)
        for doc in documents:
           if hasattr(doc, "metadata") and "page_label" in doc.metadata:
            page_num_str = doc.metadata["page_label"]
            page_numbers_str.append(page_num_str)
            try:
                _ = int(page_num_str)
                is_int_page_number[page_num_str] = True
                filtered_docs.append(doc)
            except ValueError:
                is_int_page_number[page_num_str] = False


        # Only process documents with valid page numbers
        documents = filtered_docs
        page_numbers = list(range(len(page_numbers_str)))

        # Get page thumbnails for the valid pages
        print("Processing page numbers:", len(page_numbers))
        page_thumbnails = get_page_thumbnails(file, page_numbers)

        # Add the page thumbnails to documents
        documents.extend(
            [
                {
                    "text": "Page thumbnail",  # Optional: Replace with any relevant description
                    "metadata": {
                        "image_origin": page_thumbnail,
                        "type": "thumbnail",
                        "page_label": page_number,
                        **(extra_info if extra_info else {}),
                    },
                }
                for page_thumbnail, page_number in zip(page_thumbnails, page_numbers_str)
                if is_int_page_number.get(page_number, False)
            ]
        )

        return documents
