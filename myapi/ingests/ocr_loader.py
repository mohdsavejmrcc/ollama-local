import fitz  # PyMuPDF
import pytesseract
from pytesseract import Output
from PIL import Image
import io
import os
import json     
import base64

# Ensure Tesseract is installed correctly in Colab
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Adjust this path if necessary
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Function to extract images and text from a PDF
def extract_images_and_text_from_pdf(pdf_path):
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)

    # Get file metadata (name, path, etc.)
    file_name = os.path.basename(pdf_path)
    file_path = os.path.abspath(pdf_path)

    # Initialize the result object
    results = []

    # Loop through each page in the PDF
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        images = page.get_images(full=True)
        print(f"Processing page {page_num + 1}/{len(pdf_document)}")

        # Initialize page data with metadata
        page_data = {
            "page_number": page_num + 1,
            "text": page.get_text("text"),
            "images": [],
            "ocr_tables": [],
            "metadata": {
                "file_name": file_name,
                "file_path": file_path
            }
        }

        # Process images on the page
        for img_index, img_info in enumerate(images):
            xref = img_info[0]  # XREF of the image
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # Convert image to base64 string
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            page_data["images"].append({
                "index": img_index + 1,
                "format": image_ext,
                "image_base64": image_base64
            })

            # Perform OCR on the image
            img = Image.open(io.BytesIO(image_bytes))
            ocr_result = pytesseract.image_to_string(img, config="--psm 6")  # PSM 6 is good for tables
            page_data["ocr_tables"].append({
                "image_index": img_index + 1,
                "ocr_text": ocr_result
            })

        # Append page data to results
        results.append(page_data)

    # Return the results after processing all pages
    return results
 