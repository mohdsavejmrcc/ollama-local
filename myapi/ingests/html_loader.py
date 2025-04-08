import email
from pathlib import Path
from typing import Optional, List, Dict

from bs4 import BeautifulSoup

def parse_html(file_path: Path | str, page_break_pattern: Optional[str] = None, extra_info: Optional[dict] = None) -> List[Dict]:
    """Parse HTML file and return a list of documents.

    Args:
        file_path (Path | str): Path to the HTML file.
        page_break_pattern (Optional[str]): Pattern to split the HTML into pages.
        extra_info (Optional[dict]): Extra information passed to this reader during extracting data.

    Returns:
        list: List of documents extracted from the HTML file.
    """
    try:
        import html2text  # noqa
    except ImportError:
        raise ImportError(
            "html2text is not installed. "
            "Please install it using `pip install html2text`"
        )

    file_path = Path(file_path).resolve()

    try:
        with file_path.open("r", encoding="utf-8") as f:
            html_text = f.read()
    except UnicodeDecodeError:
        # Handle different encoding errors
        print(f"Error reading {file_path}, trying different encoding.")
        with file_path.open("r", encoding="ISO-8859-1") as f:
            html_text = f.read()

    # Convert HTML to plain text using html2text
    all_text = html2text.html2text(html_text)
    
    # Split text into pages based on the provided pattern or treat as a single page
    pages = all_text.split(page_break_pattern) if page_break_pattern else [all_text]

    extra_info = extra_info or {}

    # Create Documents from the pages
    documents = [
        {
            "text": page.strip(),
            "metadata": {"page_label": page_id + 1, **extra_info},
        }
        for page_id, page in enumerate(pages)
    ]

    return documents


def parse_mhtml(file_path: Path | str, open_encoding: Optional[str] = "utf-8", bs_kwargs: Optional[dict] = None) -> List[Dict]:
    """Parse MHTML file into document objects.

    Args:
        file_path (Path | str): Path to the MHTML file.
        open_encoding (Optional[str]): Encoding to use when opening the file.
        bs_kwargs (Optional[dict]): Any kwargs to pass to BeautifulSoup.

    Returns:
        list: List of document objects extracted from the MHTML file.
    """
    extra_info = {}

    # Default BeautifulSoup kwargs
    if bs_kwargs is None:
        bs_kwargs = {"features": "lxml"}

    page = []
    file_name = Path(file_path)

    try:
        with open(file_path, "r", encoding=open_encoding) as f:
            message = email.message_from_string(f.read())
            parts = message.get_payload()

            if not isinstance(parts, list):
                parts = [message]

            for part in parts:
                if part.get_content_type() == "text/html":
                    html = part.get_payload(decode=True).decode()

                    soup = BeautifulSoup(html, **bs_kwargs)
                    text = soup.get_text()

                    # Extract title if available
                    title = str(soup.title.string) if soup.title else ""

                    metadata = {
                        "source": str(file_path),
                        "title": title,
                        **extra_info,
                    }

                    # Process the text
                    lines = [line for line in text.split("\n") if line.strip()]
                    text = "\n\n".join(lines)
                    if text:
                        page.append(text)

    except Exception as e:
        print(f"Error parsing MHTML file {file_name}: {e}")

    # Return the processed result
    return [{"text": "\n\n".join(page), "metadata": metadata}]
