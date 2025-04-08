import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, List, Dict

def xml_data(file_path: Path | str, extra_info: Optional[dict] = None) -> List[Dict]:
    file_path = Path(file_path).resolve()

    # Check if the file exists and is not empty
    if not file_path.exists() or file_path.stat().st_size == 0:
        raise ValueError(f"The file at {file_path} is empty or does not exist.")
    
    try:
        # Parse the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        # Handle XML parsing errors
        raise ValueError(f"Failed to parse XML file: {e}")
    
    extra_info = extra_info or {}
    data = []
    
    # Iterate through all child elements of the root (flexible, works with any root)
    for element in root.iter():
        text_content = (element.text or "").strip()
        
        # Only process elements with non-empty text content
        if text_content:
            element_data = {
                "text": text_content,
                "metadata": {
                    "tag": element.tag,
                    "file_path": str(file_path),
                    "file_name": file_path.name
                }
            }
            
            # Check if the element contains an 'image' tag and add it to metadata
            if element.tag.lower() == "image":
                element_data["metadata"]["image"] = text_content  # Assuming the image is specified as text (URL or file path)
            
            data.append(element_data)
    
    return data
