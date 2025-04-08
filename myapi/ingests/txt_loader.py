from pathlib import Path
from typing import Optional, List, Dict

def load_data(
    file_path: str | Path, extra_info: Optional[Dict] = None, **kwargs
) -> List[Dict]:
    """Reads and processes the content of a text file, and returns structured data."""
    try:
        # Ensure the file path is a Path object
        file_path = Path(file_path)

        # Validate that the file path is a string or Path
        if not isinstance(file_path, (str, Path)):
            raise TypeError(f"Expected file path to be a string or Path object, got {type(file_path)}.")

        # Read the text file
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        # Handle empty text case
        if not text:
            raise ValueError(f"The file {file_path} is empty.")

    except FileNotFoundError:
        raise FileNotFoundError(f"The file {file_path} was not found.")
    except ValueError as e:
        raise e  # Re-raise for empty file case
    except Exception as e:
        raise RuntimeError(f"Error reading {file_path}: {e}")

    # Include file metadata
    metadata = {"file_name": file_path.name, "file_path": str(file_path.resolve())}
    if extra_info:
        metadata.update(extra_info)

    # Return the structured result with text and metadata
    return [{"text": text, "metadata": metadata}]
