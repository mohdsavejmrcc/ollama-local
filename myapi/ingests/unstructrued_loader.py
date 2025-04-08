from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.core.readers.base import BaseReader

class UnstructuredReader(BaseReader):
    """General unstructured text reader for processing documents locally."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the reader with local processing only."""
        super().__init__(*args)
        self.api = False  # Always use local processing

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        split_documents: Optional[bool] = False,
        **kwargs,
    ) -> List[Dict]:
        """Process the file locally and extract structured text data."""
        file_path_str = str(file)

        # Use local partitioning to extract text from the document
        try:
            from unstructured.partition.auto import partition
            elements = partition(filename=file_path_str)
        except ImportError:
            raise ImportError("The 'unstructured' library is required but not installed. Install it via 'pip install unstructured'.")

        # Process extracted elements
        docs = []
        file_name = file.name
        file_path = str(file.resolve())

        if split_documents:
            # Each element becomes a separate document
            for node in elements:
                metadata = {"file_name": file_name, "file_path": file_path}
                if hasattr(node, "metadata"):
                    for field, val in vars(node.metadata).items():
                        if field not in ["_known_field_names", "coordinates", "parent_id"]:
                            metadata[field] = val

                if extra_info:
                    metadata.update(extra_info)

                docs.append({"text": node.text.strip(), "metadata": metadata})
        else:
            # Combine all extracted text into one document
            text_chunks = [node.text.strip() for node in elements if node.text]
            metadata = {"file_name": file_name, "file_path": file_path}
            if extra_info:
                metadata.update(extra_info)

            docs.append({"text": "\n\n".join(text_chunks), "metadata": metadata})

        return docs