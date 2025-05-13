from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import logging
from pathlib import Path
from .ingests.chat8 import DocumentQueryAssistant  # Import the class

# Set up logging
logger = logging.getLogger(__name__)

@csrf_exempt  # Remove in production if using CSRF protection
def process_query(request):
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)
    
    try:
        # Ensure request body contains valid JSON
        data = json.loads(request.body.decode("utf-8"))
        
        # Extract parameters from the request - match Node.js parameter names
        query_text = data.get("query")
        document_url = data.get("document_url")
        document_id = data.get("document_id")
        query_id = data.get("id", "query-" + str(hash(query_text))[:8])  # Generate ID if not provided
        
        # Validate required fields
        if not query_text:
            return JsonResponse({"error": "Missing 'query'"}, status=400)
        
        if not document_url and not document_id:
            return JsonResponse({"error": "Either document_url or document_id must be provided"}, status=400)
        
        # Initialize DocumentQueryAssistant
        assistant = DocumentQueryAssistant()
        
        # Set up base directory for processing
        base_dir = Path('./document_processing')
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Process the document and query
        result = assistant.process_and_query(
            document_url=document_url,
            query=query_text,
            document_id=document_id,
            base_dir=base_dir
        )
        
        # Handle failure case
        if not result:
            return JsonResponse({
                "id": query_id,
                "error": "Failed to process document or query"
            }, status=500)
        
        # Construct response payload
        response_data = {
            "id": query_id,
            "query": query_text,
            "response": result.get('response', "No response generated"),
            "document_id": result.get('document_id', document_id)
        }
        
        # Include relevant chunks and their scores if available
        relevant_chunks = result.get("relevant_chunks", [])
        similarity_scores = result.get("similarity_scores", [])
        
        if relevant_chunks and similarity_scores:
            response_data["relevant_chunks"] = [
                {
                    "text": chunk.get('chunk_text', ''),
                    "metadata": chunk.get('metadata', {}),
                    "score": score
                }
                for chunk, score in zip(relevant_chunks, similarity_scores)
            ]
        
        return JsonResponse(response_data, status=200)
    
    except json.JSONDecodeError:
        logger.error("Invalid JSON received", exc_info=True)
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return JsonResponse({"error": f"An error occurred: {str(e)}"}, status=500)