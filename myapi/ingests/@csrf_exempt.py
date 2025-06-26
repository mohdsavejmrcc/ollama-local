@csrf_exempt
def process_query(request):
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8"))
        query_text = data.get("query")
        document_url = data.get("document_url")
        document_id = data.get("document_id")
        query_id = data.get("id", "query-" + str(hash(query_text))[:8])

        if not query_text:
            return JsonResponse({"error": "Missing 'query'"}, status=400)

        if not document_url and not document_id:
            return JsonResponse({"error": "Either document_url or document_id must be provided"}, status=400)

        assistant = DocumentQueryAssistant()
        base_dir = Path('./document_processing')
        base_dir.mkdir(parents=True, exist_ok=True)

        def stream_response():
            try:
                for token in assistant.process_and_query(
                    document_url=document_url,
                    query=query_text,
                    document_id=document_id,
                    base_dir=base_dir,
                    stream=True  # ✅ Enable streaming
                ):
                    yield token
            except Exception as e:
                logger.error(f"Streaming error: {e}", exc_info=True)
                yield "\n[Error streaming response]\n"

        return StreamingHttpResponse(stream_response(), content_type="text/plain")

    except json.JSONDecodeError:
        logger.error("Invalid JSON received", exc_info=True)
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return JsonResponse({"error": f"An error occurred: {str(e)}"}, status=500)
    
    