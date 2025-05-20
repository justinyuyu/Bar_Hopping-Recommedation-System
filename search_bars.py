import sys
from barhopping.retriever.vector_search import get_vector_search
from barhopping.logger import logger

def format_result(result: dict, index: int) -> str:
    """Format a single search result for display."""
    name = result.get("tag_name", "Unknown")
    summary = result.get("summary", "No summary available")
    vector_score = result.get("vector_score", 0.0)
    rerank_score = result.get("rerank_score", 0.0)

    return (
        f"\nResult {index + 1}:\n"
        f"Name: {name}\n"
        f"Summary: {summary}\n"
        f"Vector Score: {vector_score:.3f}\n"
        f"Rerank Score: {rerank_score:.3f}\n"
    )

def search_loop(vector_search):
    """Interactive search loop."""
    print("\nBar Search (type 'quit' to exit)")
    print("Enter your search query:")
    
    while True:
        try:
            query = input("\n> ").strip()
            if query.lower() == "quit":
                print("Exiting search.")
                break
            if not query:
                continue
                
            results = vector_search.search(query, top_k=5)
            if not results:
                print("No results found.")
                continue
                
            print(f"\nFound {len(results)} results:")
            for i, result in enumerate(results):
                print(format_result(result, i))
                
        except KeyboardInterrupt:
            print("\nInterrupted. Goodbye!")
            break
        except Exception as e:
            logger.exception("Error during search")
            print(f"An error occurred: {e}")

def main():
    try:
        vector_search = get_vector_search()
        search_loop(vector_search)
    except Exception as e:
        logger.exception("Failed to start vector search")
        sys.exit(f"Startup error: {e}")

if __name__ == "__main__":
    main()