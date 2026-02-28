"""Interactive CLI test for LLM and embeddings."""

from app.chatbot.agent.llm import get_model, get_embeddings


def main():
    """Test LLM response generation."""
    model = get_model()
    while True:
        user_input = input("Enter your query (press 'exit' or 'quit' to quit): ")
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting chat.")
            break
        response = model.invoke([{"role": "user", "content": user_input}])
        print(f"LLM: {response.content}")


def test_embeddings():
    """Test the embeddings generation."""
    embeddings = get_embeddings()
    sample_text = "This is a sample text for embedding."
    embedding_vector = embeddings.embed_query(sample_text)
    print(f"Embedding vector for sample text: {embedding_vector[:10]}, total dims: {len(embedding_vector)}")


if __name__ == "__main__":
    main()
    # test_embeddings()

# python -m app.chatbot.test_llm
