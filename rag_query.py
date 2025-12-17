import requests
import vectorstore_manager
try:
    from config import RAG_API_KEY, RAG_API_URL, RAG_MODEL
except ImportError:
    # Fallback if config not available
    RAG_API_KEY = "sk-voidai-zAFjyYrdhOfQwZahZFXpJm4U3DC4wni8BgbWyxi99B05kiIxkCitl3LjE6z09DBsCNuJnUE0JAGwcZleA4BGJOVaAnHWp6TuHQRQIRQK5EEnuHCRBTGh0hpSDT3kib_LkY19HQ"
    RAG_API_URL = "https://api.voidai.app/v1/chat/completions"
    RAG_MODEL = "gpt-4o"

API_KEY = RAG_API_KEY
API_URL = RAG_API_URL
MODEL = RAG_MODEL


def rag_query(question: str) -> str:
    """
    Performs a RAG query using the vectorstore for context retrieval
    and VoidAI GPT-4o model for answering.
    """
    # 1️⃣ Check if vectorstore initialized
    if vectorstore_manager.vectorstore is None:
        return "Error: Vectorstore not initialized."

    vectorstore = vectorstore_manager.vectorstore

    # 2️⃣ Retrieve relevant chunks
    retrieved_docs = vectorstore.similarity_search(question, k=10)
    if not retrieved_docs:
        return "No relevant documents found."

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # 3️⃣ Build the API request payload
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an information extraction assistant.\n"
                    "Use ONLY the provided context to answer the question.\n\n"
                    "Rules:\n"
                    "- Do NOT use markdown.\n"
                    "- Do NOT use code blocks or backticks.\n"
                    "- Do NOT use bullet symbols.\n"
                    "- Write the answer as plain text in complete sentences.\n"
                    "- If the answer is not present in the context, reply exactly:\n"
                    "  Not found in the context."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            }
        ],
        "max_tokens": 300,
        "temperature": 0.2,
        "top_p": 0.9
    }


    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"  # ✅ Correct Bearer header
    }

    # 4️⃣ Send request to VoidAI
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        # 5️⃣ Extract answer from model response
        message = data.get("choices", [{}])[0].get("message", {})
        answer = message.get("content", "").strip()
        return answer or "No content returned from model."

    except requests.exceptions.HTTPError as e:
        return f"HTTP error: {response.text}"
    except Exception as e:
        return f"Error during query: {str(e)}"


# 🧠 Example usage:
# print(rag_query("What is the main clause in the policy?"))
