import os
from groq import Groq

# Get free API key from: https://console.groq.com
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks):
        parts.append(f"[Page {chunk['page']}, Chunk {i+1}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def get_answer(question: str, chunks: list[dict], chat_history: list[dict]) -> tuple[str, list[dict]]:
    context = build_context(chunks)

    messages = [
        {
            "role": "system",
            "content": """You are a precise document Q&A assistant.
Answer questions strictly based on the provided document context.

Rules:
- Only use information from the provided context.
- If the answer is not in the context, say "I could not find that information in the document."
- Always cite which page(s) your answer comes from.
- Be concise but complete.
- Format your answer clearly with bullet points where appropriate."""
        }
    ]

    # Add chat history
    for msg in chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Add current question with context
    messages.append({
        "role": "user",
        "content": f"""Document Context:
{context}

---

Question: {question}

Please answer based only on the context above and cite the page numbers."""
    })

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # Free, fast, powerful
        messages=messages,
        max_tokens=1024,
        temperature=0.1
    )

    answer = response.choices[0].message.content

    # Extract sources
    sources = []
    seen_pages = set()
    for chunk in chunks[:3]:
        if chunk["page"] not in seen_pages:
            sources.append({
                "page": chunk["page"],
                "excerpt": chunk["text"][:200] + "..."
            })
            seen_pages.add(chunk["page"])

    return answer, sources
