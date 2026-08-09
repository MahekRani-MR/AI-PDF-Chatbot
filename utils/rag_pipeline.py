import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


def create_rag_chain(vectorstore):
    """
    Creates a RAG pipeline using:
    FAISS + HuggingFace Embeddings + Groq Llama 3.1
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Please check your .env file."
        )

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key,
        temperature=0.2
    )

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = ChatPromptTemplate.from_template(
        """
You are an intelligent research assistant.

You are answering a question about an uploaded academic PDF.

Use the retrieved document context carefully.

IMPORTANT INSTRUCTIONS:

1. Answer the question using the provided context.

2. Prefer information from the abstract, introduction,
   methodology, results, and conclusion when relevant.

3. Give a complete explanation rather than only one sentence.

4. For questions asking about a problem, objective, motivation,
   contribution, or methodology, explain the answer clearly.

5. Use bullet points when there are multiple ideas.

6. Do not make up information that is not supported by the
   retrieved context.

7. If the retrieved context genuinely does not contain the answer,
   respond exactly:

"I couldn't find that information in the uploaded document."

Retrieved Context:
------------------
{context}
------------------

Question:
{question}

Answer:
"""
    )

    # --------------------------------------------------------
    # Retriever
    # --------------------------------------------------------

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 8
        }
    )

    # --------------------------------------------------------
    # Ask Function
    # --------------------------------------------------------

    def ask(question):

        # Retrieve more chunks
        docs = retriever.invoke(question)

        # Remove duplicate chunks
        unique_docs = []

        seen = set()

        for doc in docs:

            text = doc.page_content.strip()

            if text and text not in seen:
                unique_docs.append(doc)
                seen.add(text)

        # Create context
        context_parts = []

        for doc in unique_docs:

            page = doc.metadata.get("page", None)

            if page is not None:
                page_number = page + 1

                context_parts.append(
                    f"[Page {page_number}]\n"
                    f"{doc.page_content}"
                )
            else:

                context_parts.append(
                    doc.page_content
                )

        context = "\n\n".join(context_parts)

        # Send to LLM
        messages = prompt.format_messages(
            context=context,
            question=question
        )

        response = llm.invoke(messages)

        return {
            "answer": response.content,
            "sources": unique_docs
        }

    return ask