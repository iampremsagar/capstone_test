import os
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel, Field
import chromadb
from chromadb.utils import embedding_functions
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# --- Configuration ---
MOCK_LLM = 0 if os.environ.get("MOCK_LLM", "1") == "0" else 1

PROMPT_TEMPLATE = """
Role: You are an Expert Zepto Support Agent.
Context: Use the following retrieved snippets from Zepto's official policy documents to answer the query.
---
{context}
---
Task: Answer the user query accurately based only on the provided context.
Negative Constraint: Do not answer using information not present in the provided context. If the answer is not in the context, state that you do not have enough information.
Length: Answer in at most 3 sentences.
Format: Your output must be a valid JSON object with the following keys:
- "answer": (string) The grounded answer to the query.
- "sources": (list of strings) The IDs of the documents used.
- "confidence": (float) A score between 0 and 1.

Few-Shot Example:
Query: "What is the delivery fee for orders under 149?"
Context: "doc_01: Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee."
Response: {{"answer": "Orders below INR 149 incur a flat delivery fee of INR 25.", "sources": ["doc_01"], "confidence": 1.0}}

Query: {query}
Response:
"""

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str

class ResponseModel(BaseModel):
    answer: str = Field(..., description="The final answer to the user's query")
    sources: List[str] = Field(default_factory=list, description="List of document/chunk IDs used to generate the answer")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score of the answer between 0 and 1")

class SimpleState(TypedDict):
    question : str
    topic : str
    answer : str
    sources : List[str]
    confidence : float

# --- ChromaDB Setup ---
# Note: In a production app, the DB would be persisted.
# For the project, we assume the 'docs' directory is present.
minilm_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="company_docs", embedding_function=minilm_ef, metadata={"hnsw:space": "cosine"})

# We need to ensure the DB is populated if it's a fresh Client.
# In a real scenario, we'd load the pre-computed index.
def populate_db():
    docs_dir = os.path.join(os.path.dirname(__file__), "docs")
    if not os.path.exists(docs_dir):
        print(f"Error: {docs_dir} directory not found.")
        return

    ids = []
    texts = []

    for filename in os.listdir(docs_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(docs_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Use filename without extension as ID (e.g., 'doc_01')
                doc_id = os.path.splitext(filename)[0]
                ids.append(doc_id)
                texts.append(content)

    if ids:
        collection.add(documents=texts, ids=ids)
        print(f"Populated DB with {len(ids)} documents.")
    else:
        print("No .txt files found in docs directory.")

populate_db()

# --- LangGraph Nodes ---

def classify_intent(state: SimpleState):
    query = state["question"].lower()
    keywords = ["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"]
    if any(kw in query for kw in keywords):
        topic = "policy_question"
    else:
        topic = "general_question"
    return {"topic": topic}

def retrieve_and_answer(state: SimpleState):
    question = state["question"]
    results = collection.query(query_texts=[question], n_results=3)
    chunks = results["documents"][0]
    ids = results["ids"][0]

    top_snippet = chunks[0] if chunks else "No context found."

    if MOCK_LLM:
        answer = f"Based on the retrieved context: {top_snippet}"
        confidence = 1.0
        return {"answer": answer, "sources": ids, "confidence": confidence}
    else:
        # Real LLM Path with Retry Logic
        context_text = "\n".join([f"{id}: {text}" for id, text in zip(ids, chunks)])
        prompt = PROMPT_TEMPLATE.format(context=context_text, query=question)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Simulate LLM call: in a real app, this is where the API call happens
                # We simulate a failed validation on the first attempt to demonstrate the retry logic
                if attempt == 0:
                    raw_response = '{"answer": "Invalid JSON", "sources": "should be a list"}'
                else:
                    raw_response = f'{{"answer": "The answer is based on {top_snippet}", "sources": {ids}, "confidence": 0.9}}'

                # Validate using the Pydantic model
                validated = ResponseModel.model_validate_json(raw_response)
                return {"answer": validated.answer, "sources": validated.sources, "confidence": validated.confidence}

            except Exception as e:
                if attempt < max_retries - 1:
                    # Generate corrective instruction for the next attempt
                    correction = f"Your previous response failed validation: {str(e)}. Please ensure the output is valid JSON matching the schema."
                    # In a real app, the 'correction' would be appended to the prompt for the next LLM call
                    continue
                else:
                    return {"answer": "ERROR: I encountered an error generating the response. Please try again.", "sources": [], "confidence": 0.0}

def direct_answer(state: SimpleState):
    answer = "I can only answer questions about Zepto policies right now."
    return {"answer": answer, "sources": [], "confidence": 1.0}

def route_by_topic(state: SimpleState):
    return state["topic"]

# --- Graph Construction ---
workflow = StateGraph(SimpleState)
workflow.add_node("classify_intent", classify_intent)
workflow.add_node("retrieve_and_answer", retrieve_and_answer)
workflow.add_node("direct_answer", direct_answer)

workflow.add_edge(START, "classify_intent")
workflow.add_conditional_edges(
    "classify_intent",
    route_by_topic,
    {
        "policy_question": "retrieve_and_answer",
        "general_question": "direct_answer"
    }
)
workflow.add_edge("retrieve_and_answer", END)
workflow.add_edge("direct_answer", END)

app_graph = workflow.compile()

# --- FastAPI App ---
app = FastAPI(title="Zepto Support Assistant")

@app.post("/ask", response_model=ResponseModel)
def ask(request: QueryRequest):
    input_state = {"question": request.query}
    result = app_graph.invoke(input_state)

    return ResponseModel(
        answer=result["answer"],
        sources=result.get("sources", []),
        confidence=result.get("confidence", 1.0)
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
