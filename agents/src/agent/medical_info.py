"""Medical information agent with RAG retrieval and critique loop."""
import os
import uuid
import chromadb
import sys
import json
from functools import lru_cache
from typing import Any, Dict, List
from typing_extensions import Literal
from langgraph.types import Command
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langsmith import traceable
from pydantic import BaseModel, Field
from agents.src.config import (
    LLM_MODEL, LLM_TEMPERATURE,
    LLM_PROVIDER,
    OPENAI_BASE_URL,
    CUSTOM_LLM_API_KEY,
    CUSTOM_LLM_BASE_URL,
    CUSTOM_LLM_MODEL,
    CUSTOM_LLM_MAX_TOKENS,
    CUSTOM_LLM_TEMPERATURE,
)

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

AGENTS_ROOT = Path(__file__).resolve().parents[2]
if str(AGENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENTS_ROOT))

from agents.src.prompts.medi_info import (
    score_document_prompt, rewrite_prompt, generate_prompt, 
    tool_selection_prompt, critique_prompt
)
from agents.src.config import (
    LLM_MODEL, LLM_TEMPERATURE, CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME,
    RETRIEVAL_TOP_K, ENSEMBLE_DENSE_WEIGHT, MAX_CRITIQUE_ATTEMPTS, MAX_REWRITE_ATTEMPTS,
    TAVILY_INCLUDE_DOMAINS, TAVILY_MAX_RESULTS, MEDICAL_DISCLAIMER
)
from tavily import TavilyClient
from agents.src.utils import (
    setup_logger,
    retry_on_error,
    create_error_response,
    detect_user_language,
    get_language_instruction,
    get_medical_disclaimer,
)
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
load_dotenv()
from agents.src.graph.state import AgentState, AgentInputState
from agents.src.supabase_loader import load_documents_from_supabase

logger = setup_logger("medical_info_agent")

def _build_chunked_documents(all_documents: List[Document]) -> tuple[List[Document], List[Document]]:
    """Split source documents into dense/sparse chunk lists while preserving metadata."""
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=200,
        chunk_overlap=50,
    )

    dense_documents: List[Document] = []
    sparse_documents: List[Document] = []
    chunk_counter = 0

    for doc in all_documents:
        doc_splits = character_splitter.split_text(doc.page_content)
        source_reference = doc.metadata.get("source_reference", "Unknown")

        for split_text in doc_splits:
            metadata = {
                "id": str(chunk_counter),
                "chunk_index": chunk_counter,
                "source_reference": source_reference,
            }
            dense_documents.append(
                Document(page_content=split_text, metadata={**metadata, "source": "dense"})
            )
            sparse_documents.append(
                Document(page_content=split_text, metadata={**metadata, "source": "sparse"})
            )
            chunk_counter += 1

    logger.info(f"Created {len(dense_documents)} document chunks with preserved metadata")
    return dense_documents, sparse_documents


@lru_cache(maxsize=1)
def get_ensemble_retriever() -> EnsembleRetriever | None:
    """Initialize the retrieval stack lazily; return None if corpus setup is unavailable."""
    try:
        all_documents = load_documents_from_supabase()
        logger.info(f"Total documents loaded from Supabase: {len(all_documents)}")
    except Exception as exc:
        logger.exception(f"Supabase load failed: {exc}")
        return None

    if not all_documents:
        logger.warning("No corpus documents available from Supabase; retrieval will be disabled")
        return None

    dense_documents, sparse_documents = _build_chunked_documents(all_documents)

    try:
        embedding_function = OpenAIEmbeddings()
        logger.info(f"Initializing ChromaDB with persist directory: {CHROMA_PERSIST_DIR}")
        chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

        try:
            col = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)
            if col.count() > 0:
                logger.info(
                    f"Reusing cached ChromaDB collection '{CHROMA_COLLECTION_NAME}' "
                    f"({col.count()} docs) — skipping re-embed"
                )
                vectorstore = Chroma(
                    client=chroma_client,
                    collection_name=CHROMA_COLLECTION_NAME,
                    embedding_function=embedding_function,
                )
            else:
                raise ValueError("Collection exists but is empty — rebuilding")
        except Exception as cache_miss:
            logger.info(
                f"ChromaDB cache miss ({cache_miss}). Building collection from {len(dense_documents)} chunks ..."
            )
            vectorstore = Chroma.from_documents(
                documents=dense_documents,
                embedding=embedding_function,
                collection_name=CHROMA_COLLECTION_NAME,
                client=chroma_client,
            )
            logger.info(f"ChromaDB collection built with {vectorstore._collection.count()} documents")

        dense_retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_TOP_K})
        sparse_retriever = BM25Retriever.from_documents(sparse_documents, k=RETRIEVAL_TOP_K)
        sparse_weight = 1.0 - ENSEMBLE_DENSE_WEIGHT
        return EnsembleRetriever(
            retrievers=[dense_retriever, sparse_retriever],
            weights=[ENSEMBLE_DENSE_WEIGHT, sparse_weight],
            c=60,
        )
    except Exception as exc:
        logger.exception(f"Retriever initialization failed: {exc}")
        return None

# ===== RETRIEVAL TOOL ===== 

@tool("retrieve_medical_info")
def retrieve_medical_info(query: str) -> str:
    """
    Retrieve medical information related to the provided query using the ensemble retriever.
    Args:
        query (str): Natural-language question or keyword string describing the medical information to fetch.
    Returns:
        str: Concatenated textual content of all retrieved documents separated by blank lines.
    """

    try:
        ensemble_retriever = get_ensemble_retriever()
        if ensemble_retriever is None:
            logger.warning("Retrieval requested but no retriever is available")
            return json.dumps({
                "documents": [],
                "error": "Medical knowledge base is unavailable right now."
            })

        docs = ensemble_retriever.invoke(query)
        logger.debug(f"Query passed to tool: {query}")
        logger.info(f"Retrieved {len(docs)} documents")
        for idx, doc in enumerate(docs, start=1):
            snippet = doc.page_content.strip().replace("\n", " ")[:200]
            source_reference = doc.metadata.get("source_reference", "Unknown")
            chunk_id = doc.metadata.get("chunk_index", "?")

            logger.info(
                "Retrieved chunk %02d\n  Source : %s\n  Chunk  : %s\n  Preview: %s",
                idx,
                source_reference,
                chunk_id,
                snippet,
            )
        
        result = {
            "documents": [
                {
                    "content": doc.page_content,
                    "source_reference": doc.metadata.get("source_reference", "Unknown"),
                    "chunk_index": doc.metadata.get("chunk_index", 0)
                }
                for doc in docs
            ]
        }
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Retrieval error: {e}") 
        return "Unable to retrieve information at this time."

retriever_tool = retrieve_medical_info

# ===== WEB SEARCH TOOL =====
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool("web_search")
def web_search(query: str) -> str:
    """
    Search for medical information from trusted Sri Lanka health websites.
    Uses Tavily to search only within approved domains (epid.gov.lk, health.gov.lk).
    
    Args:
        query (str): The search query for medical/health information.
    
    Returns:
        str: JSON-formatted search results with content and URLs.
    """
    import json
    
    try:
        logger.info(f"Web search query: {query}")
        logger.info(f"Searching domains: {TAVILY_INCLUDE_DOMAINS}")
        
        response = tavily_client.search(
            query=query,
            include_domains=TAVILY_INCLUDE_DOMAINS,
            max_results=TAVILY_MAX_RESULTS,
            search_depth="advanced"
        )
        
        results = []
        for item in response.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "content": item.get("content", ""),
                "url": item.get("url", ""),
                "score": item.get("score", 0)
            })
        
        logger.info(f"Web search returned {len(results)} results")
        for idx, result in enumerate(results, 1):
            logger.info(f"Result {idx}: {result['title'][:50]}... from {result['url']}")
        
        return json.dumps({"web_results": results})
        
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return json.dumps({"error": str(e), "web_results": []})

web_search_tool = web_search

# OpenAI path kept for later use
# response_model = ChatOpenAI(...)

if LLM_PROVIDER == "custom_openai_compatible":
    response_model = ChatOpenAI(
        api_key=CUSTOM_LLM_API_KEY,
        base_url=CUSTOM_LLM_BASE_URL,
        model=CUSTOM_LLM_MODEL,
        temperature=CUSTOM_LLM_TEMPERATURE,
        max_tokens=CUSTOM_LLM_MAX_TOKENS,
    )
else:
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    openai_kwargs = {
        "api_key": openai_api_key,
        "model": LLM_MODEL,
        "temperature": LLM_TEMPERATURE,
    }
    if OPENAI_BASE_URL:
        openai_kwargs["base_url"] = OPENAI_BASE_URL
    response_model = ChatOpenAI(**openai_kwargs)

logger.info(
    "Medical model initialized | provider=%s | model=%s | has_api_key=%s",
    LLM_PROVIDER,
    (CUSTOM_LLM_MODEL if LLM_PROVIDER == "custom_openai_compatible" else LLM_MODEL),
    bool(CUSTOM_LLM_API_KEY if LLM_PROVIDER == "custom_openai_compatible" else openai_api_key),
)


def _latest_user_text(messages: List[BaseMessage]) -> str:
    """Return the most recent human utterance to use as a fallback query."""
    for msg in reversed(messages or []):
        if isinstance(msg, HumanMessage):
            return msg.content
    return messages[-1].content if messages else ""


# ===== STRUCTURED OUTPUT SCHEMAS =====
class CritiqueResult(BaseModel):
    """Schema for critique evaluation."""
    needs_refinement: bool = Field(description="Whether the answer needs refinement")
    issues: List[str] = Field(default=[], description="List of specific issues found")
    feedback: str = Field(default="", description="Constructive feedback for improvement")


# ===== HELPER FUNCTIONS =====
def _classify_query_intent(query: str) -> str:
    """
    Classify query intent to help guide tool selection.
    
    Args:
        query: The user's query string
        
    Returns:
        'current' if query asks about recent/current info (prefer web_search)
        'general' for standard medical information (prefer RAG retrieval)
    """
    current_indicators = [
        "recent", "latest", "current", "today", "now", 
        "outbreak", "epidemic", "this week", "this month"
    ]
    query_lower = query.lower()
    
    if any(indicator in query_lower for indicator in current_indicators):
        logger.debug(f"Query classified as 'current' - prefer web search")
        return "current"
    
    logger.debug(f"Query classified as 'general' - prefer RAG retrieval")
    return "general"


# ===== GRAPH NODES =====
@traceable(name="tool_selector")
def tool_selector(state: AgentState) -> Command[Literal["retrieve", "web_search", "__end__"]]:
    """Let the model choose between retrieval tool, web search, or directly answer the user."""
    
    try:
        system_prompt = SystemMessage(content=tool_selection_prompt)
        rag_query = state.get("rag_query")
        
        if rag_query:
            logger.info("Using cached RAG query for retrieval after intent classification or rewrite")
            response = AIMessage(
                content="",
                tool_calls=[{
                    "name": retriever_tool.name, 
                    "args": {"query": rag_query},
                    "id": f"call_{uuid.uuid4().hex}",
                }],
                additional_kwargs={"source": "intent_classifier_query"},
            )
            return Command(
                goto="retrieve",
                update={"messages": [response]},
            )

        query_text = _latest_user_text(state["messages"])
        user_message = HumanMessage(content=query_text)
        conversation = [system_prompt, user_message]

        @retry_on_error(logger=logger)
        def invoke_model():
            return (
                response_model
                .bind_tools([retriever_tool, web_search_tool])
                .invoke(conversation)
            )
        
        response = invoke_model()

        if response.tool_calls:
            tool_name = response.tool_calls[0]["name"]
            logger.info(f"Tool selected: {tool_name}")
            
            if tool_name == web_search_tool.name:
                return Command(
                    goto="web_search",
                    update={"messages": [response]},
                )
            else:
                return Command(
                    goto="retrieve",
                    update={"messages": [response]},
                )

        return Command(
            goto=END,
            update={"messages": [response]},
        )
        
    except Exception as e:
        logger.error(f"Error in generate_query_or_respond: {e}")
        return Command(
            goto=END,
            update={"messages": [AIMessage(content=create_error_response("llm", language=detect_user_language(_latest_user_text(state.get("messages", [])))))]},
        )


@traceable(name="score_document")
def score_document(state: AgentState) -> Command[Literal["generate_answer", "improve"]]:
    """Score document relevance (handles both RAG and web search results) and decide whether to improve the query."""
    
    try:
        import json
        class Scoring(BaseModel):
            binary_score: str = Field(description="relevance score 'yes' or 'no'")
        
        structured_output_model = response_model.with_structured_output(Scoring)

        raw_content = state["messages"][-1].content
        
        # Parse JSON to extract content from both RAG and web search formats
        try:
            data = json.loads(raw_content)
            
            # Handle RAG documents
            if "documents" in data:
                latest_context = "\n\n".join([doc["content"] for doc in data["documents"]])
                logger.info("Scoring RAG documents")
            
            # Handle web search results
            elif "web_results" in data:
                latest_context = "\n\n".join([res["content"] for res in data["web_results"]])
                logger.info("Scoring web search results")
            
            else:
                latest_context = raw_content
        except (json.JSONDecodeError, KeyError):
            latest_context = raw_content
            
        original_question = state.get("rag_query") or _latest_user_text(state["messages"])
        
        @retry_on_error(logger=logger)
        def invoke_model():
            return structured_output_model.invoke([
                HumanMessage(content=score_document_prompt.format(
                    context=latest_context,
                    question=original_question
                ))
            ])
        
        response = invoke_model()
        score = response.binary_score
        logger.info(f"Document relevance score: {score}")

        if score == 'yes':
            # Reset rewrite attempts so future turns can start fresh
            if state.get("rewrite_attempts"):
                return Command(goto="generate_answer", update={"rewrite_attempts": 0})
            return Command(goto="generate_answer")
        else:
            # Check rewrite limit
            rewrite_attempts = state.get("rewrite_attempts", 0)
            if rewrite_attempts >= MAX_REWRITE_ATTEMPTS:
                logger.warning(f"Max rewrite attempts ({MAX_REWRITE_ATTEMPTS}) reached, proceeding to generate")
                return Command(goto="generate_answer")
            return Command(goto="improve", update={"rewrite_attempts": rewrite_attempts + 1})
            
    except Exception as e:      
        logger.error(f"Error in score_document: {e}")
        return Command(goto="generate_answer")


@traceable(name="improve")
def improve(state: AgentState):
    """Rewrite the original user question."""
    try:
        question = state.get("rag_query") or _latest_user_text(state.get("messages", []))
        prompt = rewrite_prompt.format(question=question)
        
        @retry_on_error(logger=logger)
        def invoke_model():
            return response_model.invoke([{"role": "user", "content": prompt}])
        
        response = invoke_model()
        rewritten_query = (response.content or question or "").strip()
        logger.info("Query improved/rewritten")
        updates = {"messages": [AIMessage(content=rewritten_query)]}
        if rewritten_query:
            updates["rag_query"] = rewritten_query
        return updates
        
    except Exception as e:
        logger.error(f"Error in improve: {e}")
        return {"messages": [AIMessage(content=state.get("rag_query", ""))]}

@traceable(name="generate_answer")
def generate_answer(state: AgentState):
    """Generate an answer based on retrieved context."""
    import json
    
    try:
        latest_user_text = _latest_user_text(state["messages"])
        response_language = detect_user_language(latest_user_text)
        question = state.get("rag_query") or latest_user_text
        
        # Use critique feedback if available
        critique_feedback = state.get("critique_feedback")
        raw_content = state["messages"][-1].content
        
        # Parse JSON to extract context and citation metadata (handles both RAG and web search)
        citations_payload = []
        try:
            data = json.loads(raw_content)
            
            # Handle RAG documents
            if "documents" in data:
                context = "\n\n".join([doc["content"] for doc in data["documents"]])
                for doc in data["documents"]:
                    citations_payload.append(
                        {
                            "source_reference": doc.get("source_reference", "Unknown Source"),
                            "chunk_index": doc.get("chunk_index"),
                            "excerpt": (doc.get("content", "") or "")[:200].strip(),
                        }
                    )
            
            # Handle web search results
            elif "web_results" in data:
                context = "\n\n".join([res["content"] for res in data["web_results"]])
                for idx, res in enumerate(data["web_results"], 1):
                    citations_payload.append(
                        {
                            "source_reference": res.get("url", "Unknown Source"),
                            "title": res.get("title", ""),
                            "excerpt": (res.get("content", "") or "")[:200].strip(),
                            "score": res.get("score", 0)
                        }
                    )
            
            else:
                context = raw_content
        except (json.JSONDecodeError, KeyError):
            context = raw_content
        
        if critique_feedback:
            prompt = f"{generate_prompt.format(question=question, context=context, response_language_instruction=get_language_instruction(response_language))}\n\nPREVIOUS FEEDBACK TO ADDRESS:\n{critique_feedback}"
            logger.info("Generating answer with critique feedback")
        else:
            prompt = generate_prompt.format(
                question=question,
                context=context,
                response_language_instruction=get_language_instruction(response_language),
            )
        
        @retry_on_error(logger=logger)
        def invoke_model():
            return response_model.invoke([{"role": "user", "content": prompt}])
        
        response = invoke_model()
        logger.info("Answer generated")
        
        # Add medical disclaimer to response
        response.content = (response.content or "") + get_medical_disclaimer(response_language, MEDICAL_DISCLAIMER)
        
        if citations_payload:
            existing_kwargs = response.additional_kwargs or {}
            existing_kwargs["sources"] = [
                {
                    "id": idx,
                    "source": citation.get("source_reference", "Unknown Source"),
                    "source_reference": citation.get("source_reference", "Unknown Source"),
                    "chunk_index": citation.get("chunk_index"),
                    "title": citation.get("title", ""),
                    "excerpt": citation.get("excerpt", ""),
                    "score": citation.get("score")
                }
                for idx, citation in enumerate(citations_payload, start=1)
            ]
            response.additional_kwargs = existing_kwargs
        return {"messages": [response], "critique_feedback": None}
        
    except Exception as e:
        logger.error(f"Error in generate_answer: {e}")
        return {"messages": [AIMessage(content=create_error_response("llm", language=detect_user_language(_latest_user_text(state.get("messages", [])))))]}


@traceable(name="critique_answer")
def critique_answer(state: AgentState) -> Command[Literal["generate_answer", "__end__"]]:
    """Critique the generated answer and decide if refinement is needed."""
    
    try:
        critique_attempts = state.get("critique_attempts", 0)
        
        # Check if max critique iterations reached
        if critique_attempts >= MAX_CRITIQUE_ATTEMPTS:
            logger.info(f"Max critique attempts ({MAX_CRITIQUE_ATTEMPTS}) reached, finalizing answer")
            return Command(goto=END)
        
        question = state.get("rag_query") or _latest_user_text(state["messages"])
        answer = state["messages"][-1].content
        
        structured_output_model = response_model.with_structured_output(CritiqueResult)
        
        @retry_on_error(logger=logger)
        def invoke_model():
            return structured_output_model.invoke([
                HumanMessage(content=critique_prompt.format(
                    question=question,
                    answer=answer
                ))
            ])
        
        response = invoke_model()
        logger.info(f"Critique result - needs refinement: {response.needs_refinement}")
        
        if response.needs_refinement and response.feedback:
            logger.info(f"Critique issues: {response.issues}")
            return Command(
                goto="generate_answer",
                update={
                    "critique_attempts": critique_attempts + 1,
                    "critique_feedback": response.feedback
                }
            )
        
        logger.info("Answer passed critique, finalizing")
        return Command(goto=END)
        
    except Exception as e:
        logger.error(f"Error in critique_answer: {e}")
        return Command(goto=END)


# ===== ASSEMBLE THE GRAPH =====
workflow = StateGraph(AgentState, input_schema=AgentInputState)

# Define the nodes
workflow.add_node("tool_selector", tool_selector)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node("web_search", ToolNode([web_search_tool]))
workflow.add_node("score_document", score_document)
workflow.add_node("improve", improve)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("critique_answer", critique_answer)

# Define workflow edges
workflow.add_edge(START, "tool_selector")
workflow.add_edge("retrieve", "score_document")
workflow.add_edge("web_search", "score_document")  # Web search also goes through scoring
workflow.add_edge("improve", "tool_selector")  # Retry with improved query
workflow.add_edge("generate_answer", "critique_answer")
       
medical_info_graph = workflow.compile()

# Guard graph visualization
if __name__ == "__main__":
    output_path = Path("medical_info_graph.png")
    medical_info_graph.get_graph().draw_mermaid_png(output_file_path=output_path)
    logger.info(f"Graph exported to {output_path.resolve()}")
    
    for chunk in medical_info_graph.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "2 hospitals in sri lanka",
                }
            ]
        }
    ):
        for node, update in chunk.items():
            logger.info(f"Update from node: {node}")
            if not update:
                continue

            messages = update.get("messages")
            if not messages:
                continue

            terminal_message = messages[-1]
            if hasattr(terminal_message, "pretty_print"):
                terminal_message.pretty_print()
            else:
                logger.info(terminal_message)
