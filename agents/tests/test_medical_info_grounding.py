import json
import os
from pathlib import Path
import unittest

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from agents.src.agent import medical_info

CACHE_TEST_ROOT = Path("agents/data/cache_test_tmp")


class MedicalInfoGroundingTests(unittest.TestCase):
    def test_current_query_detection(self):
        self.assertTrue(medical_info._is_current_or_outbreak_query("latest dengue outbreak in Colombo"))
        self.assertTrue(medical_info._is_current_or_outbreak_query("Current COVID advisory"))
        self.assertFalse(medical_info._is_current_or_outbreak_query("what are dengue symptoms"))

    def test_tool_selector_routes_current_query_without_llm(self):
        command = medical_info.tool_selector({
            "messages": [HumanMessage(content="latest dengue outbreak in Colombo")],
            "session_id": "test-session",
        })

        self.assertEqual(command.goto, "web_search")
        tool_call = command.update["messages"][0].tool_calls[0]
        self.assertEqual(tool_call["name"], medical_info.web_search_tool.name)
        self.assertEqual(tool_call["args"]["query"], "latest dengue outbreak in Colombo")

    def test_tool_selector_routes_rag_query_to_retrieval_without_llm(self):
        command = medical_info.tool_selector({
            "messages": [HumanMessage(content="tell me about dengue symptoms")],
            "rag_query": "general dengue symptoms and prevention",
            "session_id": "test-session",
        })

        self.assertEqual(command.goto, "retrieve")
        tool_call = command.update["messages"][0].tool_calls[0]
        self.assertEqual(tool_call["name"], medical_info.retriever_tool.name)
        self.assertEqual(tool_call["args"]["query"], "general dengue symptoms and prevention")

    def test_extract_context_items_dedupes_and_caps_documents(self):
        payload = {
            "documents": [
                {"content": "Dengue fever guidance", "source_reference": "MOH", "chunk_index": 1},
                {"content": "Dengue   fever guidance", "source_reference": "MOH", "chunk_index": 1},
                {"content": "", "source_reference": "MOH", "chunk_index": 2},
                {"content": "Warning signs", "source_reference": "EPID", "chunk_index": 3},
                {"content": "Prevention", "source_reference": "EPID", "chunk_index": 4},
                {"content": "Care seeking", "source_reference": "EPID", "chunk_index": 5},
                {"content": "Extra item", "source_reference": "EPID", "chunk_index": 6},
                {"content": "Beyond cap", "source_reference": "EPID", "chunk_index": 7},
            ]
        }

        kind, items, error = medical_info._extract_context_items(json.dumps(payload))

        self.assertEqual(kind, "documents")
        self.assertIsNone(error)
        self.assertEqual(len(items), medical_info.MAX_CONTEXT_ITEMS)
        self.assertEqual(items[0]["content"], "Dengue fever guidance")
        self.assertEqual(items[1]["content"], "Warning signs")

    def test_extract_context_items_handles_empty_and_malformed_payloads(self):
        kind, items, _ = medical_info._extract_context_items(json.dumps({"documents": []}))
        self.assertEqual(kind, "documents")
        self.assertEqual(items, [])

        kind, items, error = medical_info._extract_context_items("not json")
        self.assertEqual(kind, "malformed")
        self.assertEqual(items, [])
        self.assertIn("valid JSON", error)

    def test_score_document_empty_documents_routes_to_no_context(self):
        command = medical_info.score_document({
            "messages": [HumanMessage(content=json.dumps({"documents": []}))],
            "session_id": "test-session",
        })

        self.assertEqual(command.goto, "no_context_response")
        self.assertIsNone(command.update["grounding_context"])
        self.assertIsNone(command.update["grounding_sources"])

    def test_grounding_context_and_sources_share_numbering(self):
        items = [
            {
                "content": "Use mosquito protection.",
                "source_reference": "https://epid.gov.lk/dengue",
                "title": "Dengue",
                "url": "https://epid.gov.lk/dengue",
                "chunk_index": 10,
                "score": 0.9,
            },
            {
                "content": "Seek care for warning signs.",
                "source_reference": "MOH Guideline",
                "chunk_index": 11,
            },
        ]

        context = medical_info._format_grounding_context(items)
        sources = medical_info._build_source_payload(items)

        self.assertIn("[1] Source: https://epid.gov.lk/dengue", context)
        self.assertIn("[2] Source: MOH Guideline", context)
        self.assertEqual([source["id"] for source in sources], [1, 2])
        self.assertEqual(sources[0]["url"], "https://epid.gov.lk/dengue")

    def test_no_context_fallback_is_localized(self):
        self.assertIn("could not find enough reliable", medical_info._no_context_message("en"))
        self.assertIn("විශ්වාසදායක", medical_info._no_context_message("si"))
        self.assertIn("நம்பகமான", medical_info._no_context_message("ta"))

    def test_chunk_cache_round_trip_rebuilds_dense_and_sparse_documents(self):
        dense_documents = [
            Document(
                page_content="Dengue guidance chunk",
                metadata={"chunk_index": 42, "source_reference": "MOH", "source": "dense"},
            ),
            Document(
                page_content="Warning signs chunk",
                metadata={"chunk_index": 43, "source_reference": "EPID", "source": "dense"},
            ),
        ]

        CACHE_TEST_ROOT.mkdir(parents=True, exist_ok=True)
        cache_path = CACHE_TEST_ROOT / "round_trip_cache.jsonl.gz"
        try:
            medical_info._save_chunk_cache(dense_documents, cache_path)
            loaded = medical_info._load_chunk_cache(cache_path)
        finally:
            if cache_path.exists():
                cache_path.unlink()

        self.assertIsNotNone(loaded)
        loaded_dense, loaded_sparse = loaded
        self.assertEqual([doc.page_content for doc in loaded_dense], [
            "Dengue guidance chunk",
            "Warning signs chunk",
        ])
        self.assertEqual([doc.metadata["source"] for doc in loaded_dense], ["dense", "dense"])
        self.assertEqual([doc.metadata["source"] for doc in loaded_sparse], ["sparse", "sparse"])
        self.assertEqual(loaded_sparse[0].metadata["source_reference"], "MOH")

    def test_chunk_cache_metadata_mismatch_returns_none(self):
        dense_documents = [
            Document(
                page_content="Dengue guidance chunk",
                metadata={"chunk_index": 42, "source_reference": "MOH", "source": "dense"},
            )
        ]

        CACHE_TEST_ROOT.mkdir(parents=True, exist_ok=True)
        cache_path = CACHE_TEST_ROOT / "metadata_cache.jsonl.gz"
        original_version = medical_info.CHUNK_CACHE_VERSION
        try:
            medical_info._save_chunk_cache(dense_documents, cache_path)
            medical_info.CHUNK_CACHE_VERSION = original_version + 1
            loaded = medical_info._load_chunk_cache(cache_path)
        finally:
            medical_info.CHUNK_CACHE_VERSION = original_version
            if cache_path.exists():
                cache_path.unlink()

        self.assertIsNone(loaded)


if __name__ == "__main__":
    unittest.main()
