import json
import os
import unittest
from unittest.mock import patch

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain_core.messages import AIMessage, HumanMessage

from agents.backend.router_logger import RouterLogRecord, RouterLogger
from agents.src.agent import intent_classifier


class IntentRouterTests(unittest.TestCase):
    def test_parse_router_output_accepts_valid_contract(self):
        payload, error = intent_classifier.parse_router_output(json.dumps({
            "language": "Sinhala English Code Mixed",
            "intent": "Medication_Information",
            "keywords": ["Panadol", "BP medicine"],
            "english_translation_or_summary": "Can Panadol be taken with blood pressure medicine?",
        }))

        self.assertIsNone(error)
        self.assertEqual(payload["intent"], "Medication_Information")
        self.assertEqual(payload["keywords"], ["Panadol", "BP medicine"])

    def test_parse_router_output_ignores_extra_rationale(self):
        payload, error = intent_classifier.parse_router_output(json.dumps({
            "language": "English",
            "intent": "Disease_Information",
            "keywords": ["dengue"],
            "english_translation_or_summary": "Tell me about dengue.",
            "rationale": "This must not be stored.",
        }))

        self.assertIsNone(error)
        self.assertNotIn("rationale", payload)

    def test_parse_router_output_invalid_json_becomes_unclear(self):
        payload, error = intent_classifier.parse_router_output("not-json")

        self.assertEqual(payload["intent"], "Unclear")
        self.assertIn("invalid_json", error)

    def test_parse_router_output_extracts_json_after_thinking_wrapper(self):
        payload, error = intent_classifier.parse_router_output("""
<think>

</think>

{
  "language": "English",
  "intent": "Emergency_Triage",
  "keywords": ["chest pain"],
  "english_translation_or_summary": "Emergency symptoms."
}
""")

        self.assertIsNone(error)
        self.assertEqual(payload["intent"], "Emergency_Triage")

    def test_parse_router_output_maps_emergency_alias(self):
        payload, error = intent_classifier.parse_router_output(json.dumps({
            "language": "English",
            "intent": "Emergency",
            "keywords": ["chest pain"],
            "english_translation_or_summary": "Emergency symptoms.",
        }))

        self.assertIsNone(error)
        self.assertEqual(payload["intent"], "Emergency_Triage")

    def test_parse_router_output_unsupported_intent_becomes_unclear(self):
        payload, error = intent_classifier.parse_router_output(json.dumps({
            "language": "English",
            "intent": "Unsupported",
            "keywords": [],
            "english_translation_or_summary": "Unknown request",
        }))

        self.assertEqual(payload["intent"], "Unclear")
        self.assertEqual(error, "unsupported_intent")

    def test_emergency_intent_returns_direct_contact_message(self):
        with patch.object(intent_classifier, "_invoke_router", return_value=({
            "language": "English",
            "intent": "Emergency_Triage",
            "keywords": ["emergency"],
            "english_translation_or_summary": "Emergency help",
        }, 10, None)):
            result = intent_classifier.classify_intent({
                "messages": [HumanMessage(content="Emergency help")],
                "session_id": "test-session",
            })

        self.assertEqual(result["active_agent"], "emergency_response")
        self.assertIn("Ambulance Service: 1990", result["messages"][0].content)
        self.assertIsNone(result["rag_query"])

    def test_non_medical_intent_returns_out_of_scope(self):
        with patch.object(intent_classifier, "_invoke_router", return_value=({
            "language": "English",
            "intent": "Non_Medical",
            "keywords": ["weather"],
            "english_translation_or_summary": "Weather question",
        }, 10, None)):
            result = intent_classifier.classify_intent({
                "messages": [HumanMessage(content="What is the weather?")],
                "session_id": "test-session",
            })

        self.assertEqual(result["active_agent"], "non_medical_response")
        self.assertIn("health-related question", result["messages"][0].content)
        self.assertIsNone(result["rag_query"])

    def test_unclear_intent_routes_to_query_classifier(self):
        with patch.object(intent_classifier, "_invoke_router", return_value=({
            "language": "English",
            "intent": "Unclear",
            "keywords": [],
            "english_translation_or_summary": "",
        }, 10, None)):
            result = intent_classifier.classify_intent({
                "messages": [HumanMessage(content="help")],
                "session_id": "test-session",
            })

        self.assertEqual(result["active_agent"], "query_classifier")
        self.assertIsNone(result["rag_query"])

    def test_medical_intent_builds_rag_query(self):
        with patch.object(intent_classifier, "_invoke_router", return_value=({
            "language": "Tamil English Code Mixed",
            "intent": "Treatment_Procedure",
            "keywords": ["physiotherapy", "knee replacement"],
            "english_translation_or_summary": "How will physiotherapy proceed after knee replacement?",
        }, 10, None)):
            result = intent_classifier.classify_intent({
                "messages": [HumanMessage(content="physiotherapy eppadi?")],
                "session_id": "test-session",
            })

        self.assertEqual(result["active_agent"], "medical_info")
        self.assertIn("Treatment_Procedure", result["rag_query"])
        self.assertIn("physiotherapy, knee replacement", result["rag_query"])
        self.assertIn("physiotherapy eppadi?", result["rag_query"])

    def test_router_failure_uses_facility_fallback_instead_of_clarification_loop(self):
        with patch.object(intent_classifier, "_invoke_router", side_effect=RuntimeError("401")):
            result = intent_classifier.classify_intent({
                "messages": [HumanMessage(content="i want details of hospitals in colombo")],
                "session_id": "test-session",
            })

        self.assertEqual(result["active_agent"], "medical_info")
        self.assertEqual(result["route_intent"], "Facility_Locator")
        self.assertIn("hospitals", result["rag_query"])

    def test_router_failure_uses_disease_fallback(self):
        with patch.object(intent_classifier, "_invoke_router", side_effect=RuntimeError("401")):
            result = intent_classifier.classify_intent({
                "messages": [HumanMessage(content="i want to know about common cold")],
                "session_id": "test-session",
            })

        self.assertEqual(result["active_agent"], "medical_info")
        self.assertEqual(result["route_intent"], "Disease_Information")
        self.assertIn("common cold", result["rag_query"])

    def test_router_failure_uses_emergency_fallback(self):
        with patch.object(intent_classifier, "_invoke_router", side_effect=RuntimeError("401")):
            result = intent_classifier.classify_intent({
                "messages": [HumanMessage(content="I have severe chest pain and cannot breathe")],
                "session_id": "test-session",
            })

        self.assertEqual(result["active_agent"], "emergency_response")
        self.assertEqual(result["route_intent"], "Emergency_Triage")
        self.assertIn("Ambulance Service: 1990", result["messages"][0].content)

    def test_router_failure_uses_social_fallback(self):
        with patch.object(intent_classifier, "_invoke_router", side_effect=RuntimeError("401")):
            result = intent_classifier.classify_intent({
                "messages": [HumanMessage(content="hi")],
                "session_id": "test-session",
            })

        self.assertEqual(result["active_agent"], "non_medical_response")
        self.assertEqual(result["route_intent"], "Non_Medical")


class RouterLoggerTests(unittest.TestCase):
    def test_router_logger_sanitizes_rationale(self):
        payload = RouterLogger.sanitize_payload({
            "language": "English",
            "intent": "Disease_Information",
            "keywords": ["dengue"],
            "english_translation_or_summary": "Dengue information",
            "rationale": "Do not store this.",
        })

        self.assertEqual(set(payload.keys()), {
            "language",
            "intent",
            "keywords",
            "english_translation_or_summary",
        })
        self.assertNotIn("rationale", payload)

    def test_router_logger_insert_is_non_blocking_on_failure(self):
        class FailingTable:
            def insert(self, row):
                raise RuntimeError("insert failed")

        class FailingClient:
            def table(self, name):
                return FailingTable()

        logger = RouterLogger.__new__(RouterLogger)
        logger._enabled = True
        logger._supabase = FailingClient()
        logger._supabase_table = "agent_router_logs"
        logger._write_success = 0
        logger._write_fail = 0

        logger.log_router_result(RouterLogRecord(
            session_id="s",
            turn_index=1,
            user_message="hello",
            language="English",
            intent="Non_Medical",
            keywords=[],
            english_translation_or_summary="hello",
            router_model="router",
            router_base_url="http://router",
            router_payload={"intent": "Non_Medical", "rationale": "ignore"},
            created_at="2026-05-05T00:00:00.000Z",
            latency_ms=10,
        ))

        self.assertEqual(logger._write_fail, 1)


class SystemRouterHandoffTests(unittest.TestCase):
    def test_system_emergency_response_goes_directly_to_finalize(self):
        from agents.src.agent import system

        class FakeRouterGraph:
            def invoke(self, state, config=None):
                return {
                    **state,
                    "messages": state["messages"] + [AIMessage(content="Emergency Contact\n\nAmbulance Service: 1990")],
                    "active_agent": "emergency_response",
                    "rag_query": None,
                    "route_intent": "Emergency_Triage",
                }

        with (
            patch.object(system, "intent_classifier_graph", FakeRouterGraph()),
            patch.object(system, "medical_info_graph") as medical_graph,
            patch.object(system, "query_classifier_graph") as query_graph,
        ):
            command = system.intent_classifier_agent({
                "messages": [HumanMessage(content="emergency help")],
                "session_id": "test-session",
            })

        self.assertEqual(command.goto, "finalize_response")
        self.assertEqual(command.update["last_active_agent"], "emergency_response")
        medical_graph.invoke.assert_not_called()
        query_graph.invoke.assert_not_called()

    def test_system_unclear_response_routes_to_query_classifier(self):
        from agents.src.agent import system

        class FakeRouterGraph:
            def invoke(self, state, config=None):
                return {
                    **state,
                    "active_agent": "query_classifier",
                    "rag_query": None,
                    "route_intent": "Unclear",
                }

        with patch.object(system, "intent_classifier_graph", FakeRouterGraph()):
            command = system.intent_classifier_agent({
                "messages": [HumanMessage(content="help")],
                "session_id": "test-session",
            })

        self.assertEqual(command.goto, "query_classifier_agent")
        self.assertEqual(command.update["active_agent"], "query_classifier")


if __name__ == "__main__":
    unittest.main()
