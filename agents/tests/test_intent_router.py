import json
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from agents.backend.router_logger import RouterLogger
from agents.src.agent import intent_classifier


class IntentRouterTests(unittest.TestCase):
    def test_parse_router_output_ignores_extra_rationale(self):
        payload, error = intent_classifier.parse_router_output(json.dumps({
            "language": "English",
            "intent": "Disease_Information",
            "keywords": ["dengue"],
            "english_translation_or_summary": "Tell me about dengue.",
            "rationale": "Do not store this.",
        }))

        self.assertIsNone(error)
        self.assertNotIn("rationale", payload)

    def test_parse_router_output_extracts_json_after_thinking_wrapper(self):
        payload, error = intent_classifier.parse_router_output("""
<think>

</think>

{
  "language": "English",
  "intent": "Emergency",
  "keywords": ["chest pain"],
  "english_translation_or_summary": "Emergency symptoms."
}
""")

        self.assertIsNone(error)
        self.assertEqual(payload["intent"], "Emergency_Triage")

    def test_router_failure_uses_facility_fallback_instead_of_old_clarifier(self):
        with patch.object(intent_classifier, "_invoke_router", side_effect=RuntimeError("401")):
            result = intent_classifier.classify_intent({
                "messages": [HumanMessage(content="list hospitals near colombo")],
                "session_id": "test-session",
            })

        self.assertEqual(result["active_agent"], "medical_info")
        self.assertEqual(result["route_intent"], "Facility_Locator")
        self.assertIn("list hospitals near colombo", result["rag_query"])

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


if __name__ == "__main__":
    unittest.main()
