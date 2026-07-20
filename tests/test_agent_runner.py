import unittest
from unittest.mock import MagicMock

from src.nodes.agent_runner import invoke_with_backoff


class TestAgentRunner(unittest.TestCase):
    def test_invoke_success_first_try(self) -> None:
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = "ok"
        result = invoke_with_backoff(
            mock_structured, ["msg"], 3, _sleep=lambda x: None
        )
        self.assertEqual(result, "ok")
        self.assertEqual(mock_structured.invoke.call_count, 1)

    def test_invoke_retries_then_succeeds(self) -> None:
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = [Exception("rate limit"), "ok"]
        result = invoke_with_backoff(
            mock_structured, ["msg"], 3, _sleep=lambda x: None
        )
        self.assertEqual(result, "ok")
        self.assertEqual(mock_structured.invoke.call_count, 2)

    def test_invoke_exhausts_all_attempts(self) -> None:
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = Exception("rate limit")
        with self.assertRaises(RuntimeError) as ctx:
            invoke_with_backoff(
                mock_structured, ["msg"], 2, _sleep=lambda x: None
            )
        self.assertIn("rate limit", str(ctx.exception))
        self.assertEqual(mock_structured.invoke.call_count, 2)

    def test_invoke_non_retryable_error_retries_without_sleep(self) -> None:
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = [Exception("bad request"), "ok"]
        result = invoke_with_backoff(
            mock_structured, ["msg"], 3, _sleep=lambda x: None
        )
        self.assertEqual(result, "ok")
        self.assertEqual(mock_structured.invoke.call_count, 2)


if __name__ == "__main__":
    unittest.main()
