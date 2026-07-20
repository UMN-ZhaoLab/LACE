import unittest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.prompts.hdl_writer import arithmetic_system_prompt, interface_system_prompt
from src.state_types import WorkflowState
from src.writers import _parse_discovery_evidence, insn_model_writer, parse_model_response
from src.interactive_engine import merge_interface_result


class TestWriters(unittest.TestCase):
    def test_discovery_rejects_invented_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "core.sv").write_text("module core; wire real_signal; endmodule\n")
            payload = '{"decode":{"file":"core.sv","lines":[1,1],"signals":["invented"],"excerpt":"module core; wire real_signal; endmodule"}}'
            evidence, error = _parse_discovery_evidence(payload, root, {"decode"})
            self.assertIsNone(evidence)
            self.assertIn("not present", error or "")

    def test_file_scoped_patch_updates_evidence_selected_child(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "top.sv").write_text("module top; child u(); endmodule\n")
            (root / "child.sv").write_text("module child; wire old_value; endmodule\n")
            state = WorkflowState(
                workspace_dir=str(root), cpu_dir=str(root), cpu_top_file="top.sv",
            )
            evidence = {
                "decode": {
                    "file": "child.sv", "lines": [1, 1],
                    "signals": ["old_value"],
                    "excerpt": "module child; wire old_value; endmodule",
                }
            }
            patch_text = """FILE: child.sv
------- SEARCH
module child; wire old_value; endmodule
------- REPLACE
module child; wire new_value; endmodule
------- END
"""
            result = merge_interface_result(state, patch_text, evidence=evidence)
            self.assertIn("new_value", (root / "child.sv").read_text())
            self.assertEqual(result.integration_evidence, evidence)

    def test_parse_write_new_file(self) -> None:
        content = "<content>\nmodule foo;\nendmodule\n</content>"
        result = parse_model_response(content, original=None)
        self.assertEqual(result, "module foo;\nendmodule")

    def test_interface_prompt_requires_native_path(self) -> None:
        text = interface_system_prompt.lower()
        self.assertIn("reuse", text)
        self.assertIn("existing register", text)
        self.assertIn("rvfi overlap", text)
        self.assertIn("encoding collision", text)
        self.assertIn("source-backed rtl evidence", text)

    def test_arithmetic_prompt_requires_combinational_and_edge_cases(self) -> None:
        text = arithmetic_system_prompt.lower()
        self.assertIn("combinational", text)
        self.assertIn("edge case", text)
        self.assertIn("undefined behavior", text)
        self.assertIn("do not pipeline", text)
        self.assertIn("module level", text)
        self.assertIn("outside any always block", text)

    def test_parse_search_replace(self) -> None:
        content = "------- SEARCH\nold\n=======\nnew\n+++++++ REPLACE"
        original = "old text"
        result = parse_model_response(content, original=original)
        self.assertEqual(result, "new text")

    def test_parse_raw_with_fences(self) -> None:
        content = "```verilog\nmodule bar;\n```"
        result = parse_model_response(content, original=None)
        self.assertEqual(result, "module bar;")

    def test_parse_replace_requires_original(self) -> None:
        content = "------- SEARCH\nx\n=======\ny\n+++++++ REPLACE"
        with self.assertRaises(ValueError) as ctx:
            parse_model_response(content, original=None)
        self.assertIn("Original content is required", str(ctx.exception))

    def test_parse_diff_tag_requires_original(self) -> None:
        content = "<diff>\n--- a\n+++ b\n@@ -1 +1 @@\n-old\n+new\n</diff>"
        with self.assertRaises(ValueError) as ctx:
            parse_model_response(content, original=None)
        self.assertIn("Original content is required", str(ctx.exception))

    @patch("src.writers.prepare_riscv_formal_sandbox")
    @patch("src.writers.get_chat_model")
    def test_insn_model_writer_writes_only_to_run_sandbox(
        self, mock_get_model: MagicMock, mock_prepare: MagicMock
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sandbox = root / "run" / "formal"
            (sandbox / "insns").mkdir(parents=True)
            shared = root / "shared-riscv-formal"
            (shared / "insns").mkdir(parents=True)
            mock_prepare.return_value = sandbox
            model = MagicMock()
            model.invoke.return_value = MagicMock(
                content="module rvfi_insn_lace_test(input wire valid); endmodule"
            )
            mock_get_model.return_value = model
            state = WorkflowState(
                spec="custom test instruction",
                cpu_name="picorv32",
                run_id="run-a",
                workspace_dir=str(root / "run" / "workspace"),
            )

            result = insn_model_writer(state)

            self.assertEqual(result.custom_insn_names, ["lace_test"])
            self.assertTrue((sandbox / "insns" / "insn_lace_test.v").exists())
            self.assertFalse((shared / "insns" / "insn_lace_test.v").exists())
            mock_prepare.assert_called_once_with(
                run_id="run-a",
                cpu_name="picorv32",
                workspace_dir=str(root / "run" / "workspace"),
            )

    @patch("src.writers.prepare_riscv_formal_sandbox")
    @patch("src.writers.get_chat_model")
    def test_insn_model_writer_reports_sandbox_failure(
        self, mock_get_model: MagicMock, mock_prepare: MagicMock
    ) -> None:
        mock_prepare.side_effect = FileNotFoundError("missing formal source")
        model = MagicMock()
        model.invoke.return_value = MagicMock(
            content="module rvfi_insn_test(input wire valid); endmodule"
        )
        mock_get_model.return_value = model
        state = WorkflowState(
            spec="custom test instruction",
            cpu_name="picorv32",
            run_id="run-a",
            workspace_dir="/tmp/run-a/workspace",
        )

        result = insn_model_writer(state)

        self.assertTrue(result.needs_review)
        self.assertTrue(result.formal_terminal)
        self.assertIn("sandbox setup failed", result.last_error)


if __name__ == "__main__":
    unittest.main()
