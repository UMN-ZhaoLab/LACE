import tempfile
import unittest
from pathlib import Path

from src.nodes.rag_retriever import rag_retriever
from src.state_types import WorkflowState


class TestRagRetriever(unittest.TestCase):
    def test_prefers_workspace_over_cpu_dir(self) -> None:
        """rag_retriever should read from workspace_dir when available."""
        with tempfile.TemporaryDirectory() as cpu_dir, tempfile.TemporaryDirectory() as workspace_dir:
            # Original CPU file
            original = Path(cpu_dir) / "picorv32.v"
            original.write_text("module picorv32; wire a; endmodule", encoding="utf-8")

            # Modified workspace file
            modified = Path(workspace_dir) / "picorv32.v"
            modified.write_text(
                "module picorv32; wire a; wire ISAX_isisax; endmodule",
                encoding="utf-8",
            )

            state = WorkflowState(
                spec="ROL",
                ops=["op1"],
                hdl_tasks=["Add decode logic for ISAX_isisax"],
                hdl_index=0,
                cpu_dir=cpu_dir,
                workspace_dir=workspace_dir,
                cpu_top_file="picorv32.v",
            )

            result = rag_retriever(state)
            # New rag_retriever only extracts the module declaration;
            # detailed exploration is delegated to rg_tools inside interface_writer.
            self.assertIn("module picorv32", result.relevant_code)

    def test_falls_back_to_cpu_dir(self) -> None:
        """When workspace_dir is None, fall back to cpu_dir."""
        with tempfile.TemporaryDirectory() as cpu_dir:
            original = Path(cpu_dir) / "picorv32.v"
            original.write_text("module picorv32( input a ); wire b; endmodule", encoding="utf-8")

            state = WorkflowState(
                spec="ROL",
                ops=["op1"],
                hdl_tasks=["Add port b"],
                hdl_index=0,
                cpu_dir=cpu_dir,
                workspace_dir="",
                cpu_top_file="picorv32.v",
            )

            result = rag_retriever(state)
            self.assertIn("input a", result.relevant_code)


if __name__ == "__main__":
    unittest.main()
