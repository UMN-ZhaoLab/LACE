import json
import tempfile
import unittest
from pathlib import Path

from src.checkpoint import capture_checkpoint, load_checkpoint, persist_failure_bundle
from src.state_types import WorkflowState


class TestCheckpoint(unittest.TestCase):
    def test_capture_checkpoint_includes_full_state(self) -> None:
        state = WorkflowState(
            spec="test",
            ops=["op1"],
            op_index=1,
            cpu_dir="/cpu",
            needs_review=True,
        )
        checkpoint = capture_checkpoint(state, "op_to_hdl_tasks")
        self.assertEqual(checkpoint.stage, "op_to_hdl_tasks")
        self.assertEqual(checkpoint.payload.op_index, 1)
        self.assertIsNotNone(checkpoint.thin_state)
        self.assertEqual(checkpoint.thin_state.get("spec"), "test")

    def test_load_checkpoint_roundtrip(self) -> None:
        state = WorkflowState(
            spec="rotate instruction",
            ops=["RdInstr()", "WrRD(data, rd)"],
            op_index=1,
            hdl_tasks=["task1"],
            cpu_dir="/cpu",
            run_id="20250420-120000",
        )
        checkpoint = capture_checkpoint(state, "op_to_hdl_tasks")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = persist_failure_bundle(
                state, "op_to_hdl_tasks", "error", checkpoint, evidence_dir=tmpdir
            )
            restored = load_checkpoint(path)
            self.assertEqual(restored.spec, "rotate instruction")
            self.assertEqual(restored.ops, ["RdInstr()", "WrRD(data, rd)"])
            self.assertEqual(restored.op_index, 1)
            self.assertEqual(restored.run_id, "20250420-120000")

    def test_load_checkpoint_missing_full_state_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bundle.json"
            path.write_text(
                json.dumps({"error": "oops", "thin_state": None}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                load_checkpoint(str(path))
            self.assertIn("thin_state", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
