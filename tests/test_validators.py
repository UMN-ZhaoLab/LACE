import unittest

from src.validators import validate_hdl_tasks, validate_ops


class TestValidators(unittest.TestCase):
    def test_validate_ops_empty(self) -> None:
        ok, reason = validate_ops([])
        self.assertFalse(ok)
        self.assertIn("empty", reason)

    def test_validate_ops_empty_entry(self) -> None:
        ok, reason = validate_ops(["RdInstr", "  "])
        self.assertFalse(ok)
        self.assertIn("empty", reason)

    def test_validate_ops_valid(self) -> None:
        ok, reason = validate_ops(["RdInstr", "WrRD"])
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")

    def test_validate_ops_invalid(self) -> None:
        ok, reason = validate_ops(["NotARealOp"])
        self.assertFalse(ok)
        self.assertIn("not recognized", reason)

    def test_validate_ops_no_false_positive_substring(self) -> None:
        # Exact word-boundary matching should reject this
        ok, reason = validate_ops(["DebugRdInstrCustom"])
        self.assertFalse(ok)
        self.assertIn("not recognized", reason)

    def test_validate_ops_arithmetic_rejected_in_ops(self) -> None:
        ok, reason = validate_ops(["MUL(X, Y)", "ADD(a, b)"])
        self.assertFalse(ok)
        self.assertIn("MUL", reason)
        self.assertIn("ARITHMETIC", reason)
        self.assertIn("arithmetic_ops", reason)

    def test_validate_ops_arithmetic_misplaced(self) -> None:
        ok, reason = validate_ops(["insn = RdInstr()", "bitmask = SLICE(insn, 31, 20)"])
        self.assertFalse(ok)
        self.assertIn("SLICE", reason)
        self.assertIn("ARITHMETIC", reason)
        self.assertIn("arithmetic_ops", reason)
        self.assertIn("Allowed interface ops", reason)

    def test_validate_ops_unknown_with_hint(self) -> None:
        ok, reason = validate_ops(["foo = unknown_op(x)"])
        self.assertFalse(ok)
        self.assertIn("not recognized", reason)
        self.assertIn("Allowed interface ops", reason)

    def test_validate_hdl_tasks_empty(self) -> None:
        ok, reason = validate_hdl_tasks([])
        self.assertFalse(ok)
        self.assertIn("empty", reason)

    def test_validate_hdl_tasks_empty_entry(self) -> None:
        ok, reason = validate_hdl_tasks(["task1", ""])
        self.assertFalse(ok)
        self.assertIn("empty", reason)

    def test_validate_hdl_tasks_valid(self) -> None:
        ok, reason = validate_hdl_tasks(["add port x", "modify alu"])
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")


if __name__ == "__main__":
    unittest.main()
