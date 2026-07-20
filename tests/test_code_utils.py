import tempfile
import unittest
from pathlib import Path

from src.utils.code_utils import get_code_of_block, get_code_of_module


class TestCodeUtils(unittest.TestCase):
    def test_get_code_of_module_v(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "foo.v").write_text("line1\nline2\n", encoding="utf-8")
            lines = get_code_of_module("foo", root_dir=tmpdir)
            self.assertEqual(lines, ["line1\n", "line2\n"])

    def test_get_code_of_module_sv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "bar.sv").write_text("a\n", encoding="utf-8")
            lines = get_code_of_module("bar", root_dir=tmpdir)
            self.assertEqual(lines, ["a\n"])

    def test_get_code_of_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "baz.v").write_text("l1\nl2\nl3\n", encoding="utf-8")
            lines = get_code_of_block("baz", begin=2, end=3, root_dir=tmpdir)
            self.assertEqual(lines, ["l2\n", "l3\n"])


if __name__ == "__main__":
    unittest.main()
