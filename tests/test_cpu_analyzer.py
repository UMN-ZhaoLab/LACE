"""Tests for CPU analyzer prompt building."""

import tempfile
import unittest
from pathlib import Path

from src.cpu_analyzer import (
    build_analysis_prompt,
    collect_module_index,
    iter_source_files,
)


class TestCpuAnalyzer(unittest.TestCase):
    def test_iter_source_files_finds_sv_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "a.sv").write_text("module a; endmodule")
            Path(tmpdir, "b.v").write_text("module b; endmodule")
            Path(tmpdir, "readme.md").write_text("# readme")
            files = list(iter_source_files(tmpdir))
            names = {p.name for p in files}
            self.assertEqual(names, {"a.sv", "b.v"})

    def test_collect_module_index_keywords(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "fetch_unit.sv").write_text("")
            Path(tmpdir, "alu.v").write_text("")
            Path(tmpdir, "decoder.v").write_text("")
            index = collect_module_index(iter_source_files(tmpdir))
            labels = [line.split(":")[0] for line in index]
            self.assertIn("Fetch", labels)
            self.assertIn("Execute", labels)
            self.assertIn("Decode", labels)

    def test_build_analysis_prompt_contains_files_and_excerpts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            top = Path(tmpdir, "top.sv")
            top.write_text("module top;\n  // top-level\nendmodule\n")
            alu = Path(tmpdir, "alu.v")
            alu.write_text("module alu;\n  assign out = a + b;\nendmodule\n")

            prompt = build_analysis_prompt(tmpdir, iter_source_files(tmpdir))
            self.assertIn("CPU directory:", prompt)
            self.assertIn("Module Index", prompt)
            self.assertIn("top.sv", prompt)
            self.assertIn("alu.v", prompt)
            self.assertIn("module top;", prompt)

    def test_build_analysis_prompt_truncates_long_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            big = Path(tmpdir, "big.sv")
            big.write_text("x\n" * 5000)
            prompt = build_analysis_prompt(tmpdir, [big])
            self.assertIn("[truncated]", prompt)


if __name__ == "__main__":
    unittest.main()
