import tempfile
import unittest
import warnings
from pathlib import Path

from src.cpu_registry import list_cpu_choices, load_cpu_registry, resolve_cpu


class TestCpuRegistry(unittest.TestCase):
    def test_load_cpu_registry_basic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cpus.yaml"
            path.write_text(
                """
cpus:
  picorv32:
    cpu_dir: cpu_prototype/picorv32
    top_file: picorv32.v
    sv_include_dir: .
  ibex:
    cpu_dir: cpu_prototype/ibex
    top_file: rtl/ibex_top.sv
    sv_include_dir: rtl
""",
                encoding="utf-8",
            )
            registry = load_cpu_registry(str(path))
            self.assertIn("picorv32", registry)
            self.assertEqual(registry["picorv32"]["top_file"], "picorv32.v")
            self.assertIn("ibex", registry)

    def test_load_cpu_registry_parses_quoted_strings_and_tabs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cpus.yaml"
            # PyYAML handles quoted strings and tabs; the old hand-rolled parser did not.
            path.write_text(
                'cpus:\n  "test-cpu":\n    cpu_dir: "cpu_prototype/test"\n    top_file: "top.v"\n    sv_include_dir: "."\n',
                encoding="utf-8",
            )
            registry = load_cpu_registry(str(path))
            self.assertIn("test-cpu", registry)
            self.assertEqual(registry["test-cpu"]["cpu_dir"], "cpu_prototype/test")

    def test_resolve_cpu_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cpu_dir = Path(tmpdir) / "cpu_prototype" / "testcpu"
            cpu_dir.mkdir(parents=True)
            (cpu_dir / "top.v").write_text("module top; endmodule", encoding="utf-8")

            path = Path(tmpdir) / "cpus.yaml"
            path.write_text(
                f"""
cpus:
  testcpu:
    cpu_dir: {cpu_dir}
    top_file: top.v
    sv_include_dir: rtl
""",
                encoding="utf-8",
            )
            cfg = resolve_cpu("testcpu", str(path))
            self.assertEqual(cfg.name, "testcpu")
            self.assertEqual(Path(cfg.cpu_dir), cpu_dir)
            self.assertEqual(cfg.top_file, "top.v")
            self.assertEqual(cfg.sv_include_dir, "rtl")

    def test_resolve_cpu_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cpus.yaml"
            path.write_text(
                "cpus:\n  a:\n    cpu_dir: d\n    top_file: t.v\n    sv_include_dir: s\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                resolve_cpu("unknown", str(path))
            self.assertIn("unknown", str(ctx.exception))
            self.assertIn("a", str(ctx.exception))

    def test_resolve_cpu_missing_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cpus.yaml"
            path.write_text(
                """
cpus:
  badcpu:
    cpu_dir: somewhere
""",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                resolve_cpu("badcpu", str(path))
            self.assertIn("top_file", str(ctx.exception))
            self.assertIn("sv_include_dir", str(ctx.exception))

    def test_resolve_cpu_directory_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cpus.yaml"
            path.write_text(
                """
cpus:
  missing_dir:
    cpu_dir: /nonexistent/path
    top_file: top.v
    sv_include_dir: .
""",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                resolve_cpu("missing_dir", str(path))
            self.assertIn("directory does not exist", str(ctx.exception))

    def test_resolve_cpu_top_file_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cpu_dir = Path(tmpdir) / "cpu"
            cpu_dir.mkdir()
            path = Path(tmpdir) / "cpus.yaml"
            path.write_text(
                f"""
cpus:
  missing_file:
    cpu_dir: {cpu_dir}
    top_file: top.v
    sv_include_dir: .
""",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                resolve_cpu("missing_file", str(path))
            self.assertIn("top file does not exist", str(ctx.exception))

    def test_resolve_cpu_warns_on_extra_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cpu_dir = Path(tmpdir) / "cpu"
            cpu_dir.mkdir()
            (cpu_dir / "top.v").write_text("module top; endmodule", encoding="utf-8")
            path = Path(tmpdir) / "cpus.yaml"
            path.write_text(
                f"""
cpus:
  extrakey:
    cpu_dir: {cpu_dir}
    top_file: top.v
    sv_include_dir: .
    extra_field: ignored
""",
                encoding="utf-8",
            )
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                resolve_cpu("extrakey", str(path))
                self.assertTrue(any("extra_field" in str(warning.message) for warning in w))

    def test_list_cpu_choices_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cpus.yaml"
            path.write_text(
                """
cpus:
  z_cpu:
    cpu_dir: z
    top_file: t.v
    sv_include_dir: s
  a_cpu:
    cpu_dir: a
    top_file: t.v
    sv_include_dir: s
""",
                encoding="utf-8",
            )
            choices = list_cpu_choices(str(path))
            self.assertEqual(choices, ["a_cpu", "z_cpu"])

    def test_load_cpu_registry_file_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_cpu_registry("/nonexistent/cpus.yaml")

    def test_load_cpu_registry_missing_cpus_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.yaml"
            path.write_text("other_key: value\n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_cpu_registry(str(path))
            self.assertIn("cpus", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
