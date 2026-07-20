import sys
import types
import unittest

if "unidiff" not in sys.modules:
    stub = types.ModuleType("unidiff")
    setattr(stub, "PatchSet", object)
    sys.modules["unidiff"] = stub

from src.file_utils import (
    _fuzzy_search,
    apply_search_replace_blocks,
    construct_new_file_content,
    parse_search_replace_blocks,
    strip_code_fences,
)


class TestFileUtils(unittest.TestCase):
    def test_strip_code_fences_multiple_blocks(self) -> None:
        content = """```python
print('a')
```

~~~
line1
line2
~~~
"""
        stripped = strip_code_fences(content)
        self.assertEqual(stripped, "print('a')\n\nline1\nline2")

    def test_strip_code_fences_no_fence(self) -> None:
        self.assertEqual(strip_code_fences("  hi "), "hi")

    def test_parse_search_replace_single_block(self) -> None:
        diff = "------- SEARCH\nold\n=======\nnew\n+++++++ REPLACE"
        blocks = parse_search_replace_blocks(diff)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0], ("old", "new"))

    def test_parse_search_replace_multiple_blocks(self) -> None:
        diff = (
            "------- SEARCH\na\n=======\nb\n+++++++ REPLACE"
            "------- SEARCH\nc\n=======\nd\n+++++++ REPLACE"
        )
        blocks = parse_search_replace_blocks(diff)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0], ("a", "b"))
        self.assertEqual(blocks[1], ("c", "d"))

    def test_apply_search_replace_blocks_ok(self) -> None:
        original = "hello world"
        diff = "------- SEARCH\nworld\n=======\nuniverse\n+++++++ REPLACE"
        result = apply_search_replace_blocks(original, diff)
        self.assertEqual(result, "hello universe")

    def test_apply_search_replace_blocks_no_match(self) -> None:
        original = "hello world"
        diff = "------- SEARCH\nfoo\n=======\nbar\n+++++++ REPLACE"
        with self.assertRaises(ValueError) as ctx:
            apply_search_replace_blocks(original, diff)
        self.assertIn("SEARCH/REPLACE block(s) failed to match", str(ctx.exception))

    def test_apply_search_replace_blocks_multiple_matches(self) -> None:
        original = "aaa bbb aaa"
        diff = "------- SEARCH\naaa\n=======\nzzz\n+++++++ REPLACE"
        with self.assertRaises(ValueError) as ctx:
            apply_search_replace_blocks(original, diff)
        self.assertIn("SEARCH/REPLACE block(s) failed to match", str(ctx.exception))

    def test_apply_search_replace_blocks_empty_search(self) -> None:
        original = "hello"
        diff = "------- SEARCH\n\n=======\nworld\n+++++++ REPLACE"
        with self.assertRaises(ValueError) as ctx:
            apply_search_replace_blocks(original, diff)
        self.assertIn("SEARCH/REPLACE block(s) failed to match", str(ctx.exception))

    def test_fuzzy_search_rejects_incomplete_match(self) -> None:
        """A high-similarity match that stops short of the search's last line
        should be rejected, otherwise replacement leaves stale content behind."""
        original = (
            "\t// Trace Interface\n"
            "\toutput reg        trace_valid,\n"
            "\toutput reg [35:0] trace_data,\n"
            "\n"
            "\t// ISA Extension Interface\n"
            "\toutput [31:0] RdInstr_0_o,\n"
            "\toutput        RdStall_0_o\n"
            ");\n"
        )
        search = (
            "\t// Trace Interface\n"
            "\toutput reg        trace_valid,\n"
            "\toutput reg [35:0] trace_data\n"
            ");"
        )
        result = _fuzzy_search(original, search)
        self.assertIsNone(result)

    def test_fuzzy_search_accepts_complete_match(self) -> None:
        """A match whose last non-empty line matches the search's last line
        should be accepted even when exact matching fails."""
        original = (
            "module top;\n"
            "  wire a;\n"
            "  wire b, c;\n"
            "endmodule\n"
        )
        search = (
            "  wire a;\n"
            "  wire b, c;"
        )
        result = _fuzzy_search(original, search)
        self.assertIsNotNone(result)
        start, end = result
        self.assertIn("wire b, c", original[start:end])

    def test_construct_new_file_content_search_replace_ok(self) -> None:
        original = "module old; endmodule"
        diff = "------- SEARCH\nold\n=======\nnew\n+++++++ REPLACE"
        result = construct_new_file_content(diff, original)
        self.assertEqual(result, "module new; endmodule")

    def test_construct_new_file_content_no_change(self) -> None:
        original = "module foo; endmodule"
        diff = "------- SEARCH\nfoo\n=======\nfoo\n+++++++ REPLACE"
        with self.assertRaises(ValueError) as ctx:
            construct_new_file_content(diff, original)
        self.assertIn("no changes", str(ctx.exception))

    def test_extract_multiple_diff_blocks(self) -> None:
        """extract_replace_diff should concatenate multiple <diff> blocks."""
        from src.file_utils import extract_replace_diff

        text = (
            "<replace_in_file>\n<diff>\n"
            "------- SEARCH\nold1\n=======\nnew1\n+++++++ REPLACE\n"
            "</diff>\n</replace_in_file>\n\n"
            "<replace_in_file>\n<diff>\n"
            "------- SEARCH\nold2\n=======\nnew2\n+++++++ REPLACE\n"
            "</diff>\n</replace_in_file>"
        )
        result = extract_replace_diff(text)
        self.assertIsNotNone(result)
        self.assertIn("old1", result)
        self.assertIn("old2", result)
        self.assertIn("new1", result)
        self.assertIn("new2", result)

    def test_parse_git_style_diff(self) -> None:
        """parse_search_replace_blocks should support Git-style diff markers."""
        from src.file_utils import parse_search_replace_blocks

        diff = "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE"
        blocks = parse_search_replace_blocks(diff)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0], ("old", "new"))

    def test_parse_git_style_diff_no_suffix(self) -> None:
        """parse_search_replace_blocks should support bare Git markers."""
        from src.file_utils import parse_search_replace_blocks

        diff = "<<<<<<<\nold\n=======\nnew\n>>>>>>>"
        blocks = parse_search_replace_blocks(diff)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0], ("old", "new"))


if __name__ == "__main__":
    unittest.main()
