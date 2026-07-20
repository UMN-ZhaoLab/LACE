import unittest

from src.utils.preprocess import (
    clean_diff_content,
    fix_model_html_escaping,
    preprocess_content,
    remove_extra_whitespace,
    remove_invalid_chars,
)


class TestPreprocess(unittest.TestCase):
    def test_fix_model_html_escaping(self) -> None:
        self.assertEqual(
            fix_model_html_escaping("Hello &amp; World &lt;test&gt;"),
            "Hello & World <test>",
        )

    def test_remove_invalid_chars(self) -> None:
        cleaned = remove_invalid_chars("Hello\x00\x07World")
        self.assertNotIn("\x00", cleaned)
        self.assertNotIn("\x07", cleaned)

    def test_remove_extra_whitespace(self) -> None:
        self.assertEqual(remove_extra_whitespace("a   b\t c\n"), "a b c\n")

    def test_preprocess_content_defaults(self) -> None:
        result = preprocess_content("  Hi &amp; Bye\x00\n")
        self.assertEqual(result, "  Hi & Bye")

    def test_clean_diff_content(self) -> None:
        diff = "line1------- SEARCH\nfoo\n=======\nbar\n+++++++ REPLACE"
        cleaned = clean_diff_content(diff)
        self.assertIn("------- SEARCH", cleaned)
        self.assertIn("+++++++ REPLACE", cleaned)


if __name__ == "__main__":
    unittest.main()
