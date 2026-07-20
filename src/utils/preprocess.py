"""Preprocessing utilities for LLM output cleaning."""

from __future__ import annotations

import html
import re


_CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")


def fix_model_html_escaping(text: str) -> str:
    """修复模型可能产生的 HTML 转义字符."""
    return html.unescape(text)


def remove_invalid_chars(text: str) -> str:
    """移除无效字符（例如某些控制字符）."""
    return _CTRL_RE.sub("", text)


def remove_extra_whitespace(text: str) -> str:
    """移除多余的空白字符，保留换行."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +\n", "\n", text)
    return text


def preprocess_content(
    content: str,
    *,
    fix_html: bool = True,
    remove_invalid: bool = True,
    rstrip: bool = True,
) -> str:
    """增强版预处理 - 支持模型特定处理."""
    if fix_html:
        content = fix_model_html_escaping(content)
    if remove_invalid:
        content = remove_invalid_chars(content)
    if rstrip:
        content = content.rstrip()
    return content


def clean_diff_content(content: str) -> str:
    """清理 diff 内容，确保格式正确."""
    content = re.sub(r"([^\n])------- SEARCH", r"\1\n------- SEARCH", content)
    content = re.sub(r"=======\n*", "=======\n", content)
    content = re.sub(r"(\n[^+])\+\+\+\+\+", r"\1+++++++ REPLACE", content)
    return content.strip()
