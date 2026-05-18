from __future__ import annotations

import ast
from typing import Literal

from coverage_agent.contracts.schemas import ContextPayload

_IO_MODULE_ROOTS = frozenset({
    "requests", "urllib", "httpx", "aiohttp", "socket", "subprocess",
    "os", "pathlib", "boto3", "pymongo", "sqlalchemy", "psycopg2",
    "redis", "celery", "pika", "smtplib", "ftplib",
})

_IO_BUILTINS = frozenset({"open", "input"})


class _IODetector(ast.NodeVisitor):
    """Walks a parsed AST and sets found_io=True on the first IO-signalling call."""

    found_io: bool = False

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in _IO_BUILTINS:
            self.found_io = True
            return
        if isinstance(node.func, ast.Attribute):
            # Traverse the attribute chain to reach the root Name node.
            # e.g. requests.Session().get() → root is Name("requests")
            root = node.func.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in _IO_MODULE_ROOTS:
                self.found_io = True
                return
        self.generic_visit(node)


class GapFilter:
    """
    Classifies a gap as 'easy' (pure logic) or 'hard' (IO-coupled).

    Uses AST analysis of the function's primary_code rather than string
    matching — no false positives from substrings, no misses from
    non-standard naming. Designed as a safety net for repos where most
    gaps are already pure: a single hard gap slipping through is handled
    by TestWriter's sandbox; a false positive would silently skip a
    testable gap.
    """

    def classify(self, context: ContextPayload) -> Literal["easy", "hard"]:
        try:
            tree = ast.parse(context.primary_code)
        except SyntaxError:
            return "easy"  # unparseable → let TestWriter decide
        detector = _IODetector()
        detector.visit(tree)
        return "hard" if detector.found_io else "easy"
