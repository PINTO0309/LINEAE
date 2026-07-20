"""Keep LINEAE independent from the removed reference-project source trees."""

import ast
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
RUNTIME_PATHS = (
    REPOSITORY / "main.py",
    REPOSITORY / "engine.py",
    REPOSITORY / "warmup.py",
    REPOSITORY / "datasets",
    REPOSITORY / "models",
    REPOSITORY / "tools",
    REPOSITORY / "util",
)


def _runtime_python_files():
    for path in RUNTIME_PATHS:
        if path.is_file():
            yield path
        else:
            yield from path.rglob("*.py")


def test_runtime_source_has_no_import_from_reference_projects():
    violations = []
    for path in _runtime_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            for module in modules:
                if module == "LINEA" or module.startswith("LINEA.") or "gazelle-dinov3" in module:
                    violations.append(f"{path.relative_to(REPOSITORY)}:{node.lineno}: {module}")
    assert not violations, "reference-project imports found:\n" + "\n".join(violations)
