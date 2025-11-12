import ast
import sys
from typing import List, Set


def dependency_analysis(code: str):
    tree = ast.parse(code)
    modules: Set[str] = set()
    relative: Set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level and module:
                relative.add(module.split(".")[0])
            elif node.level and not module:
                relative.add(".")
            else:
                modules.add(module.split(".")[0])

    std_lib = {name for name in modules if name in sys.stdlib_module_names}
    third_party = modules - std_lib

    return {
        "modules": sorted(modules),
        "standard_library": sorted(std_lib),
        "third_party": sorted(third_party),
        "relative": sorted(relative),
    }

