"""Static import preflight shared by authored jobs and repository checks."""

import ast
import sys


def forbidden_imports(
    source: str, allowed_third_party: set[str], *, standalone: bool = False
) -> list[str]:
    allowed = sys.stdlib_module_names | {"entanglix"} | allowed_third_party
    tree = ast.parse(source)
    imports = []
    dynamic_calls = {"__import__", "importlib.import_module"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
            for alias in node.names:
                if alias.name == "importlib":
                    dynamic_calls.add(f"{alias.asname or alias.name}.import_module")
        elif isinstance(node, ast.ImportFrom):
            if node.level and standalone:
                imports.append("<relative import in standalone script>")
            elif not node.level and node.module:
                imports.append(node.module)
            if node.module == "importlib":
                dynamic_calls.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "import_module"
                )
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.unparse(node.func) in dynamic_calls:
            if (
                not node.args
                or not isinstance(node.args[0], ast.Constant)
                or not isinstance(node.args[0].value, str)
            ):
                imports.append("<unreviewed dynamic import>")
            else:
                imports.append(node.args[0].value)
    return [name for name in imports if name.split(".")[0] not in allowed]
