import importlib


def test_optional_gurobi_module_imports_without_gurobi_installed() -> None:
    module = importlib.import_module("trained_ml_opt.gurobi_backend")
    assert hasattr(module, "optimize_with_gurobi_ml")
