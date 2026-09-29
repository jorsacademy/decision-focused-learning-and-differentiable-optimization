import importlib


def test_omlt_backend_imports_without_optional_dependencies() -> None:
    module=importlib.import_module("trained_ml_opt.omlt_backend")
    assert hasattr(module,"build_omlt_pyomo_model")
