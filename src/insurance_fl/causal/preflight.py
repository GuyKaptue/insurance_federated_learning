CAUSAL_DEPS = ("statsmodels", "pgmpy")


def require_causal_deps() -> None:
    """Raise RuntimeError with a clear message if either dep is missing.

    Message must be exactly:
      "Causal layer requires {missing}. Run: make install-causal"
    """
    import importlib.util

    missing = []
    for dep in CAUSAL_DEPS:
        spec = importlib.util.find_spec(dep)
        if spec is None:
            missing.append(dep)
    if missing:
        raise RuntimeError(f"Causal layer requires {missing}. Run: make install-causal")
