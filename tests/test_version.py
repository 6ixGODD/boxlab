import boxlab


def test_version() -> None:
    assert hasattr(boxlab, "__version__")
    assert isinstance(boxlab.__version__, str)
    assert len(boxlab.__version__) > 0
