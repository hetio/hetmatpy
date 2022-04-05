def test_hetnetpy_imports():
    """
    Test hetnetpy module imports (module formerly named hetio)
    """
    import hetnetpy
    import hetnetpy.hetnet
    import hetnetpy.readwrite

    # Create an empty metagraph
    hetnetpy.hetnet.MetaGraph()
