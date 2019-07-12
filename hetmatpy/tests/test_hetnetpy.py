def test_hetnetpy_imports():
    """
    Test hetnetpy module imports (module formerly named hetio)
    """
    import hetnetpy
    import hetnetpy.readwrite
    import hetnetpy.hetnet
    # Create an empty metagraph
    hetnetpy.hetnet.MetaGraph()
