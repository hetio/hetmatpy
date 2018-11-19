def test_hetio_imports():
    """
    Test hetio module imports
    """
    import hetio
    import hetio.readwrite
    import hetio.hetnet
    # Create an empty metagraph
    hetio.hetnet.MetaGraph()
