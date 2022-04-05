import json

import hetnetpy.readwrite

import hetmatpy.hetmat

format_github_url = "https://github.com/{repo_slug}/raw/{commit}/{path}".format

hetnet_urls = {
    # Figure 2D of Himmelstein & Baranzini
    # (2015) PLOS Comp Bio. https://doi.org/10.1371/journal.pcbi.1004259.g002
    "disease-gene-example": format_github_url(
        repo_slug="hetio/hetnetpy",
        commit="9dc747b8fc4e23ef3437829ffde4d047f2e1bdde",
        path="test/data/disease-gene-example-graph.json",
    ),
    # The bupropion and nicotine dependence Hetionet v1.0 subgraph.
    "bupropion-subgraph": format_github_url(
        repo_slug="hetio/hetnetpy",
        commit="30c6dbb18a17c05d71cb909cf57af7372e4d4908",
        path="test/data/bupropion-CbGpPWpGaD-subgraph.json.xz",
    ),
    # A random Hetionet v1.0 subgraph.
    "random-subgraph": format_github_url(
        repo_slug="hetio/hetnetpy",
        commit="30c6dbb18a17c05d71cb909cf57af7372e4d4908",
        path="test/data/random-subgraph.json.xz",
    ),
}


hetnet_io_cache = {}


def get_graph(name, hetmat=False, directory=None):
    """
    If hetmat=True, import graph into a hetmat located on-disk at directory.
    """
    if name not in hetnet_urls:
        raise ValueError(
            f"{name} is not a supported test hetnet.\n"
            "Choose from the following currently defined hetnets: "
            + ", ".join(hetnet_urls)
        )
    if name not in hetnet_io_cache:
        url = hetnet_urls[name]
        read_file = hetnetpy.readwrite.open_read_file(url, text_mode=True)
        hetnet_io_cache[name] = read_file.read()
    writable = json.loads(hetnet_io_cache[name])
    graph = hetnetpy.readwrite.graph_from_writable(writable)
    if not hetmat:
        return graph
    assert directory is not None
    hetmat = hetmatpy.hetmat.hetmat_from_graph(graph, directory)
    return hetmat
