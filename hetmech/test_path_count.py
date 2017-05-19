import numpy
from neo4j.v1 import GraphDatabase
import hetio.readwrite
import hetio.neo4j
import hetio.pathtools

from hetmech.matrix import get_node_to_position
from hetmech.path_count import dwpc


"""
Test hetmech.path_county.dwpc()
"""


def _get_cypher_output(metapath, source, target, damping_exponent):
    """Calls cypher on input node pair"""
    query = hetio.neo4j.construct_dwpc_query(metapath,
                                             property='identifier',
                                             unique_nodes=True)
    driver = GraphDatabase.driver("bolt://neo4j.het.io")
    params = {
        'source': source,
        'target': target,
        'w': damping_exponent,
    }
    with driver.session() as session:
        result = session.run(query, params)
        result = result.single()
    cypher_pc = result['PC']
    cypher_dwpc = result['DWPC']
    return cypher_pc, cypher_dwpc


def dwpc_damping(exponent, graph, metapath, compound_to_position,
                 disease_to_position):
    """Test dwpc with different damping parameters"""
    compound = 'DB01156'  # Bupropion
    disease = 'DOID:0050742'  # nicotine dependences
    i = compound_to_position[compound]
    j = disease_to_position[disease]
    cypher_pc, cypher_dwpc = _get_cypher_output(metapath, compound,
                                                disease, exponent)
    dwpc_matrix = dwpc(graph, metapath, exponent)

    assert numpy.allclose(dwpc_matrix[i, j], cypher_dwpc)


def exhaustive_dwpc_check(graph, metapath, compound_to_position,
                          disease_to_position):
    """Test dwpc vs cypher for all paths from single source"""
    exponent = 1
    dwpc_matrix = dwpc(graph, metapath, exponent)
    for item, idx in compound_to_position.items():
        i_idx = idx
        i_item = item
        break
    cypher_output = [0] * len(disease_to_position)
    for j_item, idx in disease_to_position.items():
        _, cypher_dwpc = _get_cypher_output(metapath, i_item, j_item, exponent)
        cypher_output[idx] = cypher_dwpc

    assert numpy.allclose(dwpc_matrix[i_idx, :].todense(), cypher_output)


def test_all():
    """Load objects for all other tests to use"""
    print("Preprocessing begin.")
    url = 'https://github.com/dhimmel/hetionet/raw/' + \
          '76550e6c93fbe92124edc71725e8c7dd4ca8b1f5/' + \
          'hetnet/json/hetionet-v1.0.json.bz2'
    graph = hetio.readwrite.read_graph(url)
    metagraph = graph.metagraph

    # CbGpPWpGaD contains duplicate metanodes,
    # so DWPC is not equivalent to DWPC
    metapath = metagraph.metapath_from_abbrev('CbGpPWpGaD')
    metapath.get_unicode_str()

    compound_to_position = {x.identifier: i for x, i in
                            get_node_to_position(graph,
                                                 'Compound').items()}
    disease_to_position = {x.identifier: i for x, i in
                           get_node_to_position(graph,
                                                'Disease').items()}

    print("Preprocessing done.")

    exhaustive_dwpc_check(graph, metapath, compound_to_position,
                          disease_to_position)

    for exponent in [0, 0.2, 0.4, 0.7, 1]:
        print("Testing dwpc with exponent {}".format(exponent))
        dwpc_damping(exponent, graph, metapath, compound_to_position,
                     disease_to_position)
