import collections
from scipy.sparse import csr_matrix
import scipy.sparse

from .matrix import metaedge_to_adjacency_matrix
from .degree_weight import dwwc_step
from .degree_weight import dwwc


def dwpc(graph, metapath, damping=1.0, verbose=False):
    """
    Compute the degree-weighted path count (DWPC) on specified metapath
    and graph, with the paths normalized by the given damping parameter.

    This function computes the dwpc for all paths of the given metpath
    type, via matrix multiplies and corrections to handle non-path walks.

    NOTE: this function can handle metapaths that do not repeat
    metatnode typs, and metapaths that repeat exactly one node type
    any number of times, but cannot handle metapaths that repeat
    more than one type of node.

    Parameters
    ==========
    graph : hetio.hetnet.Graph
        graph to extract adjacency matrices along
    metapath : hetio.hetnet.MetaPath
        metapath for which function computes DWPCs
    damping : scalar
        exponent of degree in degree-weighting of DWPC
    verbose : bool
        set to True to have function print to screen
        (temporary, for debugging)
    """

    # Check that no more than 1 metanodetype is repeated in metapath
    nodetype_sequence = collections.Counter(str(item) for item
                                            in metapath.get_nodes())
    repeated_node = list(filter(lambda x: nodetype_sequence[x] >= 2,
                         nodetype_sequence))
    if len(repeated_node) > 1:
        raise NotImplementedError("Metapath repeats more than one metanode")

    # Compute DWPC
    elif len(repeated_node) == 0:  # case 1: no repeated node types
        return dwwc(graph, metapath, damping)

    else:  # case 2: one node type repeated
        # Determine start/endpoints for head, loop, tail
        if verbose:
            print("Input metapath repeats exactly one nodetype")
            print("Repeated node list: {}, type {}"
                  "".format(repeated_node, repeated_node[0]))

        first_appearance = None
        last_appearance = None
        for idx, metaedge in enumerate(metapath):
            if repeated_node[0] == str(metaedge.source):
                if first_appearance is None:
                    first_appearance = idx
            if repeated_node[0] == str(metaedge.target):
                last_appearance = idx
        if verbose:
            print("metapath has {} nodes, and {} are the repeated ones"
                  "".format(len(metapath),
                            [first_appearance, last_appearance]))

        # Iterate over each meta-edge adjacency matrix
        dwpc_matrix = None
        head_matrix = None
        for idx, metaedge in enumerate(metapath):
            if verbose:
                print("\tWorking on {}".format(idx))
            _, _, adj_mat = metaedge_to_adjacency_matrix(graph, metaedge)
            adj_mat = dwwc_step(adj_mat, damping, damping, False)
            adj_mat = csr_matrix(adj_mat)
            if idx < first_appearance:  # before repeated_node appears
                if head_matrix is None:
                    head_matrix = adj_mat.copy()
                else:
                    head_matrix = head_matrix @ adj_mat
            elif idx <= last_appearance:  # i.e. metanodes[0] = repeatednode
                if dwpc_matrix is None:
                    dwpc_matrix = adj_mat.copy()
                else:
                    dwpc_matrix = dwpc_matrix @ adj_mat
                    # if endpoints are same type, subtract diagonal after mult
                    if str(metaedge.target) == repeated_node[0]:
                        dwpc_matrix -= \
                            scipy.sparse.diags([dwpc_matrix.diagonal(
                                ).astype(float)], [0])
            else:  # covers the tail cases
                dwpc_matrix = dwpc_matrix @ adj_mat

        return head_matrix @ dwpc_matrix
