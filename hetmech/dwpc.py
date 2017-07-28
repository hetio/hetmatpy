import functools
import operator

import numpy

from .matrix import metaedge_to_adjacency_matrix, normalize, copy_array


def remove_diag(mat):
    """Set the main diagonal of a square matrix to zeros."""
    assert mat.shape[0] == mat.shape[1]  # must be square
    return mat - numpy.diag(mat.diagonal())


def degree_weight(matrix, damping, copy=True):
    """Normalize an adjacency matrix by the in and out degree."""
    matrix = copy_array(matrix, copy)
    row_sums = numpy.array(matrix.sum(axis=1)).flatten()
    column_sums = numpy.array(matrix.sum(axis=0)).flatten()
    matrix = normalize(matrix, row_sums, 'rows', damping)
    matrix = normalize(matrix, column_sums, 'columns', damping)

    return matrix


def dwpc_no_repeats(graph, metapath, damping=0.5):
    assert len(set(metapath.edges)) == len(metapath)

    parts = list()
    for metaedge in metapath:
        rows, cols, adj = metaedge_to_adjacency_matrix(
            graph, metaedge, dtype=numpy.float64, sparse_threshold=0)
        adj = degree_weight(adj, damping)

        if metaedge == metapath[0]:
            row_names = rows
        if metaedge == metapath[-1]:
            col_names = cols
        parts.append(adj)

    dwpc_matrix = functools.reduce(operator.matmul, parts)
    return row_names, col_names, dwpc_matrix


def dwpc_baab(graph, metapath, damping=0.5):
    """
    A function to handle metapath (segments) of the form BAAB.
    This function will handle arbitrary lengths of this repeated
    pattern. For example, ABCCBA, ABCDDCBA, etc. all work with this
    function. Random non-repeat inserts are supported. The metapath
    must start and end with a repeated node, though.

    Covers all variants of symmetrically repeated metanodes with
    support for random non-repeat metanode inserts at any point.
    Metapath must start and end with a repeated metanode.


    Parameters
    ----------
    graph : hetio.hetnet.Graph
    metapath : hetio.hetnet.MetaPath
    damping : float

    Examples
    --------
    Acceptable metapaths forms include the following:
    B-A-A-B
    B-C-A-A-B
    B-C-A-D-A-E-B
    B-C-D-E-A-F-A-B
    """
    metanodes = list(metapath.get_nodes())
    repeated_nodes = [v for i, v in enumerate(metanodes) if
                      v in metanodes[i + 1:]]
    # Find the indices of the innermost repeat (eg. BACAB -> 1,3)
    first_inner, second_inner = [i for i, metanode in enumerate(metanodes) if
                                 metanode == repeated_nodes[-1]]
    dwpc_inner = None

    # Traverse between and including innermost repeated metanodes
    inner_metapath = graph.metagraph.get_metapath(
        metapath[first_inner:second_inner])
    dwpc_inner = dwpc_short_repeat(graph, inner_metapath, damping=damping)[2]

    def next_outer(first_ind, last_ind, inner_array):
        """
        A recursive function. Works outward from the middle of a
        metapath. Multiplies non-repeat metanodes as appropriate and
        builds outward. When identical metanodes are ahead of and
        behind the middle segment being worked with, this function
        multiplies by both and subtracts the main diagonal.

        Parameters
        ----------
        first_ind : int
            index at the beginning of the middle segment
        last_ind : int
            index at the end of the middle segment
        inner_array : numpy.ndarray
            The working dwpc_matrix, which is multiplied from the front
            and back depending on which side has a duplicated metanode
            at the closest position
        """
        # case where node at the end is a repeated metanode
        if metanodes[last_ind + 1] in repeated_nodes:
            # if middle segment surrounded by repeated metanodes
            if metanodes[first_ind - 1] == metanodes[last_ind + 1]:
                adj1 = metaedge_to_adjacency_matrix(
                    graph, metapath[first_ind - 1])[2]
                adj2 = metaedge_to_adjacency_matrix(
                    graph, metapath[last_ind])[2]
                adj1 = degree_weight(adj1, damping)
                adj2 = degree_weight(adj2, damping)

                inner_array = adj1 @ (inner_array @ adj2)
                inner_array = remove_diag(inner_array)
                first_ind, last_ind = first_ind - 1, last_ind + 1
            # only trailing metanode is a repeat
            else:
                adj = metaedge_to_adjacency_matrix(
                    graph, metapath[first_ind - 1])[2]
                adj = degree_weight(adj, damping)
                inner_array = adj @ inner_array
                first_ind -= 1
        # trailing metanode is not a repeated
        else:
            adj = metaedge_to_adjacency_matrix(graph, metapath[last_ind])[2]
            adj = degree_weight(adj, damping)
            inner_array = inner_array @ adj
            last_ind += 1
        # the middle segment spans the entire metapath
        if len(metapath) == last_ind - first_ind:
            return inner_array
        else:
            return next_outer(first_ind, last_ind, inner_array)

    # get source and target ID arrays
    row_names = metaedge_to_adjacency_matrix(
        graph, metapath[0], dtype=numpy.float64)[0]
    col_names = metaedge_to_adjacency_matrix(
        graph, metapath[-1], dtype=numpy.float64)[1]
    dwpc_matrix = next_outer(first_inner, second_inner, dwpc_inner)
    return row_names, col_names, dwpc_matrix


def dwpc_baba(graph, metapath, damping=0.5):
    """
    Computes the degree-weighted path count for overlapping metanode
    repeats of the form B-A-B-A. Note that this does NOT yet support
    B-A-C-B-A, or any sort of metapath wherein there are metanodes
    within the overlapping region. This is the biggest priority to add.
    """
    raise NotImplementedError("See PR #61")


def dwpc_short_repeat(graph, metapath, damping=0.5):
    """
    One metanode repeated 3 or fewer times (A-A-A), not (A-A-A-A)
    This can include other random inserts, so long as they are not
    repeats. Must start and end with the repeated node. Acceptable
    examples: (A-B-A-A), (A-B-A-C-D-E-F-A), (A-B-A-A), etc.
    """
    start_metanode = metapath.source()
    assert start_metanode == metapath.target()

    dwpc_matrix = None
    dwpc_tail = None
    index_of_repeats = [i for i, v in enumerate(metapath.get_nodes()) if
                        v == start_metanode]

    for metaedge in metapath[:index_of_repeats[1]]:
        row, col, adj = metaedge_to_adjacency_matrix(
            graph, metaedge, dtype=numpy.float64, sparse_threshold=0)
        adj = degree_weight(adj, damping)
        if dwpc_matrix is None:
            row_names = col_names = row
            dwpc_matrix = adj
        else:
            dwpc_matrix = dwpc_matrix @ adj

    dwpc_matrix = remove_diag(dwpc_matrix)

    if len(index_of_repeats) == 3:
        for metaedge in metapath[index_of_repeats[1]:]:
            row, col, adj = metaedge_to_adjacency_matrix(
                graph, metaedge, dtype=numpy.float64, sparse_threshold=0)
            adj = degree_weight(adj, damping)
            if dwpc_tail is None:
                dwpc_tail = adj
            else:
                dwpc_tail = dwpc_tail @ adj
        dwpc_tail = remove_diag(dwpc_tail)
        dwpc_matrix = dwpc_matrix @ dwpc_tail
        dwpc_matrix = remove_diag(dwpc_matrix)

    return row_names, col_names, dwpc_matrix


def dwpc_long_repeat(graph, metapath, damping=0.5):
    """One metanode repeated 4 or more times. Considerably slower than
    dwpc_short_repeat, so should only be used if necessary. This
    function uses history vectors that split the computation into more
    tasks."""
    raise NotImplementedError("See PR #59")


def get_segments(metagraph, metapath):
    """Should categorize things into more than just the five categories
    in PR # 60. We want to segment the metapath into long-repeats, short-
    repeats, BABA, (which can not at the moment include other intermediates),
    BAAB (which can have intermediates as long as the whole thing is
    symmetrical), and other, non-segment-able regions."""
    raise NotImplementedError("Will integrate PR #60")


def dwpc(graph, metapath, damping=0.5):
    """This function will call get_segments, then the appropriate function"""
    raise NotImplementedError
