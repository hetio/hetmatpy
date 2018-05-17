import functools
import pathlib
import shutil

import pandas
import numpy
import scipy.sparse

import hetio.hetnet
import hetio.readwrite
import hetio.matrix


def hetmat_from_graph(graph, path, save_metagraph=True, save_nodes=True, save_edges=True):
    """
    Create a hetmat.HetMat from a hetio.hetnet.Graph.
    """
    assert isinstance(graph, hetio.hetnet.Graph)
    hetmat = HetMat(path, initialize=True)
    hetmat.metagraph = graph.metagraph

    # Save metanodes
    metanodes = list(graph.metagraph.get_nodes())
    for metanode in metanodes:
        path = hetmat.get_nodes_path(metanode)
        rows = list()
        node_to_position = hetio.matrix.get_node_to_position(graph, metanode)
        for node, position in node_to_position.items():
            rows.append((position, node.identifier, node.name))
        node_df = pandas.DataFrame(rows, columns=['position', 'identifier', 'name'])
        path = hetmat.get_nodes_path(metanode)
        node_df.to_csv(path, index=False, sep='\t')

    # Save metaedges
    metaedges = list(graph.metagraph.get_edges(exclude_inverts=True))
    for metaedge in metaedges:
        rows, cols, matrix = hetio.matrix.metaedge_to_adjacency_matrix(graph, metaedge, dense_threshold=1)
        path = hetmat.get_edges_path(metaedge, file_format='sparse.npz')
        scipy.sparse.save_npz(str(path), matrix, compressed=True)
    return hetmat


def hetmat_from_permuted_graph(hetmat, permutation_id, permuted_graph):
    """
    Assumes subdirectory structure and that permutations inherit nodes but not
    edges.
    """
    if not hetmat.permutations_directory.is_dir():
        hetmat.permutations_directory.mkdir()
    directory = hetmat.permutations_directory.joinpath(f'{permutation_id}.hetmat')
    if directory.is_dir():
        # If directory exists, back it up using a .bak extension
        backup_directory = directory.with_name(directory.name + '.bak')
        if backup_directory.is_dir():
            shutil.rmtree(backup_directory)
        shutil.move(directory, backup_directory)
    permuted_hetmat = HetMat(directory, initialize=True)
    permuted_hetmat.is_permutation = True
    permuted_hetmat.metagraph_path.symlink_to('../../metagraph.json')
    permuted_hetmat.nodes_directory.rmdir()
    permuted_hetmat.nodes_directory.symlink_to('../../nodes', target_is_directory=True)
    permuted_hetmat = hetmat_from_graph(
        permuted_graph, directory, save_metagraph=False, save_nodes=False)
    return permuted_hetmat


def read_matrix(path, file_format='infer'):
    path = str(path)
    if file_format == 'infer':
        if path.endswith('.sparse.npz'):
            file_format = 'sparse.npz'
        if path.endswith('.npy'):
            file_format = 'npy'
    if file_format == 'infer':
        raise ValueError('Could not infer file_format for {path}')
    if file_format == 'sparse.npz':
        # https://docs.scipy.org/doc/scipy-1.0.0/reference/generated/scipy.sparse.load_npz.html
        return scipy.sparse.load_npz(path)
    if file_format == 'npy':
        # https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.load.html
        return numpy.load(path)
    raise ValueError(f'file_format={file_format} is not supported.')


def find_read_matrix(path, file_formats=['sparse.npz', 'npy']):
    """
    Read a matrix at the given path (without an extenstion)
    """
    path = pathlib.Path(path)
    for file_format in file_formats:
        path = path.with_name(f'{path.name}.{file_format}')
        if not path.is_file():
            continue
        return read_matrix(path, file_format=file_format)
    raise FileNotFoundError(
        f'No matrix files found at {path} with any of these extensions: ' +
        ', '.join(file_formats))


class HetMat:

    # Supported formats for nodes files
    nodes_formats = {
        'tsv',
        # 'feather',
        # 'pickle',
        # 'json',
    }

    # Supported formats for edges files
    edges_formats = {
        'npy',
        'sparse.npz',
        # 'tsv',
    }

    def __init__(self, directory, initialize=False):
        """
        Initialize a HetMat with its MetaGraph.
        """
        self.directory = pathlib.Path(directory)
        self.metagraph_path = self.directory.joinpath('metagraph.json')
        self.nodes_directory = self.directory.joinpath('nodes')
        self.edges_directory = self.directory.joinpath('edges')
        # Permutations should set is_permutation=True
        self.is_permutation = False
        self.permutations_directory = self.directory.joinpath('permutations')
        if initialize:
            self.initialize()

    def initialize(self):
        """
        Initialize the directory structure. This function is intended to be
        called when creating new HetMat instance on disk.
        """
        # Create directories
        directories = [
            self.directory,
            self.nodes_directory,
            self.edges_directory,
        ]
        for directory in directories:
            if not directory.is_dir():
                directory.mkdir()

    @property
    @functools.lru_cache()
    def permutations(self):
        """
        Return a dictionary of permutation name to permutation directory.
        Assumes permutation name is the directory name minus its .hetmat
        extension.
        """
        permutations = {}
        for directory in sorted(self.permutations_directory.glob('*.hetmat')):
            if not directory.is_dir():
                continue
            permutation = HetMat(directory)
            permutation.is_permutation = True
            name, _ = directory.name.rsplit('.', 1)
            permutations[name] = permutation
        return permutations

    @property
    @functools.lru_cache()
    def metagraph(self):
        """
        HetMat.metagraph is a cached property. Hence reading the metagraph from
        disk should only occur once, the first time the metagraph property is
        accessed. See https://stackoverflow.com/a/19979379/4651668. If this
        method has issues, consider using cached_property from
        https://github.com/pydanny/cached-property.
        """
        return hetio.readwrite.read_metagraph(self.metagraph_path)

    @metagraph.setter
    def metagraph(self, metagraph):
        """
        Set the metagraph property by writing the metagraph to disk.
        """
        hetio.readwrite.write_metagraph(metagraph, self.metagraph_path)

    def get_nodes_path(self, metanode, file_format='tsv'):
        """
        Get the path for the nodes file for the specified metanode. Setting
        file_format=None returns the path without any exension suffix.
        """
        metanode = self.metagraph.get_metanode(metanode)
        path = self.nodes_directory.joinpath(f'{metanode}')
        if file_format is not None:
            path = path.with_name(f'{path.name}.{file_format}')
        return path

    def get_edges_path(self, metaedge, file_format='npy'):
        """
        Get the path for the edges file for the specified metaedge. Setting
        file_format=None returns the path without any exension suffix.
        """
        metaedge_abbrev = self.metagraph.get_metaedge(metaedge).get_abbrev()
        path = self.edges_directory.joinpath(f'{metaedge_abbrev}')
        if file_format is not None:
            path = path.with_name(f'{path.name}.{file_format}')
        return path

    @functools.lru_cache()
    def get_node_identifiers(self, metanode):
        """
        Returns a list of node identifiers for a metapath
        """
        path = self.get_nodes_path(metanode, file_format='tsv')
        node_df = pandas.read_table(path)
        return list(node_df['identifier'])

    def metaedge_to_adjacency_matrix(
            self, metaedge,
            dtype=None, dense_threshold=None,
            file_formats=['sparse.npz', 'npy']):
        """
        file_formats sets the precedence of which file to read in
        """
        metaedge = self.metagraph.get_metaedge(metaedge)
        # Attempt to read the edge matrix in the specified orientation
        path = self.get_edges_path(metaedge, file_format=None)
        try:
            matrix = find_read_matrix(path, file_formats=file_formats)
        except FileNotFoundError:
            # Read the edge matrix in the inverse orientation, then transpose
            path = self.get_edges_path(metaedge.inverse, file_format=None)
            matrix = find_read_matrix(path, file_formats=file_formats)
            matrix = matrix.transpose()
        if dense_threshold is not None:
            matrix = hetio.matrix.sparsify_or_densify(matrix, dense_threshold=dense_threshold)
        if dtype is not None:
            matrix = matrix.astype(dtype)
        row_ids = self.get_node_identifiers(metaedge.source)
        col_ids = self.get_node_identifiers(metaedge.target)
        return row_ids, col_ids, matrix
