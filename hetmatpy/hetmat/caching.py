import functools
import heapq
import inspect
import itertools
import textwrap
import time

import numpy
import scipy.sparse
from hetnetpy.matrix import sparsify_or_densify

import hetmatpy.hetmat


def path_count_cache(metric):
    """
    Decorator to apply caching to the DWWC and DWPC functions from
    hetmatpy.degree_weight.
    """

    def decorator(user_function):
        signature = inspect.signature(user_function)

        @functools.wraps(user_function)
        def wrapper(*args, **kwargs):
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            arguments = bound_args.arguments
            graph = arguments["graph"]
            metapath = graph.metagraph.get_metapath(arguments["metapath"])
            arguments["metapath"] = metapath
            damping = arguments["damping"]
            cached_result = None
            start = time.perf_counter()
            supports_cache = (
                isinstance(graph, hetmatpy.hetmat.HetMat) and graph.path_counts_cache
            )
            if supports_cache:
                cache_key = {"metapath": metapath, "metric": metric, "damping": damping}
                cached_result = graph.path_counts_cache.get(**cache_key)
                if cached_result:
                    row_names, col_names, matrix = cached_result
                    matrix = sparsify_or_densify(matrix, arguments["dense_threshold"])
                    matrix = matrix.astype(arguments["dtype"])
            if cached_result is None:
                if arguments["dwwc_method"] is None:
                    # import default_dwwc_method here to avoid circular dependencies
                    from hetmatpy.degree_weight import default_dwwc_method

                    arguments["dwwc_method"] = default_dwwc_method
                row_names, col_names, matrix = user_function(**arguments)
            if supports_cache:
                runtime = time.perf_counter() - start
                graph.path_counts_cache.set(**cache_key, matrix=matrix, runtime=runtime)
            return row_names, col_names, matrix

        return wrapper

    return decorator


class PathCountCache:
    def __init__(self, hetmat):
        self.hetmat = hetmat
        self.cache = {}
        self.hits = {
            "memory": 0,
            "disk": 0,
            "absent": 0,
        }

    def get(self, metapath, metric, damping):
        """
        Return cached (row_ids, col_ids, matrix) with the specified path count
        metric or None if the cache does not contain the matrix. Attempts
        in-memory cache before falling back to on-disk cache.
        """
        matrix = None
        for metapath_, invert in (metapath, False), (metapath.inverse, True):
            key = metapath_, metric, damping
            if key in self.cache:
                matrix = self.cache[key]
                if invert:
                    matrix = matrix.transpose()
        if matrix is not None:
            self.hits["memory"] += 1
            row_ids = self.hetmat.get_node_identifiers(metapath.source())
            col_ids = self.hetmat.get_node_identifiers(metapath.target())
            return row_ids, col_ids, matrix
        try:
            result = self.hetmat.read_path_counts(metapath, metric, damping)
            self.hits["disk"] += 1
            return result
        except FileNotFoundError:
            self.hits["absent"] += 1
            return None

    def set(self, metapath, metric, damping, matrix, runtime):
        """
        Gives the cache the option of caching this matrix. This method never
        caches anything. Override this method in a subclass to enable setting
        the cache.
        """
        pass

    def get_stats(self):
        """
        Return a str with formatted stats about cache operations
        """
        hits_str = ", ".join(f"{kind} = {count:,}" for kind, count in self.hits.items())
        stats_str = textwrap.dedent(
            f"""\
            {self.__class__.__name__} containing {len(self.cache):,} items
              total gets: {sum(self.hits.values()):,}
              cache hits: {hits_str}"""
        )
        return stats_str


class PathCountPriorityCache(PathCountCache):
    def __init__(self, hetmat, allocate_GB):
        super().__init__(hetmat)
        self.bytes_per_gigabyte = 1_000_000_000
        self.allocate_B = self.bytes_per_gigabyte * allocate_GB
        self.current_B = 0
        # Dictionary of key to priority, where higher numbers are higher caching priority
        self.priorities = {}
        self.priority_queue = []
        # Use to generate a tie-breaker value for the queue as per
        # https://stackoverflow.com/a/40205720/4651668
        self.priority_queue_counter = itertools.count()

    def set(self, metapath, metric, damping, matrix, runtime):
        """
        Gives the cache the option of caching this matrix.
        """
        key = metapath, metric, damping
        if key in self.cache:
            return
        priority = self.priorities.get(key, 0)
        nbytes = get_matrix_size(matrix)
        if nbytes > self.allocate_B:
            return
        tie_breaker = next(self.priority_queue_counter)
        item = priority, tie_breaker, key, nbytes
        while self.current_B + nbytes > self.allocate_B:
            popped = heapq.heappop(self.priority_queue)
            popped_priority, _, popped_key, popped_nbytes = popped
            if popped_priority > priority:
                heapq.heappush(self.priority_queue, popped)
                break
            del self.cache[popped_key]
            self.current_B -= popped_nbytes
        else:
            heapq.heappush(self.priority_queue, item)
            self.cache[key] = matrix
            self.current_B += nbytes

    def get_stats(self):
        """
        Return a str with formatted stats about cache operations
        """
        stats_str = super().get_stats()
        stats_str += f"\n  {self.current_B / self.bytes_per_gigabyte:.2f} GB in use of {self.allocate_B / self.bytes_per_gigabyte:.2f} GB allocated"  # noqa: E501
        return stats_str


def get_matrix_size(matrix):
    """
    Estimate the size of a matrix object in bytes.
    """
    if isinstance(matrix, numpy.ndarray):
        return matrix.nbytes
    if scipy.sparse.isspmatrix_coo(matrix):
        return matrix.col.nbytes + matrix.row.nbytes + matrix.data.nbytes
    if (
        scipy.sparse.isspmatrix_csc(matrix)
        or scipy.sparse.isspmatrix_csr(matrix)
        or scipy.sparse.isspmatrix_bsr(matrix)
    ):  # noqa: E501
        return matrix.data.nbytes + matrix.indptr.nbytes + matrix.indices.nbytes
    if scipy.sparse.isparse(matrix):
        # Estimate size based on number of nonzeros for remaining sparse types
        return 2 * matrix.nnz * 4 + matrix.data.nbytes
    raise NotImplementedError(
        f"cannot calculate get_matrix_size for type {type(matrix)}"
    )
