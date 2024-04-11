import networkx as nx
import pickle
from itertools import product, islice


class PathCacheManager:
    def __init__(self, network, cache_file="path_cache.pkl"):
        self.network = network
        self.cache_file = cache_file
        self.path_cache = {}
        self.cached_loaded = False

    def cache_paths(self, origins, destinations, truncate=5):
        """
        Compute and cache paths for all combinations of i, o, d.

        :param origins: List of origin nodes.
        :param destinations: List of destination nodes.
        :param truncate: Maximum number of paths to cache for each combination.
        """
        path_cache = {}
        for i, o, d in product(self.network.nodes, origins, destinations):
            if i[-1] == "*":
                continue
            print(f"Caching paths for i={i}, o={o}, d={d}.")
            io_paths = list(
                nx.all_simple_edge_paths(
                    self.network, source=i, target=o, cutoff=truncate
                )
            )
            od_paths = list(
                nx.all_simple_edge_paths(
                    self.network, source=o, target=d, cutoff=truncate
                )
            )
            path_cache[(i, o, d)] = (io_paths, od_paths)

        # Save the cache to a file
        with open(self.cache_file, "wb") as f:
            pickle.dump(path_cache, f)
        print(f"Paths cached to {self.cache_file}.")
        self.path_cache = path_cache
        self.cached_loaded = True

    def load_cache(self):
        try:
            with open(self.cache_file, "rb") as f:
                self.path_cache = pickle.load(f)
            self.cache_loaded = True
        except FileNotFoundError:
            print(
                f"Cache file {self.cache_file} not found. Please generate the cache first."
            )

    def get_cached_paths(self, i, o, d):
        """
        Retrieve cached paths for a specific combination of i, o, d.

        :param i: Initial node.
        :param o: Origin node.
        :param d: Destination node.
        :return: A tuple of (io_paths, od_paths) if found, else None.
        """
        if not self.cache_loaded:
            print("Cache not loaded. Loading now...")
            self.load_cache()

        return self.path_cache.get((i, o, d))
