import pstats
from pstats import SortKey
p = pstats.Stats('cProfile')
p.strip_dirs().sort_stats('cumulative').print_stats()