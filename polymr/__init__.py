"""
Polymr is a light-weight MapReduce library for single machine computation.  It supports a number of 
useful features such as map and reduce side joins, associative reduces, 
aggregations, multiprocessing, and more.

While the underlying engine uses MapReduce, Polymr is best utilized via it's DSL which provides
higher level functionality for complex workflows.

"""
from .polymr import Polymr, PMap, PReduce, PJoin, ARReduce


__all__ = ["Polymr", "PMap", "PReduce", "PJoin", "ARReduce"]

