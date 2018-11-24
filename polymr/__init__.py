"""
Polymr is a light-weight MapReduce library for single machine computation.  It supports a number of 
useful features such as map and reduce side joins, associative reduces, 
aggregations, multiprocessing, and more.  It operates by default out-of-core, allowing
processing of 100s of gbs to tbs of data on a single machine.

While the underlying engine uses the MapReduce paradigm for processing, Polymr is 
best utilized via it's DSL which provides higher level functionality for complex workflows.

It loosely attempts to replicate interfaces such as Spark or Scalding for ease of pickup.
"""
import logging
from .polymr import Polymr, PMap, PReduce, PJoin, ARReduce
from .base import BlockMapper, BlockReducer


__all__ = ["Polymr", "PMap", "PReduce", "PJoin", "ARReduce", 
        "BlockMapper", "BlockReducer"]


def setup_logging(loglevel=logging.DEBUG):
    """
    Convenience function for enabling logging
    """
    logging.basicConfig(level=loglevel,
                        format='%(asctime)s %(levelname)s %(message)s')
