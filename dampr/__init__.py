"""
Dampr is a light-weight MapReduce library for single machine computation.  It supports a number of 
useful features such as map and reduce side joins, associative reduces, 
aggregations, multiprocessing, and more.  It operates by default out-of-core, allowing
processing of 100s of gbs to tbs of data on a single machine.

While the underlying engine uses the MapReduce paradigm for processing, Dampr is 
best utilized via it's DSL which provides higher level functionality for complex workflows.

It loosely attempts to replicate interfaces such as Spark or Scalding for ease of pickup.
"""
import logging
import sys
from .dampr import Dampr, PMap, PReduce, PJoin, ARReduce
from .base import BlockMapper, BlockReducer
from .dataset import Dataset

if sys.version_info.major == 3:
    import dampr.settings as settings
else:
    import settings

__all__ = ["Dampr", "PMap", "PReduce", "PJoin", "ARReduce", 
        "BlockMapper", "BlockReducer", "settings"]


def setup_logging(debug=False):
    """
    Convenience function for enabling logging
    """
    loglevel = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=loglevel,
                        format='%(asctime)s %(levelname)s %(message)s')
