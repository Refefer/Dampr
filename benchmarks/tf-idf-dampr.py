import os
import re
import sys
import math
import multiprocessing

from dampr import Dampr, setup_logging

chunk_size = os.stat(sys.argv[1]).st_size / multiprocessing.cpu_count()
docs = Dampr.text(sys.argv[1], chunk_size + 1)

RX = re.compile(r'[^\w]+')
doc_freq = docs \
        .flat_map(lambda x: set(RX.split(x.lower()))) \
        .count(reduce_buffer=float('inf'))

idf = doc_freq.cross_right(docs.len(), 
        lambda df, total: (df[0], df[1], math.log(1 + (float(total) / df[1]))),
        memory=True)

idf.sink_tsv("/tmp/idfs").run()

