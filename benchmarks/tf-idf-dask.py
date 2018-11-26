import re
import sys
import math
import os
import multiprocessing

import dask.bag as db

chunk_size = os.stat(sys.argv[1]).st_size / multiprocessing.cpu_count()
docs = db.read_text(sys.argv[1], blocksize=chunk_size + 1)

RX = re.compile(r'[^\w]+')
doc_freq = docs \
        .map(lambda x: set(RX.split(x.lower()))) \
        .flatten() \
        .frequencies()

total = docs.foldby(lambda x: True, lambda x,y: x + 1,initial=0).map(lambda x: x[1])
idf = doc_freq.product(total) \
        .map(lambda items: (items[0][0], items[0][1], math.log(1 + items[1] / float(items[0][1])))) \
        .map(lambda x: u'\t'.join(str(xi) for xi in x)) \
        .to_textfiles("/tmp/dask-idf", compute=False)[0] \
        .compute(scheduler='processes')
