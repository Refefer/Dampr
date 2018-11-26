from __future__ import print_function

import os
import sys
import re
import math
from collections import Counter

OUTDIR = '/tmp/baseline-idf'

RX = re.compile(r'[^\w]+')
with open(sys.argv[1]) as f:
    counter = Counter()
    for num_rows, line in enumerate(f):
        counter.update(set(RX.split(line.lower())))

    total = num_rows + 1
    if not os.path.isdir(OUTDIR):
        os.makedirs(OUTDIR)

    with open(os.path.join(OUTDIR, 'out'), 'w') as out:
        for word, count in counter.items():
            print("\t".join((word, unicode(count), 
                unicode(math.log(1 + float(total) / count)))), file=out)
