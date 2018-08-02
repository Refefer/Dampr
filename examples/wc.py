import json
import sys
import logging

from polymr import Polymr

def main(fname):
    logging.basicConfig(level=logging.INFO,
            format='%(asctime)s %(levelname)s %(message)s')

    wc = Polymr.text(fname) \
            .flat_map(lambda v: v.split()) \
            .fold_by(lambda x: x, value=lambda x: 1, binop=lambda x, y: x + y) \
            .sort_by(lambda x: -x[1])

    results = wc.run("word-count")
    for k, v in results:
        print("{}:".format(k), v)

    results.delete()

if __name__ == '__main__':
    main(sys.argv[1])
