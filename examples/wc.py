import json
import sys

from polymr import Polymr

def main(fname):
    import logging
    logging.basicConfig(level=logging.INFO,
            format='%(asctime)s %(levelname)s %(message)s')

    polymr = Polymr()
    wc = polymr.text(fname) \
            .map(lambda v: len(v.split())) \
            .group_by(lambda x: 1) \
            .sum()

    results = wc.run("word-count")[0]
    for k, v in results.read():
        print("Word Count:", v)

    results.delete()

if __name__ == '__main__':
    main(sys.argv[1])
