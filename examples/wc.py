import json
import sys
import logging

from polymr import Polymr

def main(fname):
    logging.basicConfig(level=logging.INFO,
            format='%(asctime)s %(levelname)s %(message)s')

    wc = Polymr.text(fname) \
            .map(lambda v: len(v.split())) \
            .a_group_by(lambda x: 1) \
                .sum()

    results = wc.run("word-count")
    for k, v in results:
        print("Word Count:", v)

    results.delete()

if __name__ == '__main__':
    main(sys.argv[1])
