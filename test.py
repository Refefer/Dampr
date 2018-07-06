import json
import sys

from base import EZMap, EZReduce, TextInput
from runner import MTRunner, SimpleRunner, Graph
from utils import first

@EZMap
def extract_listings(k, line):
    data = json.loads(line)
    yield data['listingId1'], data['listing1']
    yield data['listingId2'], data['listing2']

@EZMap
def split(k, doc):
    for tag in doc['tags'].lower().split('.'):
        for w in tag.split():
            yield w, doc['price']

@EZReduce
def count(key, it):
    s = c = 0
    for price in it:
        c += 1
        s += price

    return s, c 

@EZMap
def sort_values(key, value):
    s, c = value
    if c > 10:
        yield s / float(c), (s, c, key)

def main(fname):
    graph = Graph()
    out = graph.add_input(TextInput(fname))

    listings = graph.add_mapper([out], extract_listings)
    out = graph.add_reducer([listings], first())
    out = graph.add_mapper([out], split)
    out = graph.add_reducer([out], count)
    out = graph.add_mapper([out], sort_values)
    graph.add_output(out)
    runner = MTRunner('top-query-words', graph)
    results = runner.run()[0]
    for k, v in results.read():
        print k, v

if __name__ == '__main__':
    main(sys.argv[1])
