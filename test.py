import sys

from base import EZMap, EZReduce, TextInput
from runner import MTRunner, SimpleRunner, Graph

@EZMap
def split(k, line):
    yield len(line.split()), (1, len(line))

@EZReduce
def count(key, it):
    lc, bc = 0, 0
    for lines, bytes in it:
        lc += lines
        bc += bytes

    return lc, bc

def main(fname):
    graph = Graph()
    inp = graph.add_input(TextInput(fname))
    out = graph.add_mapper([inp], split)
    out = graph.add_reducer([out], count)
    graph.add_output(out)
    runner = MTRunner(graph)
    results = runner.run()[0]
    for k, v in results.read():
        print k, v

if __name__ == '__main__':
    main(sys.argv[1])
