"""
Polymr is a light-weight MapReduce library for single machine computation.  It supports a number of 
useful features such as map and reduce side joins, associative reduces, 
aggregations, multiprocessing, and more.

While the underlying engine uses MapReduce, Polymr is best utilized via it's DSL which provides
higher level functionality for complex workflows.
"""
import itertools
import sys
import operator
import time
import logging
import json
import random

from .base import *
from .runner import MTRunner, Graph, Source
from .dataset import MemoryInput, DirectoryInput, TextInput, Chunker

class ValueEmitter(object):
    """
    Reads values from a processed dataset.  Can be used on the shell
    to read the results of a computation.
    """
    def __init__(self, datasets):
        self.datasets = datasets

    def stream(self):
        """
        Streams the values in the dataset.
        """
        for _, v in self.datasets.read():
            yield v

    def read(self, k=None):
        """
        Reads the first k items from the dataset.  If k is None, reads the
        entire dataset.
        """
        if k is None:
            return list(self.stream())

        return list(itertools.islice(self.stream(), k))

    def __iter__(self):
        return self.stream()

    def delete(self):
        """
        Deletes the cached on-disk dataset.
        """
        self.datasets.delete()

class PBase(object):
    """
    Base Polymr class
    """
    def __init__(self, source, pmer):
        assert isinstance(source, Source)
        self.source = source
        self.pmer = pmer

    def run(self, name=None, **kwargs):
        """
        Evaluates the composed Polymr graph with the provided name and subsequent options.
        By default, uses /tmp as temporary storage.

        Returns a ValueEmitter useful for shell access.
        """
        if name is None:
            name = 'polymr/{}'.format(random.random())

        logging.debug("run source: %s", self.source)
        ds = self.pmer.runner(name, self.pmer.graph, **kwargs).run([self.source])
        return ValueEmitter(ds[0])

    def read(self, k=None, **kwargs):
        """
        Shorthand for run() followed by a read()
        """
        return self.run(**kwargs).read(k)

def _identity(k, v):
    yield k, v

class PMap(PBase):
    """
    Represents most mapping processes.  Internally, it collects consecutive mapping operations
    and fuses them together into a single operation, speeding up many types of operations.

    This class shouldn't be intialized manually.
    """

    def __init__(self, source, pmer, agg=None):
        super(PMap, self).__init__(source, pmer)
        self.agg = [] if agg is None else agg

    def run(self, name=None, **kwargs):
        """
        Run the defined graph.
        """
        if len(self.agg) > 0:
            return self.checkpoint().run(name, **kwargs)
        else:
            return super(PMap, self).run(name, **kwargs)

    def _add_map(self, f):
        return PMap(self.source, self.pmer, self.agg + [Map(f)])

    def sample(self, prob):
        """
        Samples data with a given probability.  For example:
        
        graph.sample(0.1) will uniformly sample 10% of the data in the collection.
        """
        assert 0 <= prob <= 1.0

        def _sample(k, v):
            if get_rand().random() < prob:
                yield k, v

        return self._add_map(_sample)
        
    def checkpoint(self, force=False, combiner=None, options=None):
        """
        Checkpoint forces Polymr to fuse all cached maps and add it as a MR stage.

        This is useful when sharing the results of a computation with multiple other graphs.
        
        Without checkpoint(), Polymr would execute the shared graph multiple times rather than reuse
        the results of the computation:

        ```
        >>> evens = Polymr.memory([1,2,3,4,5]).filter(lambda x: x % 2 == 0).checkpoint()
        >>> summed = evens.group_by(lambda x: 1).sum()
        >>> multiplied = evens.group_by(lambda x: 1).reduce(lambda x, y: x * y)
        ```
        """
        if len(self.agg) > 0 or force:
            aggs = [Map(_identity)] if len(self.agg) == 0 else self.agg[:]
            name = ' -> ' .join('{}'.format(a.mapper.__name__) for a in aggs)
            name = 'Stage {}: %s => %s' % (self.source, name)
            source, pmer = self.pmer._add_mapper([self.source], 
                    Map(fuse(aggs)), 
                    combiner=combiner,
                    name=name,
                    options=options)
            return PMap(source, pmer) 

        return self

    def map(self, f):
        """
        Maps elements in the underlying collection using function f:

        ```
        >>> Polymr.memory([1,2,3,4,5]).map(lambda x: x + 1).read()
        [2, 3, 4, 5, 6]
        ```
        """
        def _map(k, v):
            yield k, f(v)

        return self._add_map(_map)

    def filter(self, f):
        """
        Filters items from a collection based on a predicate f:

        ```
        >>> Polymr.memory([1,2,3,4,5]).filter(lambda x: x % 2 == 1).read()
        [1, 3, 5]
        ```
        """
        def _filter(k, v):
            if f(v):
                yield k, v

        return self._add_map(_filter)

    def flat_map(self, f):
        """
        Maps elements in the underlying collection using function f, flattening the results:
        ```
        >>> Polymr.memory([1,2,3,4,5]).flat_map(range).read()
        [0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4]
        ```
        """
        def _flat_map(k, v):
            for vi in f(v):
                yield k, vi

        return self._add_map(_flat_map)

    def group_by(self, key, vf=lambda x: x):
        """
        Groups a collections of X by a key function, optionally mapping X to Y 
        using `vf`.  Returns a Reducer object for different types of aggregations

        ```
        >>> Polymr.memory([1,2,3,4,5]).group_by(lambda x: x % 2).reduce(lambda k, it: sum(it)).read()
        [(0, 6), (1, 9)]
        ```
        """
        def _group_by(_key, value):
            yield key(value), vf(value)

        pm = self._add_map(_group_by).checkpoint()
        return PReduce(pm.source, pm.pmer)

    def a_group_by(self, key, vf=lambda x: x):
        """
        Groups a collection of X by a key function and optionally mapping X to Y
        using `vf`.  It differs from `group_by` by requiring an associative reduction
        operator: by forcing this restriction, Polymr is able to perform a partial 
        reduction of the collection during the mapping sequence which can dramatically
        speed up the performace of the reduce stage.

        When possible, use a_group_by over the more general group_by
        ```
        >>> Polymr.memory([1,2,3,4,5]).a_group_by(lambda x: x % 2).reduce(lambda x, y: x+y).read()
        [(0, 6), (1, 9)]
        ```
        """
        def _a_group_by(_key, value):
            yield key(value), vf(value)

        # We don't checkpoint here!
        pm = self._add_map(_a_group_by)
        return ARReduce(pm)

    def fold_by(self, key, binop, value=lambda x: x, **options):
        """
        Shortcut for a_group_by(key, value).reduce(binop)
        """
        return self.a_group_by(key, value).reduce(binop, **options)

    def sort_by(self, key, **options):
        """
        Sorts the results by a given key function.
        
        ```
        Polymr.memory([1,2,3,4,5]).filter(lambda x: x % 2 == 1).sort_by(lambda x: -x).read()
        [5, 3, 1]
        ```
        """
        def _sort_by(_key, value):
            yield key(value), value

        return self._add_map(_sort_by).checkpoint(options=options)

    def join(self, other):
        """
        Joins two independent computations, returning a Joining class.

        This is a powerful and expensive operation which can merge two Polymr 
        collections together.
        """
        assert isinstance(other, PBase)
        me = self.checkpoint(True)
        if isinstance(other, PMap):
            other = other.checkpoint(True)

        pmer = Polymr(me.pmer.graph.union(other.pmer.graph))
        return PJoin(me.source, pmer, other.source)

    def count(self, key=lambda x: x, **options):
        """
        Counts each item X in the collection by its key function:

        ```
        >>> Polymr.memory([1,2,3,4,5]).count(lambda x: x % 2).read()
        [(0, 2), (1, 3)]
        ```
        """
        return self.a_group_by(key, lambda v: 1) \
                .reduce(operator.add, **options)

    def mean(self, key=lambda x: 1, value=lambda x: x, **options):
        """
        Finds the mean of X, grouped by its key function and optionally
        mapped by a value function:

        ```
        >>> ages = [("Andrew", 33), ("Alice", 42), ("Andrew", 12), ("Bob", 51)]
        >>> Polymr.memory(ages).mean(lambda x: x[0], lambda v: v[1]).read()
        [('Alice', 42.0), ('Andrew', 22.5), ('Bob', 51.0)]
        ```
        """
        def _binop(x, y):
            return x[0] + y[0], x[1] + y[1]

        return self.a_group_by(key, lambda v: (value(v), 1)) \
                .reduce(_binop, **options) \
                .map(lambda x: (x[0], x[1][0] / float(x[1][1])))

    def inspect(self, prefix="", exit=False):
        """
        Inspect is a debug function which prints each item X that flows through
        it.  It's valueable in inspecting intermediate results of a pipeline 
        without changing the values internally.
        """
        def _inspect(k, v):
            print("{}: {}".format(prefix, v))
            yield k, v

        ins = self._add_map(_inspect)
        if exit:
            ins.run()
            sys.exit(0)

        return ins

    def cached(self, **options):
        """
        The cached function runs a graph and stores it in memory.  This is 
        useful for small datasets to gain extra performance in subsequent 
        computations.

        ```
        >>> Polymr.memory([1,2,3,4,5,6]).mean(lambda x: x % 2).cached().read()
        [(0, 4.0), (1, 3.0)]
        ```
        """
        # Run the pipeline, load it into memory, and create a new graph
        options['memory'] = True
        return self.checkpoint(options=options)

    def sink(self, path):
        """
        Since writes each X in a collection to a given path.  The path will
        create a directory and write each map or reduce partition into the
        directory as partitions.

        Sink assumes each X in the collection is already a unicode string.

        ```
        >>> Polymr.memory(["foo", "bar", "baz"]).sink("/tmp/foo").run()
        >>> open("/tmp/foo/0").read()
        'foo\n'
        >>> open("/tmp/foo/1").read()
        'bar\n'
        >>> open("/tmp/foo/2").read()
        'baz\n'
        ```
        """
        aggs = [Map(_identity)] if len(self.agg) == 0 else self.agg[:]
        name = ' -> ' .join('{}'.format(a.mapper.__name__) for a in aggs)
        name = 'Stage {}: %s => %s' % (self.source, name)
        source, pmer = self.pmer._add_sink([self.source], 
                Map(fuse(aggs)), 
                path=path,
                name=name,
                options=None)
        return PMap(source, pmer) 

    def sink_tsv(self, path):
        """
        A convenience function which takes a tuple or list, creates a simple
        tab-delimited output, and sinks it to the provided path:

        ```
        >>> Polymr.memory([("Hank Aaron", 755)]).sink_tsv("/tmp/foo").run()
        >>> open("/tmp/foo/0").read()
        'Hank Aaron\t755\n'
        ```
        """
        return self.map(lambda x: u'\t'.join(unicode(p) for p in x)).sink(path)

    def sink_json(self, path):
        """
        A convenience function which takes a simple python object and serializes
        it to a line-delimited json to the given path:

        ```
        >>> Polymr.memory([{"name": "Hank Aaron", "home runs": 755}]).sink_json("/tmp/foo").run()
        >>> open("/tmp/foo/0").read()
        '{"home runs": 755, "name": "Hank Aaron"}\n'
        ```
        """
        return self.map(json.dumps).sink(path)

    def cross_tiny_right(self, other, cross):
        """
        Produces the cross product between two datasets during the map stage.

        This is an incredibly expensive operation when done against two large
        datasets.  cross_tiny_right attempts to speed up the operation by caching
        the right dataset in memory.

        Two items, Xi and Yi, are joined with the cross function.

        ```
        >>> left = Polymr.memory([1,2,3,4,5])
        >>> right = Polymr.memory(['foo', 'bar'])
        >>> left.cross_tiny_right(right, lambda x, y: (x, y)).read()
        [(1, 'foo'), (1, 'bar'), (2, 'foo'), (2, 'bar'), (3, 'foo'), (3, 'bar'), (4, 'foo'), (4, 'bar'), (5, 'foo'), (5, 'bar')]
        ```
        """
        assert isinstance(other, PMap)
        return other.cross_tiny_left(self, lambda xi, yi: cross(yi, xi))

    def cross_tiny_left(self, other, cross, **options):
        """
        Produces the cross product between two datasets during the map stage.

        This is an incredibly expensive operation when done against two large
        datasets.  cross_tiny_right attempts to speed up the operation by caching
        the right dataset in memory.

        Two items, Xi and Yi, are joined with the cross function.

        ```
        >>> left = Polymr.memory([1,2,3,4,5])
        >>> right = Polymr.memory(['foo', 'bar'])
        >>> left.cross_tiny_left(right, lambda x, y: (x, y)).read()
        [(1, 'foo'), (2, 'foo'), (3, 'foo'), (4, 'foo'), (5, 'foo'), (1, 'bar'), (2, 'bar'), (3, 'bar'), (4, 'bar'), (5, 'bar')]
        ```
        """
        def _cross(k1, v1, k2, v2):
            yield k1, cross(v2, v1)

        pmer = self.checkpoint()
        other = other.checkpoint()
        pmer = Polymr(self.pmer.graph.union(other.pmer.graph))
        name = 'Stage {}: (%s X %s)' % (self.source, other.source)
        source, pmer = pmer._add_mapper([other.source, self.source], 
                MapCrossJoin(_cross, cache=True), 
                combiner=None,
                name=name,
                options=options)
        return PMap(source, pmer) 

class ARReduce(object):
    """
    Associative Reducer operators.
    """
    def __init__(self, pmap):
        self.pmap = pmap

    def reduce(self, binop, reduce_buffer=1000, **options):
        """
        Reduces a grouped dataset by an associative binary operator.
        This will do a partial reduce in the map stage, completing the reduction
        in the reduce stage.  It is often substantially faster than the more
        general group_by.

        ``
        >>> Polymr.memory([1,2,3,4,5]).a_group_by(lambda x: 1).reduce(lambda x, y: x + y).read()
        [(1, 15)]
        ```
        """
        def _reduce(key, vs):
            acc = next(vs)
            for v in vs:
                acc = binop(acc, v)

            return acc

        red = Reduce(_reduce)
        options.update({"binop": binop, "reduce_buffer": reduce_buffer})
        # We add the associative aggregator to the combiner during map
        pm = self.pmap.checkpoint(True, 
                combiner=PartialReduceCombiner(red), 
                options=options)
        return PReduce(pm.source, pm.pmer).reduce(_reduce)
    
    def first(self, **options):
        """
        Returns the first item found for a given key. 
        ```
        >>> Polymr.memory([1,2,3,4,5]).a_group_by(lambda x: x % 2).first().read()
        [(0, 2), (1, 1)]
        ```
        """
        return self.reduce(lambda x, _y: x, **options)

    def sum(self, **options):
        """
        Simple sum of values by key.

        ```
        >>> Polymr.memory([1,2,3,4,5]).a_group_by(lambda x: x % 2).sum().read()
        [(0, 6), (1, 9)]
        ```
        """
        return self.reduce(lambda x, y: x + y, **options)

class PReduce(PBase):
    """
    A more general reduce class.
    """

    def reduce(self, f):
        """
        Reduces a grouped set of items by f, which takes two arguments: the group key
        and an iterator lazily yield all items in the group.

        ```
        >>> Polymr.memory([1,2,3,4,5]).group_by(lambda x: x % 2).reduce(lambda k, it: sum(it)).read()
        [(0, 6), (1, 9)]
        ```
        """
        new_source, pmer = self.pmer._add_reducer([self.source], KeyedReduce(f))
        return PMap(new_source, pmer)

    def unique(self, key=lambda x: x):
        """
        Returns the unique set of items in the grouping.

        ```
        >>> names = [("Andrew", 1), ("Andrew", 1), ("Andrew", 2), ("Becky", 13)]
        >>> Polymr.memory(names).group_by(lambda x: x[0], lambda x: x[1]).unique().read()
        [('Andrew', [1, 2]), ('Becky', [13])]
        ```
        """
        def _uniq(k, it):
            seen = set()
            agg = []
            for v in it:
                fv = key(v)
                if fv not in seen:
                    seen.add(fv)
                    agg.append(v)

            return agg

        return self.reduce(_uniq)

    def join(self, other):
        """
        Performs a join between two datasets on a given key, returning a PJoin object.
        """
        assert isinstance(other, PBase)
        if isinstance(other, PMap):
            other = other.checkpoint(True)

        pmer = Polymr(self.pmer.graph.union(other.pmer.graph))
        return PJoin(self.source, pmer, other.source)

class PJoin(PBase):
    """
    Performs different types of joins between two grouped datasets.
    """

    def __init__(self, source, pmer, right):
        super(PJoin, self).__init__(source, pmer)
        self.right = right

    def run(self, name=None, **kwargs):
        return self.reduce(lambda l, r: (list(l), list(r))).run(name, **kwargs)

    def reduce(self, aggregate, many=False):
        """
        Performs an inner join between two datasets.

        ```
        >>> left = Polymr.memory([("foo", 13), ("bar", 14)]).group_by(lambda x: x[0])
        >>> right = Polymr.memory([("bar", "baller"), ("baz", "bag")]).group_by(lambda x: x[0])
        >>> left.join(right).reduce(lambda lit, rit: (list(lit), list(rit))).read()
        [('bar', ([('bar', 14)], [('bar', 'baller')]))]
        ```
        """
        def _reduce(k, left, right):
            return aggregate(left, right)

        source, pmer = self.pmer._add_reducer([self.source, self.right], 
                KeyedInnerJoin(_reduce, many))
        return PMap(source, pmer)

    def left_reduce(self, aggregate):
        """
        Performs a left join on two datasets.  In the case where the right dataset
        is missing the join key, it will call the aggregate function with an empty
        iterator.

        ```
        >>> left = Polymr.memory([("foo", 13), ("bar", 14)]).group_by(lambda x: x[0])
        >>> right = Polymr.memory([("bar", "baller"), ("baz", "bag")]).group_by(lambda x: x[0])
        >>> left.join(right).left_reduce(lambda lit, rit: (list(lit), list(rit))).read()
        [('bar', ([('bar', 14)], [('bar', 'baller')])), ('foo', ([('foo', 13)], []))]
        ```
        """
        def _reduce(k, left, right):
            return aggregate(left, right)

        source, pmer = self.pmer._add_reducer([self.source, self.right], 
                KeyedLeftJoin(_reduce))
        return PMap(source, pmer)

    def _cross(self, crosser):
        def _cross(k1, v1, k2, v2):
            return k1, crosser(v1, v2)

        source, pmer = self.pmer._add_reducer([self.source, self.right],
                KeyedCrossJoin(_cross))

        return PMap(source, pmer).map(lambda x: x[1])

class Polymr(object):
    """
    Entrypoint into the Polymr processing functions.
    """
    def __init__(self, graph=None, runner=None):
        if graph is None:
            graph = Graph()

        self.graph = graph 
        if runner is None:
            runner = MTRunner

        self.runner = runner

    @classmethod
    def memory(cls, items, partitions=50):
        """
        Create an in-memory dataset from the provided items.  `partitions` define how
        many initial functions there will be.

        ```
        >>> Polymr.memory([1,2,3,4,5])
        ```
        """
        mi = MemoryInput(list(enumerate(items)), partitions)
        source, ng = Graph().add_input(mi)
        return PMap(source, Polymr(ng))

    @classmethod
    def text(cls, fname, chunk_size=16*1024**2):
        """
        Reads a file or directory of files into Polymr.  Each record is assumed to 
        be newline delimited.  

        When fname is a directory, it will walk the directory, collecting all
        files within it as part of the collection.
        
        `chunk_size` describes how big each map portion will be.

        Returns a PMap object.

        ```
        >>> Polymr.text('/tmp', chunk_size=64*1024**2)
        ```
        """
        if os.path.isdir(fname):
            inp = DirectoryInput(fname, chunk_size)
        else:
            inp = TextInput(fname, chunk_size)

        source, ng = Graph().add_input(inp)
        return PMap(source, Polymr(ng))

    @classmethod
    def json(cls, *args, **kwargs):
        """
        Convenience function which reads newline-delimited json records.
        """
        return cls.text(*args, **kwargs).map(json.loads)

    @classmethod
    def from_dataset(cls, dataset):
        """
        Typically not used, this will read the raw outputs of a Polymr stage 
        as an input.
        """
        assert isinstance(dataset, Chunker)
        source, ng = Graph().add_input(dataset)
        return PMap(source, Polymr(ng))

    @classmethod
    def run(self, *pmers, **kwargs):
        """
        Runs a graph or set of graphs.

        ```
        >>> foo = Polymr.memory([1,2,3,4,5])
        >>> bar = Polymr.memory([6,7,8,9,10])
        >>> left, right = Polymr.run(foo, bar)
        >>> left.read()
        [1, 2, 3, 4, 5]
        >>> right.read()
        [6, 7, 8, 9, 10]
        ```
        """
        sources = []
        graph = None
        for i, pmer in enumerate(pmers):
            if isinstance(pmer, PMap):
                pmer = pmer.checkpoint()
            elif isinstance(pmer, PJoin):
                pmer = pmer.reduce(lambda l, r: (list(l), list(r)))
            
            if i == 0:
                graph = pmer.pmer.graph
            else:
                graph = pmer.pmer.graph.union(graph)

            sources.append(pmer.source)

        name = kwargs.pop('name', 'polymr/{}'.format(random.random()))
        ds = pmer.pmer.runner(name, graph, **kwargs).run(sources)
        return [ValueEmitter(d) for d in ds]

    def _add_mapper(self, *args, **kwargs): 
        output, ng = self.graph.add_mapper(*args, **kwargs)
        return output, Polymr(ng)

    def _add_reducer(self, *args, **kwargs): 
        output, ng = self.graph.add_reducer(*args, **kwargs)
        return output, Polymr(ng)

    def _add_sink(self, *args, **kwargs): 
        output, ng = self.graph.add_sink(*args, **kwargs)
        return output, Polymr(ng)

def fuse(aggs):
    if len(aggs) == 1:
        return aggs[0].mapper

    def run(it, agg):
        return ((ki, vi) for k, v in it for ki, vi in agg.mapper(k, v))

    def _fuse(k, v):
        it = iter([(k, v)])
        for agg in aggs:
            it = run(it, agg)

        return it

    return _fuse

# This reinitializaes everytime
RANDOM = None
def get_rand():
    global RANDOM
    if RANDOM is None:
        RANDOM = random.Random(time.time())

    return RANDOM
