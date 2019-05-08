import itertools
import sys
import operator
import time
import logging
import json
import random

from .base import *
from .runner import MTRunner, Graph, Source
from .dataset import Chunker, CatDataset
from .inputs import MemoryInput, PathInput

try:
    unicode
except NameError:
    unicode = str

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
    Base Dampr class
    """
    def __init__(self, source, pmer):
        assert isinstance(source, Source)
        self.source = source
        self.pmer = pmer

    def run(self, name=None, **kwargs):
        """
        Evaluates the composed Dampr graph with the provided name and subsequent options.
        By default, uses /tmp as temporary storage.

        Returns a ValueEmitter useful for shell access.
        """
        if name is None:
            name = 'dampr/{}'.format(random.random())

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

    def _add_mapper(self, mapper):
        assert isinstance(mapper, Streamable)
        # If we have a fusable mapper, just add it to the set.
        return PMap(self.source, self.pmer, self.agg + [mapper])

    def _add_map(self, f):
        return self._add_mapper(Map(f))

    def sample(self, prob):
        """
        Samples data with a given probability.  For example:
        
        `graph.sample(0.1)` will uniformly sample 10% of the data in the collection.
        """
        assert 0 <= prob <= 1.0

        def _sample(k, v):
            if get_rand().random() < prob:
                yield k, v

        return self._add_map(_sample)
        
    def checkpoint(self, force=False, combiner=None, options=None):
        """
        Checkpoint forces Dampr to fuse all cached maps and add it as a MR stage.

        This is useful when sharing the results of a computation with multiple other graphs.
        
        Without checkpoint(), Dampr would execute the shared graph multiple times rather than reuse
        the results of the computation:

            >>> evens = Dampr.memory([1,2,3,4,5]).filter(lambda x: x % 2 == 0).checkpoint()
            >>> summed = evens.group_by(lambda x: 1).sum()
            >>> multiplied = evens.group_by(lambda x: 1).reduce(lambda x, y: x * y)

        """
        if len(self.agg) > 0 or force:
            aggs = [Map(_identity)] if len(self.agg) == 0 else self.agg[:]
            name = ' -> ' .join('{}'.format(str(a)) for a in aggs)
            name = 'Stage {}: %s' % (name)
            source, pmer = self.pmer._add_mapper([self.source], 
                    fuse(aggs), 
                    combiner=combiner,
                    name=name,
                    options=options)
            return PMap(source, pmer) 

        return self

    def custom_mapper(self, mapper, name=None, **options):
        """
        Custom Mapper provides a low-level interface to the underlying map function.
        Users can provide any instance which adhere's to the Mapper interface, allowing
        for powerful or specific implementations.

        This is typically used very rarely and has some caveats such as not fusing
        with other Mappers.  Similarly, the implementer needs to understand more of the
        nuances associated with keys.

            >>> from dampr.base import Map
            >>> Dampr.memory([1,2,3,4,5]).custom_mapper(Map(lambda k, x: [(k, x+1)])).read()
            [2, 3, 4, 5, 6]
        """
        if isinstance(mapper, Streamable):
            return self._add_mapper(mapper)

        assert isinstance(mapper, Mapper)
        name = name if name is not None else str(mapper)
        me = self.checkpoint()
        source, pmer = me.pmer._add_mapper([me.source], 
                mapper,
                name=name,
                options=options)
        
        return PMap(source, pmer)

    def custom_reducer(self, reducer, name=None, **options):
        """
        Allows the user to provide any Reducer which adheres to the Reducer interface.
        This is a very powerful, low-level interface which should be avoided when possible
        as it's easy to write bugs.

            >>> Dampr.memory([1,2,3,4,5]).custom_reducer(Reduce(lambda k, x: [(k, sum(x))])).read()
            [[(0, 1)], [(1, 2)], [(2, 3)], [(3, 4)], [(4, 5)]]
        """
        assert isinstance(reducer, Reducer)
        me = self.checkpoint(force=True)
        name = name if name is not None else str(reducer)
        new_source, pmer = me.pmer._add_reducer([me.source], 
                reducer, 
                name=name,
                options=options)
        
        return PMap(new_source, pmer)

    def partition_map(self, f, **options):
        """
        Provides a medium-level interface for writing custom functionality for mapping 
        a partition.  This can be used for creating custom logic in mappers for the sake
        of performance or additional functionality.

        Note: partition_map will still execute even if the partition is empty!  Make sure
        the logic handles empty sets!

        `f` is a function that takes in an iterator of items in the partition and yields 
        out group keys and new values.


            >>> def plus_one(items):
            ...   for num in items:
            ...     yield num, num + 1
            ...
            >>> Dampr.memory([1,2,3,4,5]).partition_map(plus_one).read()
            [2, 3, 4, 5, 6]

        """
        return self.custom_mapper(StreamMapper(f), **options)

    def partition_reduce(self, f):
        """
        `partition_reduce` is a medium-level function that allows for more complex logic
        during reductions.  It can be useful in certain cases where reductions over sets
        of reduced values is convenient, such as returning only the top K items in a dataset

        Note: `partition_reduce` will still execute on empty partitions!  Make sure the
        logic handles cases where a partition is empty!

            >>> def largest_number(it):
            ...   largest = float('-inf')
            ...   for group_key, its in it:
            ...     for value in its:
            ...       largest = max(largest, value)
            ...   yield "Largest", largest
            ...
            >>> Dampr.memory([1,2,3,4,5]).partition_reduce(largest_number).read(n_partitions=1)
            [('Largest', 5)]
        """
        return self.custom_reducer(StreamReducer(f))

    def len(self):
        """
        Counts the number of items in the 
	collection.

	    >>> Dampr.memory([1,2,3,4,5]).len().read()
	    [5]
        """

        def _map_count(items):
            count = 0
            for _ in items:
                count += 1

            yield 1, count

        def _reduce_count(groups):
            count = 0
            not_empty = False
            for _, counts in groups:
                not_empty = True
                for c in counts:
                    count += c

            if not_empty:
                yield 1, count

        return self \
                .partition_map(_map_count) \
                .partition_reduce(_reduce_count) \
                .map(lambda x: x[1])

    def map(self, f):
        """
        Maps elements in the underlying collection using function 
        `f`:

            >>> Dampr.memory([1,2,3,4,5]).map(lambda x: x + 1).read()
            [2, 3, 4, 5, 6]
        """
        def _map(k, v):
            yield k, f(v)

        return self._add_map(_map)

    def filter(self, f):
        """
        Filters items from a collection based on a predicate f.
        A predicate return True keeps the item.

            >>> Dampr.memory([1,2,3,4,5]).filter(lambda x: x % 2 == 1).read()
            [1, 3, 5]

        """
        def _filter(k, v):
            if f(v):
                yield k, v

        return self._add_map(_filter)

    def flat_map(self, f):
        """
        Maps elements in the underlying collection using function f, 
        flattening the results

            >>> Dampr.memory([1,2,3,4,5]).flat_map(range).read()
            [0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4]
        """
        def _flat_map(k, v):
            for vi in f(v):
                yield k, vi

        return self._add_map(_flat_map)

    def group_by(self, key, vf=lambda x: x):
        """
        Groups a collections of X by a key function, optionally mapping X to Y 
        using `vf`.  Returns a Reducer object for different types of aggregations

            >>> Dampr.memory([1,2,3,4,5]).group_by(lambda x: x % 2).reduce(lambda k, it: sum(it)).read()
            [(0, 6), (1, 9)]
        """
        def _group_by(_key, value):
            yield key(value), vf(value)

        pm = self._add_map(_group_by).checkpoint()
        return PReduce(pm.source, pm.pmer)

    def a_group_by(self, key, vf=lambda x: x):
        """
        Groups a collection of X by a key function and optionally mapping X to Y
        using `vf`.  It differs from `group_by` by requiring an associative reduction
        operator: by forcing this restriction, Dampr is able to perform a partial 
        reduction of the collection during the mapping sequence which can dramatically
        speed up the performace of the reduce stage.

        When possible, use a_group_by over the more general group_by.

            >>> Dampr.memory([1,2,3,4,5]).a_group_by(lambda x: x % 2).reduce(lambda x, y: x+y).read()
            [(0, 6), (1, 9)]
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
        
            >>> Dampr.memory([1,2,3,4,5]).filter(lambda x: x % 2 == 1).sort_by(lambda x: -x).read()
            [5, 3, 1]
        """
        def _sort_by(_key, value):
            yield key(value), value

        return self._add_map(_sort_by).checkpoint(options=options)

    def join(self, other):
        """
        Joins two independent computations, returning a Joining class.

        This is a powerful and expensive operation which can merge two Dampr 
        collections together.
        """
        assert isinstance(other, PBase)
        me = self.checkpoint(True)
        if isinstance(other, PMap):
            other = other.checkpoint(True)

        pmer = Dampr(me.pmer.graph.union(other.pmer.graph))
        return PJoin(me.source, pmer, other.source)

    def count(self, key=lambda x: x, **options):
        """
        Counts each item X in the collection by its key 
        function.

            >>> Dampr.memory([1,2,3,4,5]).count(lambda x: x % 2).read()
            [(0, 2), (1, 3)]
        """
        return self.a_group_by(key, lambda v: 1) \
                .reduce(operator.add, **options)

    def mean(self, key=lambda x: 1, value=lambda x: x, **options):
        """
        Finds the mean of X, grouped by its key function and optionally
        mapped by a value function:

            >>> ages = [("Andrew", 33), ("Alice", 42), ("Andrew", 12), ("Bob", 51)]
            >>> Dampr.memory(ages).mean(lambda x: x[0], lambda v: v[1]).read()
            [('Alice', 42.0), ('Andrew', 22.5), ('Bob', 51.0)]
        """
        def _mean_binop(x, y):
            return x[0] + y[0], x[1] + y[1]

        def _average(x):
            return (x[0], x[1][0] / float(x[1][1]))

        return self.a_group_by(key, lambda v: (value(v), 1)) \
                .reduce(_mean_binop, **options) \
                .map(_average)

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

            >>> Dampr.memory([1,2,3,4,5,6]).mean(lambda x: x % 2).cached().read()
            [(0, 4.0), (1, 3.0)]
        """
        # Run the pipeline, load it into memory, and create a new graph
        options['memory'] = True
        return self.checkpoint(options=options)

    def sink(self, path):
        """
        Since writes each X in a collection to a given path.  The path will
        create a directory and write each map or reduce partition into the
        directory as partitions.  Sink assumes each X in the collection is 
        already a unicode string.

            >>> Dampr.memory(["foo", "bar", "baz"]).sink("/tmp/foo").run()
            >>> open("/tmp/foo/0").read()
            >>> open("/tmp/foo/1").read()
            >>> open("/tmp/foo/2").read()
    """
        aggs = [Map(_identity)] if len(self.agg) == 0 else self.agg[:]
        name = ' -> ' .join('{}'.format(unicode(a)) for a in aggs)
        name = 'Stage {}: %s' % (name)
        source, pmer = self.pmer._add_sink([self.source], 
                fuse(aggs), 
                path=path,
                name=name,
                options=None)
        return PMap(source, pmer) 

    def sink_tsv(self, path):
        """
        A convenience function which takes a tuple or list, creates a simple
        tab-delimited output, and sinks it to the provided path:

            >>> Dampr.memory([("Hank Aaron", 755)]).sink_tsv("/tmp/foo").run()
            >>> open("/tmp/foo/0").read()
        """
        return self.map(lambda x: u'\t'.join(unicode(p) for p in x)).sink(path)

    def sink_json(self, path):
        """
        A convenience function which takes a simple python object and serializes
        it to a line-delimited json to the given path:

            >>> Dampr.memory([{"name": "Hank Aaron", "home runs": 755}]).sink_json("/tmp/foo").run()
            >>> open("/tmp/foo/0").read()
        """
        return self.map(json.dumps).sink(path)

    def cross_right(self, other, cross, memory=False):
        """
        Produces the cross product between two datasets during the map stage.
        This is an incredibly expensive operation when done against two large
        datasets.  

        When it is known that the right dataset is small, setting `memory` to `True`
        will cache the right dataset in memory, speeding up the computation significantly.

        Two items, Xi and Yi, are joined with the cross function.

            >>> left = Dampr.memory([1,2,3,4,5])
            >>> right = Dampr.memory(['foo', 'bar'])
            >>> left.cross_right(right, lambda x, y: (x, y)).read()
            [(1, 'foo'), (1, 'bar'), (2, 'foo'), (2, 'bar'), (3, 'foo'), (3, 'bar'), (4, 'foo'), (4, 'bar'), (5, 'foo'), (5, 'bar')]
        """
        assert isinstance(other, PMap)
        return other.cross_left(self, lambda xi, yi: cross(yi, xi), memory)

    def cross_left(self, other, cross, memory=False, **options):
        """
        Produces the cross product between two datasets during the map stage.
        This is an incredibly expensive operation when done against two large
        datasets.  

        When it is known that the left dataset is small, setting `memory` to `True`
        will cache the right dataset in memory, speeding up the computation significantly.

        Two items, Xi and Yi, are joined with the cross function.

            >>> left = Dampr.memory([1,2,3,4,5])
            >>> right = Dampr.memory(['foo', 'bar'])
            >>> left.cross_left(right, lambda x, y: (x, y)).read()
            [(1, 'foo'), (2, 'foo'), (3, 'foo'), (4, 'foo'), (5, 'foo'), (1, 'bar'), (2, 'bar'), (3, 'bar'), (4, 'bar'), (5, 'bar')]
        """
        def _cross(k1, v1, k2, v2):
            yield k1, cross(v2, v1)

        me = self.checkpoint()
        other = other.checkpoint()
        pmer = Dampr(me.pmer.graph.union(other.pmer.graph))
        name = 'Stage {}: Cross'
        source, pmer = pmer._add_mapper([other.source, me.source], 
                MapCrossJoin(_cross, cache=memory), 
                combiner=None,
                name=name,
                options=options)
        return PMap(source, pmer) 

    def cross_set(self, other, cross, agg=None, **options):
        """
        Joins across the entire set in other.  This will load up all the data in `other`
        using the aggregate function and pass it whole sale into a crossing function, `cross`.
        Note, the cross set should _not_ be modified and is assumed to be immutable.

            >>> left = Dampr.memory([1,2,3,4,5])
            >>> right = Dampr.memory([3, 5])
            >>> left.cross_set(right, lambda x, y: x in y, agg=set).read()
            [(1, False), (2, False), (3, True), (4, False), (5, True)]
        """
        def _cross(k1, v1, right):
            yield k1, cross(v1, right)

        if agg is None:
            agg = list

        def _aggregate(d):
            return agg(v for k, v in d)

        me = self.checkpoint()
        other = other.checkpoint()
        pmer = Dampr(me.pmer.graph.union(other.pmer.graph))
        name = 'Stage {}: CrossAll'
        source, pmer = pmer._add_mapper([other.source, me.source], 
                MapAllJoin(_cross, _aggregate), 
                combiner=None,
                name=name,
                options=options)
        return PMap(source, pmer) 

    def topk(self, k, value=None):
        """
        Computes the top k elements in the set.  The 'value' function is responsible for
        returning a comparable object.

            >>> Dampr.memory([1,3,2,4,2.2]).topk(2).read()
            [3, 4]
            >>> Dampr.memory([1,3,2,4,2.2]).topk(2, lambda x: -x).read()
            [1, 2]
        """
        if value is None:
            value = lambda x: x

        import heapq
        def map_topk(it):
            heap = []
            for x in it: 
                heapq.heappush(heap, (value(x), x))
                if len(heap) > k:
                    heapq.heappop(heap)

            return ((1, x) for x in heap)

        def reduce_topk(it):
            counts = (v for k, vit in it for v in vit)
            for count, x in heapq.nlargest(k, counts):
                yield x, 1

        return self \
                .partition_map(map_topk) \
                .partition_reduce(reduce_topk) \
                .map(lambda x: x[0])

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

        `reduce_buffer` is a constant which tells Dampr how much temporary storage
        to keep in memory on the map side reductions.  For example with `1000`, Dampr
        will keep 1000 unique keys in memory.  In the case where a new key would spill
        over the buffer size, Dampr will flush the buffer to disk and create a new
        buffer.  By increasing the `reduce_buffer`, you can increase efficiency while
        sacrificing memory.

            >>> Dampr.memory([1,2,3,4,5]).a_group_by(lambda x: 1).reduce(lambda x, y: x + y).read()
            [(1, 15)]
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
            >>> Dampr.memory([1,2,3,4,5]).a_group_by(lambda x: x % 2).first().read()
            [(0, 2), (1, 1)]
        """
        return self.reduce(lambda x, _y: x, **options)

    def sum(self, **options):
        """
        Simple sum of values by key.

            >>> Dampr.memory([1,2,3,4,5]).a_group_by(lambda x: x % 2).sum().read()
            [(0, 6), (1, 9)]
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

            >>> Dampr.memory([1,2,3,4,5]).group_by(lambda x: x % 2).reduce(lambda k, it: sum(it)).read()
            [(0, 6), (1, 9)]
        """
        new_source, pmer = self.pmer._add_reducer([self.source], KeyedReduce(f))
        return PMap(new_source, pmer)

    def unique(self, key=lambda x: x):
        """
        Returns the unique set of items in the grouping.

            >>> names = [("Andrew", 1), ("Andrew", 1), ("Andrew", 2), ("Becky", 13)]
            >>> Dampr.memory(names).group_by(lambda x: x[0], lambda x: x[1]).unique().read()
            [('Andrew', [1, 2]), ('Becky', [13])]
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

        pmer = Dampr(self.pmer.graph.union(other.pmer.graph))
        return PJoin(self.source, pmer, other.source)

    def partition_reduce(self, f):
        """
        Provides medium-level functionality for partition reductions.  See 
        PMap.partition_reduce for more details.
        """
        reducer = StreamReducer(f)
        new_source, pmer = self.pmer._add_reducer([self.source], reducer)
        return PMap(new_source, pmer)

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
        Performs an inner join between two datasets.  The aggregate function
        will be provied two arguments: the left iterator and the right iterator.


            >>> left = Dampr.memory([("foo", 13), ("bar", 14)]).group_by(lambda x: x[0])
            >>> right = Dampr.memory([("bar", "baller"), ("baz", "bag")]).group_by(lambda x: x[0])
            >>> left.join(right).reduce(lambda lit, rit: (list(lit), list(rit))).read()
            [('bar', ([('bar', 14)], [('bar', 'baller')]))]


        If `many` is True, reduce will flatten the output into seperate records:

            >>> left.join(right).reduce(lambda lit, rit: list(lit) + list(rit), many=True).read()
            [('bar', ('bar', 14)), ('bar', ('bar', 'baller'))]
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

            >>> left = Dampr.memory([("foo", 13), ("bar", 14)]).group_by(lambda x: x[0])
            >>> right = Dampr.memory([("bar", "baller"), ("baz", "bag")]).group_by(lambda x: x[0])
            >>> left.join(right).left_reduce(lambda lit, rit: (list(lit), list(rit))).read()
            [('bar', ([('bar', 14)], [('bar', 'baller')])), ('foo', ([('foo', 13)], []))]
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

class Dampr(object):
    """
    Entrypoint into the Dampr processing functions.
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

            >>> Dampr.memory([1,2,3,4,5])
        """
        mi = MemoryInput(list(enumerate(items)), partitions)
        source, ng = Graph().add_input(mi)
        return PMap(source, Dampr(ng))

    @classmethod
    def read_input(cls, *datasets):
        """
        Reads from the provided datasets.  When provided multiple datasets, it treats
        each as a separate partition.

        `read_input` can also take Chunker, which lazily returns a set of Datasets to
        operate over.

            >>> from dampr.inputs import MemoryInput
            >>> Dampr.read_input(MemoryInput(enumerate(range(5,10)), 2))
        """
        if len(datasets) == 1 and isinstance(datasets, Chunker):
            datasets = datasets[0]
        elif len(datasets) > 1:
            datasets = CatDataset(datasets)
        else:
            datasets = datasets[0]

        source, ng = Graph().add_input(datasets)
        return PMap(source, Dampr(ng))

    @classmethod
    def text(cls, fname, chunk_size=16*1024**2, followlinks=False):
        """
        Reads a file or directory of files into Dampr.  Each record is assumed to 
        be newline delimited.

        When fname is a directory, it will walk the directory, collecting all
        files within it as part of the collection.  `text` accepts globs.
        
        `chunk_size` describes how big each map portion will be.
        `followlinks` indiicates that Dampr will follow symlinked directories.

        Returns a PMap object.

            >>> Dampr.text('/tmp', chunk_size=64*1024**2)
        """
        return cls.read_input(PathInput(fname, chunk_size, followlinks))

    @classmethod
    def json(cls, *args, **kwargs):
        """
        Convenience function which reads newline-delimited json records.
        """
        return cls.text(*args, **kwargs).map(json.loads)

    @classmethod
    def from_dataset(cls, dataset):
        """
        Typically not used, this will read the raw outputs of a Dampr stage 
        as an input.
        """
        assert isinstance(dataset, Chunker)
        source, ng = Graph().add_input(dataset)
        return PMap(source, Dampr(ng))

    @classmethod
    def run(self, *pmers, **kwargs):
        """
        Runs a graph or set of graphs.

            >>> foo = Dampr.memory([1,2,3,4,5])
            >>> bar = Dampr.memory([6,7,8,9,10])
            >>> left, right = Dampr.run(foo, bar)
            >>> left.read()
            [1, 2, 3, 4, 5]
            >>> right.read()
            [6, 7, 8, 9, 10]
        """
        assert len(pmers) > 0, "Need at least one graph to run!"
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

        name = kwargs.pop('name', 'dampr/{}'.format(random.random()))
        ds = pmer.pmer.runner(name, graph, **kwargs).run(sources)
        return [ValueEmitter(d) for d in ds]

    def _add_mapper(self, *args, **kwargs): 
        output, ng = self.graph.add_mapper(*args, **kwargs)
        return output, Dampr(ng)

    def _add_reducer(self, *args, **kwargs): 
        output, ng = self.graph.add_reducer(*args, **kwargs)
        return output, Dampr(ng)

    def _add_sink(self, *args, **kwargs): 
        output, ng = self.graph.add_sink(*args, **kwargs)
        return output, Dampr(ng)

def fuse(aggs):
    if len(aggs) == 1:
        return aggs[0]

    s = aggs[1]
    for i in range(2, len(aggs)):
        s = ComposedStreamable(s, aggs[i])

    return ComposedMapper(aggs[0], s)

# This reinitializaes everytime
RANDOM = None
def get_rand():
    global RANDOM
    if RANDOM is None:
        RANDOM = random.Random(time.time())

    return RANDOM
