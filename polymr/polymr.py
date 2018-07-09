import sys
import time
import logging
import json
import random

from .base import *
from .runner import MTRunner, Graph, Source
from .inputs import MemoryInput, TextInput

class ValueEmitter(object):
    def __init__(self, datasets):
        self.datasets = datasets

    def read(self):
        for _, v in self.datasets[0].read():
            yield v

    def __iter__(self):
        return self.read()

    def delete(self):
        self.datasets[0].delete()

class PBase(object):
    def __init__(self, source, pmer):
        assert isinstance(source, Source)
        self.source = source
        self.pmer = pmer

    def run(self, name=None):
        name = 'tmp' if name is None else name
        logging.info("run source: %s", self.source)
        ds = self.pmer.runner(name, self.pmer.graph).run([self.source])
        return ValueEmitter(ds)

def _identity(k, v):
    yield k, v

class PMap(PBase):

    def __init__(self, source, pmer, agg=None):
        super(PMap, self).__init__(source, pmer)
        self.agg = [] if agg is None else agg

    def run(self, name=None):
        if len(self.agg) > 0:
            return self.checkpoint().run(name)
        else:
            return super(PMap, self).run(name)

    def _add_map(self, f):
        return PMap(self.source, self.pmer, self.agg + [Map(f)])

    def sample(self, prob):
        def _sample(k, v):
            if get_rand().random() < prob:
                yield k, v

        return self._add_map(_sample)
        
    def checkpoint(self, force=False, combiner=None):
        if len(self.agg) > 0 or force:
            aggs = [Map(_identity)] if len(self.agg) == 0 else self.agg[:]
            name = ' -> ' .join('{}'.format(a.mapper.__name__) for a in aggs)
            name = 'Stage {}: %s => %s' % (self.source, name)
            source, pmer = self.pmer._add_mapper([self.source], 
                    Map(combine(aggs)), 
                    combiner=combiner,
                    name=name)
            return PMap(source, pmer) 

        return self

    def map(self, f):
        def _map(k, v):
            yield k, f(v)

        return self._add_map(_map)

    def filter(self, f):
        def _filter(k, v):
            if f(v):
                yield k, v

        return self._add_map(_filter)

    def flat_map(self, f):
        def _flat_map(k, v):
            for vi in f(v):
                yield k, vi

        return self._add_map(_flat_map)

    def group_by(self, key, vf=lambda x: x):
        def _group_by(_key, value):
            yield key(value), vf(value)

        pm = self._add_map(_group_by).checkpoint()
        return PReduce(pm.source, pm.pmer)

    def a_group_by(self, key, vf=lambda x: x):
        def _a_group_by(_key, value):
            yield key(value), vf(value)

        # We don't checkpoint here!
        pm = self._add_map(_a_group_by)
        return ARReduce(pm)

    def sort_by(self, key):
        def _sort_by(_key, value):
            yield key(value), value

        pm = self._add_map(_sort_by).checkpoint()
        return PReduce(pm.source, pm.pmer)

    def join(self, other):
        assert isinstance(other, PBase)
        me = self.checkpoint(True)
        if isinstance(other, PMap):
            other = other.checkpoint(True)

        pmer = Polymr(me.pmer.graph.union(other.pmer.graph))
        return PJoin(me.source, pmer, other.source)

    def count(self, key=lambda x: x):
        return self.a_group_by(key, lambda v: 1) \
                .reduce(lambda k, vs: sum(vs))

    def inspect(self, prefix="", run_exit=False):
        def _inspect(k, v):
            print("{}: {}".format(prefix, v))
            yield k, v

        ins = self._add_map(_inspect)
        if run_exit:
            ins.run()
            sys.exit(0)

        return ins

    def cached(self):
        # Run the pipeline, load it into memory, and create a new graph
        results = self.checkpoint().run()
        p = Polymr().memory(list(results.read()))
        results.delete()
        return p

class ARReduce(object):
    def __init__(self, pmap):
        self.pmap = pmap

    def reduce(self, agg):
        red = Reduce(agg)
        # We add the associative aggregator to the combiner during map
        pm = self.pmap.checkpoint(True, combiner=PartialReduceCombiner(red))
        return PReduce(pm.source, pm.pmer).reduce(agg)
    
    def first(self):
        return self.reduce(lambda k, vs: next(vs))

    def sum(self):
        return self.reduce(lambda k, vs: sum(vs))

class PReduce(PBase):

    def reduce(self, f):
        new_source, pmer = self.pmer._add_reducer([self.source], KeyedReduce(f))
        return PMap(new_source, pmer)

    def unique(self, key=lambda x: x):
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
        assert isinstance(other, PBase)
        if isinstance(other, PMap):
            other = other.checkpoint(True)

        pmer = Polymr(self.pmer.graph.union(other.pmer.graph))
        return PJoin(self.source, pmer, other.source)

class PJoin(PBase):

    def __init__(self, source, pmer, right):
        super(PJoin, self).__init__(source, pmer)
        self.right = right

    def run(self):
        return self.reduce(lambda l, r: (list(l), list(r))).run()

    def reduce(self, aggregate):
        def _reduce(k, left, right):
            return aggregate(left, right)

        source, pmer = self.pmer._add_reducer([self.source, self.right], 
                KeyedInnerJoin(_reduce))
        return PMap(source, pmer)

    def left_reduce(self, aggregate):
        def _reduce(k, left, right):
            return aggregate(left, right)

        source, pmer = self.pmer._add_reducer([self.source, self.right], 
                KeyedLeftJoin(_reduce))
        return PMap(source, pmer)

class Polymr(object):
    def __init__(self, graph=None, runner=None):
        if graph is None:
            graph = Graph()

        self.graph = graph 
        if runner is None:
            runner = MTRunner

        self.runner = runner

    def memory(self, items, chunk_size=1000):
        mi = MemoryInput(list(enumerate(items)), chunk_size)
        source, ng = self.graph.add_input(mi)
        return PMap(source, Polymr(ng))

    def text(self, fname):
        source, ng = self.graph.add_input(TextInput(fname))
        return PMap(source, Polymr(ng))

    def json(self, fname):
        return self.text(fname).map(json.loads)

    def _add_mapper(self, *args, **kwargs): 
        output, ng = self.graph.add_mapper(*args, **kwargs)
        return output, Polymr(ng)

    def _add_reducer(self, *args, **kwargs): 
        output, ng = self.graph.add_reducer(*args, **kwargs)
        return output, Polymr(ng)


def combine(aggs):
    if len(aggs) == 1:
        return aggs[0].mapper

    def run(it, agg):
        return ((ki, vi) for k, v in it for ki, vi in agg.mapper(k, v))

    def _combine(k, v):
        it = iter([(k, v)])
        for agg in aggs:
            it = run(it, agg)

        return it

    return _combine

# This reinitializaes everytime
RANDOM = None
def get_rand():
    global RANDOM
    if RANDOM is None:
        RANDOM = random.Random(time.time())

    return RANDOM
