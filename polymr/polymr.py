import time
import logging
import collections
import itertools
import json
import random

from .base import Mapper, Map, Reduce, Join, TextInput, MemoryInput
from .runner import MTRunner, Graph, Source

class PBase(object):
    def __init__(self, source, pmer):
        assert isinstance(source, Source)
        self.source = source
        self.pmer = pmer

    def run(self, name=None):
        name = 'tmp' if name is None else name
        logging.info("run source: %s", self.source)
        return self.pmer.runner(name, self.pmer.graph).run([self.source])

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
        
    def checkpoint(self):
        if len(self.agg) > 0:
            aggs = self.agg[:]
            name = ' -> ' .join('{}'.format(a.mapper.__name__) for a in aggs)
            name = 'Stage {}: %s => %s' % (self.source, name)
            source = self.pmer.graph.add_mapper([self.source], Map(combine(aggs)), name)
            return PMap(source, self.pmer) 

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
        return PReduce(pm.source, self.pmer)

    def group_bys(self, key, vf=lambda x: x):
        def _group_bys(_key, value):
            keys = key(value)
            for k in keys:
                yield k, vf(value)

        pm = self._add_map(_group_bys).checkpoint()
        return PReduce(pm.source, self.pmer)

    def sort_by(self, key):
        def _sort_by(_key, value):
            v = key(value)
            yield v, value

        pm = self._add_map(_sort_by).checkpoint()
        return PReduce(pm.source, self.pmer)

    def join(self, other):
        assert isinstance(other, PBase)
        left_source = self.checkpoint().source
        if isinstance(other, PMap):
            other = other.checkpoint()

        return PJoin(left_source, self.pmer, other.source)

class PReduce(PBase):

    def reduce(self, f):
        new_source = self.pmer.graph.add_reducer([self.source], Reduce(f))
        return PMap(new_source, self.pmer)

    def count(self):
        def _count(k, it):
            for i, _ in enumerate(it):
                pass

            return i + 1

        return self.reduce(_f)

    def first(self):
        return self.reduce(lambda k, vs: next(vs))

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
            other = other.checkpoint()

        return PJoin(self.source, self.pmer, other.source)

    def sum(self):
        return self.reduce(lambda k, vs: sum(vs))

class PJoin(PBase):

    def __init__(self, source, pmer, right):
        super(PJoin, self).__init__(source, pmer)
        self.right = right

    def reduce(self, aggregate):
        def _reduce(k, left, right):
            return aggregate(left, right)

        source = self.pmer.graph.add_reducer([self.source, self.right], Join(_reduce))
        return PMap(source, self.pmer)

    def cross(self):
        def _cross(left, right):
            right = list(right)
            agg = []
            for l in left:
                for r in right:
                    agg.append(l, r)

            return agg

        return self.reduce(_cross).flat_map(lambda x: iter(x))

class Polymr(object):
    def __init__(self, graph=None, runner=None):
        if graph is None:
            graph = Graph()

        self.graph = graph 
        if runner is None:
            runner = MTRunner

        self.runner = runner

    def memory(self, items):
        mi = MemoryInput(list(enumerate(items)))
        source = self.graph.add_input(mi)
        return PMap(source, self)

    def text(self, fname):
        source = self.graph.add_input(TextInput(fname))
        return PMap(source, self)

    def json(self, fname):
        return self.text(fname).map(json.loads)

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


