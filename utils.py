from base import *

def filter_p(predicate):
    def _f(key, value):
        if predicate(key, value):
            yield key, value

    return Map(_f)

def dedup(f=lambda x: x):
    def _dedup(key, value):
        seen = set()
        agg = []
        for v in value:
            fv = f(v)
            if fv not in seen:
                seen.add(fv)
                agg.append(v)

        return agg

    return Reduce(_dedup)

def first():
    return Reduce(lambda k, v: next(v))

def swap():
    def _f(k, v):
        yield v, k

    return Map(_f)

