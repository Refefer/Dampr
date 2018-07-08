import itertools
import unittest

from polymr import Polymr

class PolymrTest(unittest.TestCase):
    def setUp(self):
        items = list(range(10, 20))
        self.polymer = Polymr()
        self.items = self.polymer.memory(items)

    def test_identity(self):
        results = self.items.run()
        self.assertEquals(list(range(10, 20)), [kv[1] for kv in results[0]])

    def test_count(self):
        res = self.items \
                .group_by(lambda x: 1, lambda x: 1) \
                .reduce(lambda k, it: sum(it)) \
                .run()

        self.assertEquals(10, next(iter(res[0]))[1])

    def test_count_red(self):
        res = self.items \
                .group_by(lambda x: 1, lambda x: 1) \
                .count() \
                .run()

        self.assertEquals(10, next(iter(res[0]))[1])

    def test_sum(self):
        res = self.items \
                .group_by(lambda x: 1) \
                .reduce(lambda k, it: sum(it)) \
                .run()

        self.assertEquals(sum(range(10, 20)), next(iter(res[0]))[1])

        # Sum odds/evens
        res = self.items \
                .group_by(lambda v: v % 2) \
                    .reduce(lambda k, it: sum(it)) \
                .run()

        evens_odds = [kv[1] for kv in res[0]]
        self.assertEquals([10+12+14+16+18, 11+13+15+17+19], evens_odds)

    def test_filter(self):
        odds = self.items \
                .filter(lambda i: i % 2 == 1) \
                .run()

        odds = [kv[1] for kv in odds[0]]
        self.assertEquals([11, 13, 15, 17, 19], odds)

    def test_sort(self):
        res = self.items.sort_by(lambda x: -x).run()
        items = list([kv[1] for kv in res[0]])
        self.assertEquals([19,18,17,16,15,14,13,12,11,10],items)

    def test_reduce_join(self):
        items2 = self.polymer.memory(list(range(10)))
        res = self.items \
                .group_by(lambda x: x % 2) \
                .join(items2.group_by(lambda x: x % 2)) \
                    .reduce(lambda l, r: list(sorted(itertools.chain(l, r)))) \
                .run()

        output = list(res[0])
        self.assertEquals([0,2,4,6,8,10,12,14,16,18], output[0][1])
        self.assertEquals([1,3,5,7,9,11,13,15,17,19], output[1][1])

    def test_disjoint(self):
        items2 = self.polymer.memory(list(range(10))) \
                .group_by(lambda x: -x)
        output = self.items.group_by(lambda x: x) \
                .join(items2) \
                .run()
        output = [v for k,v in output[0]]
        self.assertEquals([], output)

    def test_repartition(self):
        items2 = self.polymer.memory(list(range(10))) \
                .group_by(lambda x: -x) \
                .reduce(lambda k, vs: sum(vs))
        output = self.items.group_by(lambda x: x) \
                .join(items2) \
                .run()
        output = [v for k,v in output[0]]
        self.assertEquals([], output)

if __name__ == '__main__':
    unittest.main()
