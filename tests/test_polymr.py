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
        self.assertEquals(list(range(10, 20)), list(results))

    def test_count(self):
        res = self.items \
                .group_by(lambda x: 1, lambda x: 1) \
                .reduce(lambda k, it: sum(it)) \
                .run()

        self.assertEquals(10, next(iter(res))[1])

    def test_count_red(self):
        res = self.items \
                .count(lambda x: None) \
                .run()

        self.assertEquals((None, 10), next(iter(res)))

    def test_sum(self):
        res = self.items \
                .group_by(lambda x: 1) \
                .reduce(lambda k, it: sum(it)) \
                .run()

        self.assertEquals(sum(range(10, 20)), next(iter(res))[1])

        # Sum odds/evens
        res = self.items \
                .group_by(lambda v: v % 2) \
                    .reduce(lambda k, it: sum(it)) \
                .run()

        evens_odds = [kv[1] for kv in res]
        self.assertEquals([10+12+14+16+18, 11+13+15+17+19], evens_odds)

    def test_filter(self):
        odds = self.items \
                .filter(lambda i: i % 2 == 1) \
                .run()

        odds = list(odds)
        self.assertEquals([11, 13, 15, 17, 19], odds)

    def test_sort(self):
        res = self.items.sort_by(lambda x: -x).run()
        self.assertEquals([19,18,17,16,15,14,13,12,11,10],list(res))

    def test_reduce_join(self):
        items2 = self.polymer.memory(list(range(10)))
        res = self.items \
                .group_by(lambda x: x % 2) \
                .join(items2.group_by(lambda x: x % 2)) \
                    .reduce(lambda l, r: list(sorted(itertools.chain(l, r)))) \
                .run()

        output = list(res)
        self.assertEquals((0, [0,2,4,6,8,10,12,14,16,18]), output[0])
        self.assertEquals((1, [1,3,5,7,9,11,13,15,17,19]), output[1])

    def test_disjoint(self):
        items2 = self.polymer.memory(list(range(10))) \
                .group_by(lambda x: -x)
        output = self.items.group_by(lambda x: x) \
                .join(items2) \
                .run()
        output = [v for k,v in output]
        self.assertEquals([], output)

    def test_repartition(self):
        items2 = self.polymer.memory(list(range(10))) \
                .group_by(lambda x: -x) \
                .reduce(lambda k, vs: sum(vs))
        output = self.items.group_by(lambda x: x) \
                .join(items2) \
                .run()
        output = [v for k,v in output]
        self.assertEquals([], output)

    def test_associative_reduce(self):
        output = self.items \
                .a_group_by(lambda x: x % 2) \
                    .reduce(lambda k, vs: sum(vs)) \
                .run()

        output = list(output)
        self.assertEquals(10 + 12 + 14 + 16 + 18, output[0][1])
        self.assertEquals(11 + 13 + 15 + 17 + 19, output[1][1])
        

if __name__ == '__main__':
    unittest.main()
