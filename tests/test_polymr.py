import shutil
import itertools
import unittest

from polymr import Polymr

class PolymrTest(unittest.TestCase):

    def setUp(self):
        items = list(range(10, 20))
        self.items = Polymr.memory(items, partitions=2)

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
        items2 = Polymr.memory(list(range(10)))
        res = self.items \
                .group_by(lambda x: x % 2) \
                .join(items2.group_by(lambda x: x % 2)) \
                    .reduce(lambda l, r: list(sorted(itertools.chain(l, r)))) \
                .run()

        output = list(res)
        self.assertEquals((0, [0,2,4,6,8,10,12,14,16,18]), output[0])
        self.assertEquals((1, [1,3,5,7,9,11,13,15,17,19]), output[1])

    def test_disjoint(self):
        items2 = Polymr.memory(list(range(10))) \
                .group_by(lambda x: -x)
        output = self.items.group_by(lambda x: x) \
                .join(items2) \
                .run()
        output = [v for k,v in output]
        self.assertEquals([], output)

    def test_repartition(self):
        items2 = Polymr.memory(list(range(10))) \
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
                    .reduce(lambda x,y: x + y) \
                .run()

        output = list(output)
        self.assertEquals(10 + 12 + 14 + 16 + 18, output[0][1])
        self.assertEquals(11 + 13 + 15 + 17 + 19, output[1][1])
        
    def test_left_join(self):
        to_remove = Polymr.memory(list(range(10, 13)))
        
        output = self.items.group_by(lambda x: x) \
                .join(to_remove.group_by(lambda x: x)) \
                    .left_reduce(lambda l, r: (list(l), list(r))) \
                .filter(lambda llrs: len(llrs[1][1]) == 0) \
                .map(lambda llrs: llrs[1][0][0]) \
                .sort_by(lambda x: x) \
                .run()

        output = list(output)
        self.assertEquals(list(range(13,20)), output)

    def test_combine(self):
        even = self.items.filter(lambda x: x % 2 == 0)
        odd  = self.items.filter(lambda x: x % 2 == 1)

        even_ve, odd_ve = Polymr.run(even, odd)
        self.assertEquals([10,12,14,16,18], list(even_ve))
        self.assertEquals([11,13,15,17,19], list(odd_ve))

    def test_reduce_many(self):
        even = self.items.filter(lambda x: x % 2 == 0)
        odd  = self.items.filter(lambda x: x % 2 == 1)

        def cross(x, y):
            y = list(y)
            for xi in x:
                for yi in y:
                    yield xi * yi

        results = even.group_by(lambda x: 1) \
            .join(odd.group_by(lambda x: 1)) \
                .reduce(cross, many=True) \
            .run().read()

        results = sorted(results)
        e = [10,12,14,16,18]
        o = [11,13,15,17,19]
        expected = sorted([(1, ei * oi) for ei in e for oi in o])
        self.assertEquals(results, expected)

    def test_fold_by(self):
        output = self.items \
                .fold_by(lambda x: 1, 
                        value=lambda x: x % 2, 
                        binop=lambda x, y: x + y)

        results = list(output.run())
        self.assertEquals([(1, 5)], results)

    def test_empty_map(self):
        """
        Test that empty mrcs writes don't blow up
        """
        output = self.items \
                .sample(0.0) \
                .fold_by(lambda x: 1, 
                        value=lambda x: x % 2, 
                        binop=lambda x, y: x + y)

        results = list(output.run())
        self.assertEquals([], results)

    def test_sink(self):
        """
        Test Sinking to disk works and doesn't delete on cleanup
        """
        path = "/tmp/polymr_test_sink"
        sink = self.items.map(lambda x: str(x)) \
                .sink(path=path)

        output = sink.count()

        results = list(output.run())
        self.assertEquals([('{}'.format(i), 1) for i in range(10, 20)], results)

        shutil.rmtree(path)

    def test_cached(self):
        """
        Tests caching stages works
        """
        sink = self.items.map(lambda x: str(x)) \
                .cached()

        res = sink.run()
        output = sink.count()

        results = list(output.run())
        self.assertEquals([(str(i), 1) for i in range(10, 20)], results)

    def test_cross_join(self):
        """
        Tests cross product
        """
        total = self.items \
                .a_group_by(lambda x: 1) \
                    .sum()

        output = self.items \
                .cross_right(total, lambda v1, v2: round(v1 / float(v2[1]), 4)) \
                .sort_by(lambda x: x)

        results = list(output.run())
        count = sum(range(10, 20))
        expected = [round(i / float(count), 4) for i in range(10, 20)]
        self.assertEquals(expected, results)

    def test_cross_join_multi(self):
        """
        Tests cross product of multiple values
        """
        output = self.items \
                .cross_left(self.items, lambda v1, v2: v1 * v2)

        results = sorted(list(output.run()))
        expected = sorted([i * k for i in range(10, 20) for k in range(10, 20)])
        self.assertEquals(expected, results)

if __name__ == '__main__':
    unittest.main()
