import shutil
import itertools
import unittest

from dampr import Dampr, BlockMapper, BlockReducer, Dataset, settings
from dampr.inputs import UrlsInput

class RangeDataset(Dataset):
    def __init__(self, n):
        self.n = n

    def read(self):
        for i in range(self.n):
            yield i, i

class DamprTest(unittest.TestCase):

    def setUp(self):
        items = list(range(10, 20))
        self.items = Dampr.memory(items, partitions=2)

    def test_identity(self):
        results = self.items.run()
        self.assertEquals(list(range(10, 20)), list(results))

    def test_map(self):
        results = self.items.map(lambda x: x + 1).run()
        self.assertEquals(list(range(11, 21)), list(results))

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
        items2 = Dampr.memory(list(range(10)))
        res = self.items \
                .group_by(lambda x: x % 2) \
                .join(items2.group_by(lambda x: x % 2)) \
                    .reduce(lambda l, r: list(sorted(itertools.chain(l, r)))) \
                .run()

        output = list(res)
        self.assertEquals((0, [0,2,4,6,8,10,12,14,16,18]), output[0])
        self.assertEquals((1, [1,3,5,7,9,11,13,15,17,19]), output[1])

    def test_disjoint(self):
        items2 = Dampr.memory(list(range(10))) \
                .group_by(lambda x: -x)
        output = self.items.group_by(lambda x: x) \
                .join(items2) \
                .run()
        output = [v for k,v in output]
        self.assertEquals([], output)

    def test_repartition(self):
        items2 = Dampr.memory(list(range(10))) \
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
        to_remove = Dampr.memory(list(range(10, 13)))
        
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

        even_ve, odd_ve = Dampr.run(even, odd)
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
        path = "/tmp/dampr_test_sink"
        sink = self.items.map(lambda x: str(x)) \
                .sink(path=path)

        output = sink.count()

        results = sorted(list(output.run()))
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

        results = sorted(list(output.run()))
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

        results = sorted(list(output.run()))
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

    def test_blocks(self):
        """
        Tests Custom Blocks
        """
        from collections import defaultdict
        import heapq
        class TopKMapper(BlockMapper):
            def __init__(self, k):
                self.k = k

            def start(self):
                self.heap = []

            def add(self, _k, lc):
                heapq.heappush(self.heap, (lc[1], lc[0]))
                if len(self.heap) > self.k:
                    heapq.heappop(self.heap)

                return iter([])

            def finish(self):
                for cl in self.heap:
                    yield 1, cl

        class TopKReducer(BlockReducer):
            def __init__(self, k):
                self.k = k

            def start(self):
                pass

            def add(self, k, it):
                for count, letter in heapq.nlargest(self.k, it):
                    yield letter, (letter, count)

        word = Dampr.memory(["supercalifragilisticexpialidociousa"])
        letter_counts = word.flat_map(lambda w: list(w)).count()

        topk = letter_counts \
                .custom_mapper(TopKMapper(2)) \
                .custom_reducer(TopKReducer(2))

        results = sorted(list(topk.run()))
        self.assertEquals(results, [('a', 4), ('i', 7)])

    def test_stream_blocks(self):
        """
        Tests stream blocks
        """
        import heapq
        def map_topk(it):
            heap = []
            for symbol, count in it:
                heapq.heappush(heap, (count, symbol))
                if len(heap) > 2:
                    heapq.heappop(heap)

            return ((1, x) for x in heap)

        def reduce_topk(it):
            counts = (v for k, vit in it for v in vit)
            for count, symbol in heapq.nlargest(2, counts):
                yield symbol, count

        word = Dampr.memory(["supercalifragilisticexpialidociousa"])
        letter_counts = word.flat_map(lambda w: list(w)).count()

        topk = letter_counts \
                .partition_map(map_topk) \
                .partition_reduce(reduce_topk)

        results = sorted(list(topk.run()))
        self.assertEquals(results, [('a', 4), ('i', 7)])

    def test_cross_map(self):
        """
        Tests the caae where only values are sent during crosses
        """
        item_counts = self.items.count()

        total = self.items \
                .a_group_by(lambda x: 1, lambda x: 1) \
                    .sum() \
                .map(lambda x: float(x[1]))

        results = item_counts \
                .cross_right(total, lambda ic, t: (ic[0], ic[1] / t)) \
                .read()

        results.sort()

        self.assertEquals(results, [(i , 1 / float(10)) for i in range(10, 20)])

    def test_len(self):
        """
        Tests the number of items in a collection.
        """

        self.assertEquals(self.items.len().read(), [10])
        self.assertEquals(Dampr.memory([]).len().read(), [0])

    def test_read_input(self):
        """
        Tests that custom taps work as expected.
        """
        class RangeDataset(Dataset):
            def __init__(self, n):
                self.n = n

            def read(self):
                for i in range(self.n):
                    yield i, i

        results = Dampr.read_input(RangeDataset(5), RangeDataset(10)) \
                .fold_by(lambda x: 1, lambda x, y: x + y) \
                .read()

        self.assertEqual(results[0][1], sum(range(5)) + sum(range(10)))

    def test_read_input(self):
        """
        Tests that custom taps work as expected.
        """
        
        results = Dampr.read_input(RangeDataset(5), RangeDataset(10)) \
                .fold_by(lambda x: 1, lambda x, y: x + y) \
                .read()

        self.assertEqual(results[0][1], sum(range(5)) + sum(range(10)))

    def test_read_url(self):
        """
        Tests that we can read urls.
        """
        results = Dampr.read_input(UrlsInput(["http://www.example.com"])) \
                .filter(lambda line: 'h1' in line) \
                .map(lambda line: line.strip()) \
                .read()

        self.assertEqual(results, ['<h1>Example Domain</h1>'])

    def test_file_glob(self):
        """
        Tests that we can read file globs
        """
        import os
        files = []
        for i in range(10):
            path = os.path.join('/tmp', '_test_dampr_{}'.format(i))
            with open(path, 'w') as out:
                out.write(str(i))

            files.append(path)

        results = Dampr.text("/tmp/_test_dampr_[135]") \
                .map(int) \
                .fold_by(lambda x: 1, lambda x,y: x + y) \
                .read()

        self.assertEqual(results, [(1, 1 + 3 + 5)])

        for fname in files:
            os.unlink(fname)

    def test_top_k(self):
        """
        Tests getting the top k items
        """

        word = Dampr.memory(["supercalifragilisticexpialidociousa"])
        topk = word.flat_map(lambda w: list(w)).count() \
                .topk(5, lambda x: x[1])

        results = sorted(list(topk.run()))
        self.assertEquals(results, [('a', 4), ('c', 3), ('i', 7), ('l', 3), ('s', 3)])

    def test_file_link(self):
        """
        Tests that we can read file globs
        """
        import os
        dirnames = []
        for i in range(10):
            dirname = os.path.join('/tmp', '_test_dampr_dir_{}'.format(i))
            if os.path.isdir(dirname):
                shutil.rmtree(dirname)

            os.makedirs(dirname)
            dirnames.append(dirname)

            fname = os.path.join(dirname, 'foo')
            with open(fname, 'w') as out:
                out.write(str(i))

        # Symlink into a new directory
        base = '/tmp/_dampr_test_link'
        if os.path.isdir(base):
            shutil.rmtree(base)

        dirnames.append(base)
        os.makedirs(base)

        for i in (1, 3, 5):
            os.symlink(dirnames[i], os.path.join(base, os.path.basename(dirnames[i])))

        # Yields nothing!
        results = Dampr.text(base) \
                .map(int) \
                .fold_by(lambda x: 1, lambda x,y: x + y) \
                .read()

        self.assertEqual(results, [])


        # Yields something!
        results = Dampr.text(base, followlinks=True) \
                .map(int) \
                .fold_by(lambda x: 1, lambda x,y: x + y) \
                .read()

        self.assertEqual(results, [(1, 1 + 3 + 5)])

        for d in dirnames:
            shutil.rmtree(d)

    def _test_concat(self):
        """
        Tests concatenating K datasets into a new Dampr
        """

        word1 = Dampr.memory("abcdefg")
        word1.concat(Dampr.memory("hijklmn"))

        results = sorted(list(word1.run()))
        self.assertEquals(results, list('abcdefghijklmn'))

    def test_cross_set(self):
        """
        Tests that we can join a smaller dataset on the map pass
        """

        right = Dampr.memory(range(5, 15))

        results = self.items.cross_set(right, lambda x, r: (x, x in r)) \
            .filter(lambda x: x[1]) \
            .map(lambda x: x[0]) \
            .read()

        results.sort()

        self.assertEqual(results, list(range(10,15)))

if __name__ == '__main__':
    unittest.main()
