import multiprocessing
from multiprocessing import Queue

from .dataset import *
from .base import *

class StageRunner(object):
    def __init__(self, max_procs):
        self.max_procs = max_procs

    def launch_process(self, p_id, input_q, output_q):
        raise NotImplementedError()

    def run(self, job_queue):
        input_q = Queue()
        output_q = Queue()
        total_jobs = 0
        for job in job_queue:
            input_q.put(job)
            total_jobs += 1

        logging.debug("Total tasks: %s", total_jobs)

        # launch jobs
        processes = []
        for pid in range(self.max_procs):
            processes.append(self.launch_process(pid, input_q, output_q))
            processes[-1].start()
            # Add sentinel
            input_q.put(None)

        # Assign tasks via forking
        finished = []
        while len(finished) < self.max_procs:
            payload = output_q.get()
            finished.append(payload)

        # Cleanup
        for p in processes:
            p.join()

        return finished

class MapStageRunner(StageRunner):

    def __init__(self, max_procs, fs, n_partitions, mapper, options):
        super(MapStageRunner, self).__init__(max_procs)
        self.fs = fs
        self.n_partitions = n_partitions
        self.mapper = mapper
        self.options = options

    def simple_map(self, in_q, out_q, fs):
        w_id = os.getpid()

        # Default job, nothing special
        if self.options.get('memory', False):
            writer_cls = SortedMemoryWriter
        else:
            writer_cls = SortedDiskWriter

        dw = CSDatasetWriter(fs, Splitter(), self.n_partitions, writer_cls=writer_cls)

        dw.start()
        while True:
            job = in_q.get()
            if job is None: 
                break

            m_id, main, supplemental = job
            logging.debug("Mapper %i: Computing map: %i", w_id, m_id)
            for k, v in self.mapper.mapper.map(main, *supplemental):
                dw.add_record(k, v)

        out_q.put(dw.finished())
        logging.debug("Mapper: %i: Finished", w_id)

    def medium_map(self, in_q, out_q, combiner, shuffler, fs):
        """
        Runs a more fine grained map/combine/shuffler
        """
        w_id = os.getpid()
        if self.options.get('memory', False):
            dw = SortedMemoryWriter(fs)
        else:
            dw = SortedDiskWriter(fs)

        # Do we have a map side partial reducer?
        binop = self.options.get('binop')
        if callable(binop):
            # Zero buffer means all reductions will happen reduce side
            dw = MaxMemoryWriter(ReducedWriter(dw, binop))
        else:
            dw = MaxMemoryWriter(dw)

        # run the jobz
        dw.start()
        while True:
            job = in_q.get()
            if job is None: 
                break

            m_id, main, supplemental = job
        
            logging.debug("Mapper %i: Computing map: %i", w_id, m_id)
            for k, v in self.mapper.mapper.map(main, *supplemental):
                dw.add_record(k, v)

        sources = dw.finished()[0]

        if len(sources) > 1:
            logging.debug("Combining outputs: found %i files", len(sources))
            logging.debug("Combining: %s", sources)
            combined_stream = combiner.combine(sources)
        elif len(sources) == 1:
            combined_stream = sources[0]
        else:
            combined_stream = EmptyDataset()

        results = shuffler.shuffle(fs, [combined_stream])

        out_q.put(results)
        logging.debug("Mapper: %i: Finished Map-Combine", w_id)

    def launch_process(self, p_id, input_q, output_q):
        fs = self.fs.get_worker('map/{}'.format(p_id))

        if self.mapper.combiner is None and self.mapper.shuffler is None:

            p = multiprocessing.Process(target=self.simple_map,
                args=(input_q, output_q, fs))
        else:
            c = NoopCombiner() if self.mapper.combiner is None else self.mapper.combiner
            if self.options.get('memory', False):
                writer_cls = lambda fs: MaxMemoryWriter(UnorderedDiskWriter(fs))
            else:
                writer_cls = lambda fs: MaxMemoryWriter(UnorderedMemoryWriter(fs))

            s = DefaultShuffler(self.n_partitions, Splitter(), writer_cls)
            o = self.mapper.options
            p = multiprocessing.Process(target=self.medium_map,
                args=(input_q, output_q, c, s, fs))

        return p

class SinkStageRunner(StageRunner):
    def __init__(self, max_procs, mapper, path):
        super(SinkStageRunner, self).__init__(max_procs)
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except OSError as e:
                if e.errno != 17:
                    raise

        self.path = path
        self.mapper = mapper

    def sink(self, input_q, out_q):
        """
        Writes line delimited items as a sink
        """
        w_id = os.getpid()

        finished = [{0: []}]
        while True:
            job = input_q.get()
            if job is None: break
            m_id, main, supplemental = job
            dw = SinkWriter(self.mapper.path, m_id)
            dw.start()
            logging.debug("Sink %i: Computing map: %i", w_id, m_id)
            for k, v in self.mapper.mapper.map(main, *supplemental):
                dw.add_record(k, v)

            finished.append(dw.finished())
        
        # concatentate them all
        for i in range(1, len(finished)):
            finished[0][0].extend(finished[i][0])

        out_q.put(finished[0])
        logging.debug("Sink: %i: Finished", w_id)

    def launch_process(self, pid, input_q, output_q):
        p = multiprocessing.Process(target=self.sink,
            args=(input_q, output_q))

        return p

class CombinerStageRunner(StageRunner):
    """
    Merges files together to reduce their on disk presence
    """
    def __init__(self, max_procs, fs, combiner, options, per_tid=False):
        super(CombinerStageRunner, self).__init__(max_procs)
        self.fs = fs
        self.combiner = combiner
        self.options = options
        self.per_tid = per_tid

    def combine_per_key(self, input_q, output_q, fs):
        w_id = os.getpid()
        finished = []
        while True:
            job = input_q.get()
            if job is None: break
            t_id, datasets = job

            if self.options.get('memory', False):
                dw = ContiguousMemoryWriter(fs)
            else:
                dw = ContiguousDiskWriter(fs)

            dw.start()
            for k,v in self.combiner.combine(datasets):
                dw.add_record(k,v)

            for d in datasets:
                d.delete()

            finished.append((t_id, dw.finished()[0]))

        output_q.put(finished)

    def combine(self, input_q, output_q, fs):
        w_id = os.getpid()

        if self.options.get('memory', False):
            dw = ContiguousMemoryWriter(fs)
        else:
            dw = ContiguousDiskWriter(fs)

        dw.start()
        while True:
            job = input_q.get()
            if job is None: break
            _t_id, datasets = job
            for k,v in self.combiner.combine(datasets):
                dw.add_record(k,v)

            for d in datasets:
                d.delete()

        output_q.put(dw.finished()[0])

    def launch_process(self, p_id, input_q, output_q):
        fs = self.fs.get_worker('merge/{}'.format(p_id))

        f = self.combine_per_key if self.per_tid else self.combine
        p = multiprocessing.Process(target=f,
            args=(input_q, output_q, fs))

        return p

class ReduceStageRunner(StageRunner):
    def __init__(self, max_procs, fs, reducer, options):
        super(ReduceStageRunner, self).__init__(max_procs)
        self.fs = fs
        self.reducer = reducer
        self.options = options

    def reduce(self, input_q, out_q, dw):
        w_id = os.getpid()
        dw.start()
        while True:
            job = input_q.get()
            if job is None: break

            r_id, datasets = job
            logging.debug("Reducer %i: Computing reduce: %i", w_id, r_id)
            for k, v in self.reducer.reducer.reduce(*datasets):
                dw.add_record(k, v)

        out_q.put(dw.finished())
        logging.debug("Reducer: %i: Finished", w_id)

    def launch_process(self, p_id, input_q, output_q):
        fs = self.fs.get_worker('red/{}'.format(p_id))

        if self.options.get('memory', False):
            dw = ContiguousMemoryWriter(fs)
        else:
            dw = ContiguousDiskWriter(fs)

        m = multiprocessing.Process(target=self.reduce,
            args=(input_q, output_q, dw))

        return m

