Basic Benchmarks
---

This attempts to benchmark a baseline versus a Dask and Dampr implementation.  While fairly simplistic, it attempts to show how Dampr can scale better than Dask's Bag implementation on even fairly easy tasks.

Note: I am unable to run the benchmark against Dask on the 500x benchmark due to OOM on my laptop, so only Dampr and the baseline are compared.

We use the Shakespeare dataset as our corpus: it weighs in at a small 5.3 mb and has 23903 unique words.  To scale up our computation, we duplicate the data X times, as indicated by the log.
