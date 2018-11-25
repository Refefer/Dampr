Dampr - Pure Python Data Processing
===

Dampr is intended for use as single machine data processing: it's natively out of core, supports map and reduce side joins, associative reduce combiners, and provides a high level interface for constructing Dataflow DAGs.

It's reasonably fast, easy to get started, and scales linearly by core.  It has no external dependencies, making it extremely lightweight and easy to install.  It has reasonable REPL support for data analysis, though there are better tools for the job for it.

Features
---

* Self-Contained: No external dependencies and simple to install
* High-Level API: Easy computation
* Out-Of-Core: Scales up to 100s of GB to TBs of data.  No need to worry about Out of Memory errors!
* Reasonably Fast: Linearly scales to number of cores on the machine
* Powerful: Provides a number of advanced joins and other functions for complex workflows

Setup
---

```
pip install dampr
```

or

```
python setup.py install
```

API
---
[docs/dampr/index.html](http://htmlpreview.github.io/?https://github.com/Refefer/Dampr/blob/master/docs/dampr/index.html)

Examples
---

Look at the `examples` directory for more complete examples.

Similarly, the tests are intended to be fairly readable as well.  You can view them in the `tests` directory.

## Example - WC

```python
import sys 

from dampr import Dampr

def main(fname):

    wc = Dampr.text(fname) \
            .map(lambda v: len(v.split())) \
            .a_group_by(lambda x: 1) \
            .sum()

    results = wc.run("word-count")
    for k, v in results:
        print("Word Count:", v)

    results.delete()

if __name__ == '__main__':
    main(sys.argv[1])
```

Why not Dask for data processing?
---
Dask is great!  I'd highly recommend it for fast analytics and datasets which don't need complex joins!

However.

Dask is really intended for in-memory computation and more analytics processing via interfaces like DataFrames.  While it does have a reasonable `bag` implementation for data processing, it's missing some important features such as joins across large datasets.  I have routinely run into OOM errors when processing datasets larger than memory when trying more complicated processes.

In that sense, Dampr is attempting to bridge that gap of complex data processing on a single machine and heavy-weight systems, geared toward ease of use.

Why not PySpark for data processing?
---
PySpark is great!  I'd highly recommend it for extremely large datasets and cluster computation!

However.

PySpark requires large amounts of setup to really get going.  It's the antithesis of "light-weight" and really geared for large scale production deployments.  I personally don't like it for proof of concepts or one-offs; it requires just a bit too much tuning to get what you need.


