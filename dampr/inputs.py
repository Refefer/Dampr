try:
    from urllib.request import urlopen
    from urllib.error import HTTPError
except ImportError:
    from urllib2 import urlopen, HTTPError

import glob
from contextlib import closing
import os

from .dataset import Chunker, TextLineDataset, GzipLineDataset, MemoryDataset, Dataset

class PathInput(Chunker):
    def __init__(self, path, chunk_size=64*1024**2, follow_links=True):
        self.path = path
        self.chunk_size = chunk_size
        self.follow_links = follow_links

    def chunks(self):
        if not isinstance(self.path, list):
            paths = [self.path]
        else:
            paths = self.path
        for path_glob in paths:
            for path in glob.glob(path_glob):
                if os.path.isfile(path):
                    for c in TextInput(path, self.chunk_size).chunks():
                        yield c
                else:
                    for root, dirs, files in os.walk(path, followlinks=self.follow_links):
                        for fname in files:
                            path = os.path.join(root, fname)
                            for chunk in TextInput(path, self.chunk_size).chunks():
                                yield chunk

class TextInput(Chunker):
    def __init__(self, path, chunk_size=64*1024**2):
        self.path = path
        self.chunk_size = chunk_size

    def chunks(self):
        if self.path.endswith('.gz'):
            yield GzipLineDataset(self.path)
        else: 
            file_size = os.stat(self.path).st_size
            offset = 0
            while offset < file_size:
                yield TextLineDataset(self.path, offset, offset + self.chunk_size)
                offset += self.chunk_size

class MemoryInput(Chunker):
    def __init__(self, items, partitions=50):
        self.items = items
        self.partitions = min(len(items), partitions)

    def chunks(self):
        # Memory Dataset could be zero
        if self.partitions == 0:
            yield MemoryDataset(self.items)
        else:
            chunk_size = int(len(self.items) // float(self.partitions))
            for start in range(0, len(self.items), chunk_size):
                yield MemoryDataset(self.items[start:start+chunk_size])

class UrlsInput(Chunker):
    def __init__(self, urls, skip_on_error=True):
        self.urls = urls
        self.soe = skip_on_error

    def chunks(self):
        for url in self.urls:
            yield UrlDataset(url, self.soe)

class UrlDataset(Dataset):
    def __init__(self, path, skip_on_error=True):
        self.path = path
        self.soe = skip_on_error

    def read(self):
        try:
            with closing(urlopen(self.path)) as h:
                for i, line in enumerate(h):
                    yield i, line.decode('utf-8')

        except HTTPError:
            if not self.soe:
                raise

