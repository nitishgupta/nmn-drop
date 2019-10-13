#!/usr/bin/env python
# encoding: utf-8

import time
import multiprocessing
from textwrap import dedent


def process_chunk(d):
    """Replace this with your own function
    that processes data one line at a
    time"""

    d = d.strip() + " processed"
    time.sleep(0.001)
    return d


def grouper(n, iterable, padvalue=None):
    """grouper(3, 'abcdefg', 'x') -->
	('a','b','c'), ('d','e','f'), ('g','x','x')"""

    chunk_size = n
    return [iterable[i : i + chunk_size] for i in range(0, len(iterable), chunk_size)]

    # return izip_longest(*[iter(iterable)]*n, fillvalue=padvalue)


if __name__ == "__main__":

    # test data
    test_data = ""
    for i in range(100000):
        test_data += f"{i} test data" + "\n"

    test_data = test_data.strip()
    test_data = test_data.split("\n")

    # Create pool (p)
    p = multiprocessing.Pool(10)

    # Use 'grouper' to split test data into
    # groups you can process without using a
    # ton of RAM. You'll probably want to
    # increase the chunk size considerably
    # to something like 1000 lines per core.

    # The idea is that you replace 'test_data'
    # with a file-handle
    # e.g., testdata = open(file.txt,'rU')

    # And, you'd write to a file instead of
    # printing to the stout

    results = []
    group_num = 1
    for chunk in grouper(1000, test_data):
        result = p.map(process_chunk, chunk)
        results.extend(result)
        print(f"group done: {group_num}")
        group_num += 1
    for r in results:
        print(r)  # replace with outfile.write()
