#!/usr/bin/env python3
"""Compare sparse NPY cold reads with byte-identical dense sample references.

The reference is a sample of the same rows, not the former full-file physical
placement. This complements the pre-replacement original/candidate warm test.
"""
import json
import mmap
import os
from pathlib import Path
import sys
import time
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/storage_rebuild_20261001'


def main():
    queue = json.loads((OUT/'queue_status.json').read_text())
    assert queue['status'] == 'completed'
    records = []
    with tempfile.TemporaryDirectory(prefix='cold-read-', dir=OUT) as temp:
        for item in queue['records']:
            if not item['accepted'] or item['original_allocated_bytes'] < 100*1024**2:
                continue
            path = ROOT/item['path']
            source = np.load(path, mmap_mode='r')
            n = min(32768, len(source))
            start = (len(source)//2//32768)*32768
            if start+n > len(source):
                start = len(source)-n
            reference = Path(temp)/'dense.npy'
            np.save(reference, np.array(source[start:start+n], copy=True))
            with reference.open('rb') as f:
                os.fsync(f.fileno())
            dense = np.load(reference, mmap_mode='r')
            assert np.array_equal(np.asarray(source[start:start+n]).view(np.uint8), dense.view(np.uint8))
            for mode in ('sequential', 'random'):
                rng = np.random.default_rng(42)
                ids = [slice(i, min(n, i+4096)) for i in range(0, n, 4096)] if mode == 'sequential' else [rng.integers(0, n, 4096) for _ in range(8)]
                times = {'dense_sample': [], 'sparse': []}
                for repeat in range(7):
                    order = [('dense_sample', dense, reference, 0), ('sparse', source, path, start)]
                    if repeat % 2:
                        order.reverse()
                    for label, array, file, base in order:
                        array._mmap.madvise(mmap.MADV_DONTNEED)
                        offset = int(array.offset)+base*array.shape[1]*array.dtype.itemsize
                        low = offset//4096*4096
                        high = ((offset+n*array.shape[1]*array.dtype.itemsize+4095)//4096)*4096
                        with file.open('rb') as fd:
                            os.posix_fadvise(fd.fileno(), low, high-low, os.POSIX_FADV_DONTNEED)
                        began = time.perf_counter()
                        for index in ids:
                            if isinstance(index, slice):
                                value = np.array(array[base+index.start:base+index.stop], copy=True)
                            else:
                                value = np.array(array[base+index], copy=True)
                            float(value.flat[0])
                        times[label].append(time.perf_counter()-began)
                record = dict(path=item['path'], mode=mode, sample_rows=n,
                              timing_seconds=times, ratio=float(np.median(times['sparse'])/np.median(times['dense_sample'])))
                records.append(record)
                print(json.dumps({k:record[k] for k in ('path','mode','ratio')}), flush=True)
            del dense, source
            reference.unlink()
    result = dict(complete=True, records=records,
                  scope='Cold mmap sequential/random reads on fixed real row regions against identical dense sample references; seven alternating repeats; not the original full-file disk placement',
                  max_ratio=max(r['ratio'] for r in records))
    (OUT/'cold_read_benchmark.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
