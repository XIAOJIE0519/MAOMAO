#!/usr/bin/env python3
"""Rebuild NPY allocation losslessly; preserve the public dense/mmap interface.

An unverified candidate never replaces its source. Zero filesystem blocks are
represented by holes, without changing dtype, shape, header, indexing or bytes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / 'outputs/storage_rebuild_20261001'
PAGE = 4096
CHUNK = 32 * 1024**2


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def benchmark(original, candidate):
    old = np.load(original, mmap_mode='r')
    new = np.load(candidate, mmap_mode='r')
    assert old.shape == new.shape and old.dtype == new.dtype
    batch = min(4096, len(old))
    rng = np.random.default_rng(42)
    starts = rng.integers(0, max(1, len(old)-batch), 24)
    requests = {
        'sequential_4096': [slice(int(i), int(i)+batch) for i in starts],
        'random_4096': [rng.integers(0, len(old), batch) for _ in range(24)],
        'current_token_column': [slice(int(i), int(i)+batch) for i in starts],
    }
    results = {}
    for name, ids in requests.items():
        for index in ids:
            assert np.array_equal(old[index].view(np.uint8), new[index].view(np.uint8))
        runs = {'original': [], 'candidate': []}
        # Warm both paths, alternate order; timing deliberately excludes fitting.
        for repeat in range(9):
            order = [('original', old), ('candidate', new)]
            if repeat % 2:
                order.reverse()
            for label, array in order:
                began = time.perf_counter()
                for index in ids:
                    value = np.array(array[index, 0] if name == 'current_token_column'
                                     else array[index], copy=True)
                    float(value.flat[0])
                runs[label].append(time.perf_counter()-began)
        results[name] = {
            'original_seconds': runs['original'], 'candidate_seconds': runs['candidate'],
            'median_ratio': float(np.median(runs['candidate'])/np.median(runs['original'])),
        }
    return results


def matching_cold_benchmark(original, candidate):
    import mmap
    old = np.load(original, mmap_mode='r')
    new = np.load(candidate, mmap_mode='r')
    n = min(32768, len(old))
    base = (len(old)//2//32768)*32768
    if base+n > len(old):
        base = len(old)-n
    rng = np.random.default_rng(42)
    result = {}
    for mode in ('sequential', 'random'):
        ids = [slice(base+i, base+min(n, i+4096)) for i in range(0, n, 4096)] if mode == 'sequential' else [base+rng.integers(0, n, 4096) for _ in range(8)]
        runs = {'original': [], 'candidate': []}
        for repeat in range(5):
            order = [('original', old, original), ('candidate', new, candidate)]
            if repeat % 2:
                order.reverse()
            for label, array, path in order:
                array._mmap.madvise(mmap.MADV_DONTNEED)
                offset = int(array.offset)+base*array.shape[1]*array.dtype.itemsize
                low = offset//PAGE*PAGE
                high = ((offset+n*array.shape[1]*array.dtype.itemsize+PAGE-1)//PAGE)*PAGE
                with path.open('rb') as fd:
                    os.posix_fadvise(fd.fileno(), low, high-low, os.POSIX_FADV_DONTNEED)
                began = time.perf_counter()
                for index in ids:
                    value = np.array(array[index], copy=True)
                    float(value.flat[0])
                runs[label].append(time.perf_counter()-began)
        result[mode] = dict(original_seconds=runs['original'], candidate_seconds=runs['candidate'],
                            median_ratio=float(np.median(runs['candidate'])/np.median(runs['original'])))
    return result


def restore_dense(path, original_sha):
    st = path.stat()
    candidate = path.with_suffix('.dense-restore.npy')
    assert not candidate.exists()
    h = hashlib.sha256()
    began = time.monotonic()
    with path.open('rb') as source, candidate.open('xb') as dest:
        os.posix_fallocate(dest.fileno(), 0, st.st_size)
        offset = 0
        while block := source.read(CHUNK):
            h.update(block)
            dest.write(block)
            offset += len(block)
            if offset % (1024**3) < CHUNK:
                print(f'restore {path.name}: {offset/1024**3:.1f}/{st.st_size/1024**3:.1f} GiB', flush=True)
        dest.flush()
        os.fsync(dest.fileno())
    assert h.hexdigest() == original_sha == digest(candidate)
    timings = benchmark(path, candidate)
    cold = matching_cold_benchmark(path, candidate)
    os.chmod(candidate, stat.S_IMODE(st.st_mode))
    os.utime(candidate, ns=(st.st_atime_ns, st.st_mtime_ns))
    candidate.replace(path)
    return dict(path=str(path.relative_to(ROOT)), restored_original_dense_representation=True,
                complete_file_byte_equality=True, original_sha256=original_sha,
                warm_timings=timings, cold_matching_full_file_timings=cold,
                elapsed_seconds=time.monotonic()-began,
                final_allocated_bytes=path.stat().st_blocks*512)


def rebuild(path):
    st = path.stat()
    assert path.is_file() and not path.is_symlink() and st.st_nlink == 1
    candidate = path.with_suffix('.sparse-rebuild.npy')
    assert not candidate.exists(), f'Existing candidate requires inspection: {candidate}'
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or proc.name == str(os.getpid()):
            continue
        try:
            descriptors = list((proc/'fd').iterdir())
        except OSError:
            continue
        for descriptor in descriptors:
            try:
                opened = descriptor.stat()
            except OSError:
                continue
            assert (opened.st_dev, opened.st_ino) != (st.st_dev, st.st_ino), f'Matrix open in PID {proc.name}'
    original_sha = hashlib.sha256()
    began = time.monotonic()
    zero_bytes = 0
    with path.open('rb') as source, candidate.open('xb') as dest:
        offset = 0
        while block := source.read(CHUNK):
            original_sha.update(block)
            full = len(block)//PAGE
            pages = np.frombuffer(block, np.uint8, count=full*PAGE).reshape(full, PAGE)
            nonzero = np.any(pages != 0, axis=1)
            edges = np.flatnonzero(np.r_[False, nonzero, False][1:] !=
                                   np.r_[False, nonzero, False][:-1])
            view = memoryview(block)
            for start, end in zip(edges[::2], edges[1::2]):
                dest.seek(offset+int(start)*PAGE)
                dest.write(view[int(start)*PAGE:int(end)*PAGE])
            if full*PAGE < len(block):
                dest.seek(offset+full*PAGE)
                dest.write(view[full*PAGE:])
            zero_bytes += int((~nonzero).sum())*PAGE
            offset += len(block)
            if offset % (1024**3) < CHUNK:
                print(f'{path.name}: {offset/1024**3:.1f}/{st.st_size/1024**3:.1f} GiB rebuilt', flush=True)
        dest.truncate(st.st_size)
        dest.flush()
        os.fsync(dest.fileno())
    expected = original_sha.hexdigest()
    assert digest(candidate) == expected, 'Full-file bytes differ; source retained'
    assert (path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns) == (st.st_ino, st.st_size, st.st_mtime_ns)
    timings = benchmark(path, candidate)
    allocated = candidate.stat().st_blocks*512
    # Conservative gate. Full byte equality proves every downstream numerical
    # input unchanged; timing is a workload sample, never a universal guarantee.
    accepted = allocated < st.st_blocks*512 and all(x['median_ratio'] <= 1.10 for x in timings.values())
    record = dict(path=str(path.relative_to(ROOT)), original_sha256=expected,
                  complete_file_byte_equality=True, zero_bytes=zero_bytes,
                  original_allocated_bytes=st.st_blocks*512,
                  candidate_allocated_bytes=allocated, timings=timings,
                  timing_scope='Warm alternating native NPY/mmap sequential, random and column reads; median, nine repeats,24 batches',
                  accepted=accepted, elapsed_seconds=time.monotonic()-began)
    if accepted:
        os.chmod(candidate, stat.S_IMODE(st.st_mode))
        os.utime(candidate, ns=(st.st_atime_ns, st.st_mtime_ns))
        # Atomic path replacement after byte and timing checks; original blocks
        # are released only here. No model/result bytes are rewritten.
        candidate.replace(path)
        record['released_bytes'] = st.st_blocks*512-path.stat().st_blocks*512
    else:
        candidate.unlink()
        record['released_bytes'] = 0
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=Path)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--restore-dense', action='store_true')
    args = parser.parse_args()
    AUDIT.mkdir(exist_ok=True, parents=True)
    base = ROOT/'outputs/final_experiment_results_20260923/classical_full_scale'
    if args.restore_dense:
        prior = json.loads((AUDIT/'queue_status.json').read_text())
        status = dict(status='running', pid=os.getpid(), files_total=sum(r['accepted'] for r in prior['records']), records=[])
        proof = AUDIT/'dense_restore_queue_status.json'
        save(proof, status)
        for record in prior['records']:
            if not record['accepted']:
                continue
            path = ROOT/record['path']
            status['active_path'] = record['path']
            save(proof, status)
            status['records'].append(restore_dense(path, record['original_sha256']))
            save(proof, status)
        status.update(status='completed', active_path=None)
        save(proof, status)
        return
    assert bool(args.path) != args.all
    paths = sorted(base.rglob('*_X.npy'), key=lambda p: p.stat().st_size) if args.all else [args.path.resolve()]
    status = dict(status='running', pid=os.getpid(), files_total=len(paths), records=[])
    save(AUDIT/'queue_status.json', status)
    for path in paths:
        assert path.is_relative_to(base) and path.name.endswith('_X.npy')
        proof = AUDIT/(str(path.relative_to(ROOT)).replace('/', '__')+'.json')
        if proof.exists():
            previous = json.loads(proof.read_text())
            if previous['accepted'] and digest(path) == previous['original_sha256']:
                status['records'].append(previous)
                save(AUDIT/'queue_status.json', status)
                continue
        status['active_path'] = str(path.relative_to(ROOT))
        save(AUDIT/'queue_status.json', status)
        result = rebuild(path)
        result['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save(proof, result)
        status['records'].append(result)
        save(AUDIT/'queue_status.json', status)
        print(json.dumps({k:result[k] for k in ('path','accepted','released_bytes','elapsed_seconds')}), flush=True)
    status.update(status='completed', active_path=None, finished_utc=datetime.now(timezone.utc).isoformat())
    save(AUDIT/'queue_status.json', status)


if __name__ == '__main__':
    main()
