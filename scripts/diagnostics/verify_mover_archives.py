#!/usr/bin/env python3
"""Read-only, streaming full-content comparison of MOVER archives and extracts."""
import gzip
import hashlib
import json
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/MOVER'
EXTRACT = DATA / 'extracted'
OUT = ROOT / 'outputs/mover_extract_cleanup_20261001'
CHUNK = 8 * 1024 * 1024


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while b := f.read(CHUNK):
            h.update(b)
    return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    initial = {}
    for p in EXTRACT.rglob('*'):
        assert not p.is_symlink(), p
        if p.is_file():
            s = p.stat()
            initial[str(p.relative_to(EXTRACT))] = {
                'size': s.st_size, 'mtime_ns': s.st_mtime_ns,
                'allocated_bytes': s.st_blocks * 512}
    status = dict(status='running', files_total=len(initial), files=[], archives=[],
                  started=time.time(), allocated_bytes=sum(x['allocated_bytes'] for x in initial.values()))

    def save():
        (OUT / 'archive_verification.json').write_text(json.dumps(status, indent=2)+'\n')

    save()
    seen = set()
    try:
        for name in ('EPIC_EMR.tar.gz', 'Epic_flowsheets_cleaned.tar.gz'):
            archive = DATA / name
            astat = archive.stat()
            status['active_archive'] = name
            save()
            with gzip.open(archive, 'rb') as gz:
                with tarfile.open(fileobj=gz, mode='r|') as tf:
                    for member in tf:
                        relative = Path(member.name)
                        assert not relative.is_absolute() and '..' not in relative.parts, member.name
                        if member.isdir():
                            continue
                        assert member.isfile(), member.name
                        key = str(relative)
                        assert key in initial and key not in seen, key
                        target = EXTRACT / relative
                        assert target.stat().st_size == member.size == initial[key]['size'], key
                        status['active_file'] = key
                        save()
                        packed = tf.extractfile(member)
                        hp, hl = hashlib.sha256(), hashlib.sha256()
                        with target.open('rb') as local:
                            while b := packed.read(CHUNK):
                                matching = local.read(len(b))
                                assert b == matching, key
                                hp.update(b)
                                hl.update(matching)
                            assert local.read(1) == b'', key
                        assert hp.digest() == hl.digest(), key
                        assert target.stat().st_mtime_ns == initial[key]['mtime_ns'], key
                        seen.add(key)
                        status['files'].append(dict(path=key, sha256=hp.hexdigest(), **initial[key],
                                                    archive=name, byte_equal=True))
                        save()
                        print(f"verified {len(seen)}/{len(initial)} {key}", flush=True)
                # Read the gzip footer and any trailing data to check CRC and length.
                while gz.read(CHUNK):
                    pass
            assert archive.stat().st_size == astat.st_size and archive.stat().st_mtime_ns == astat.st_mtime_ns
            status['archives'].append(dict(path=str(archive.relative_to(ROOT)), size=astat.st_size,
                                           mtime_ns=astat.st_mtime_ns, sha256=digest(archive), gzip_crc_verified=True))
            save()
        assert seen == set(initial), sorted(set(initial)-seen)
        status.update(status='verified', complete=True, finished=time.time(),
                      all_extracted_files_covered=True)
        save()
    except Exception as e:
        status.update(status='failed', complete=False, error=repr(e))
        save()
        raise


if __name__ == '__main__':
    main()
