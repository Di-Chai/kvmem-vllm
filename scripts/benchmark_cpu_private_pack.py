"""Serial archive-thread allocation plus bounded parallel record copies.

Reuse the existing AB/BA whole-lifecycle benchmark, including admission,
allocation, hash, publication, read/verification and retirement. Only arm 8
changes its private packing implementation. Neither production defaults nor
the archive's immutable record/publication semantics change.
"""
import argparse
import json
from pathlib import Path
import shutil

import benchmark_cpu_seal_pack as base


def prepare(directory, source_root):
    base.prepare(directory, source_root)
    sources = {'benchmark_cpu_private_pack.py': Path(__file__),
               'cpu_private_record_pack.py': source_root/'scripts/cpu_private_record_pack.py',
               'source-proof/csrc/private_record_pack.hpp': source_root/'csrc/private_record_pack.hpp',
               'source-proof/tests/test_cpu_private_record_pack.py': source_root/'tests/test_cpu_private_record_pack.py'}
    for name, source in sources.items():
        destination = directory/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
    manifest = json.loads((directory/'manifest.json').read_text())
    manifest['files'].update({name:base.sha256(directory/name) for name in sources})
    manifest['scope'] = __doc__
    manifest['arm_labels'] = {'1':'Original serial allocation and serial memcpy',
                              '8':'Serial archive-thread allocation, parallel memcpy on existing pool'}
    manifest['production_integrated'] = False
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


def run(directory,pairs):
    import cpu_private_record_pack as candidate
    original = base.make_archive
    def create(directory,count,workers):
        archive = original(directory,count,workers)
        return candidate.install(archive) if workers==8 else archive
    base.make_archive = create
    try:
        base.run(directory,pairs)
    finally:
        base.make_archive = original


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run'))
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--pairs',type=int,default=12)
    args=parser.parse_args()
    if args.command=='prepare':
        prepare(args.directory.resolve(),args.source_root.resolve())
    else:
        run(args.directory.resolve(),args.pairs)
