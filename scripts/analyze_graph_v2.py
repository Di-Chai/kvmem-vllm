"""Analyze frozen V2 MAB/LME answers with the original frozen evaluators.

Local scoring/reuse is the default. --external explicitly enables the unchanged
judge APIs using an already-authorized credential supplied by the caller's environment.
No inference is run and no credential is copied into the bundle or reports.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import benchmark_graph_v2 as campaign
import graph_v2_evaluation as evaluation
from graph_v2_protocol import require


def analyze(args):
    root = args.campaign.resolve()
    require(Path(__file__).resolve() == root/'frozen/analyze_graph_v2.py', 'Run the frozen analyzer')
    manifest = campaign.validate(root)
    identity = evaluation.validate_evaluation(root, manifest.get('evaluation_sha256'))
    job = next((job for job in manifest['jobs'] if job['id'] == args.job), None)
    require(job is not None and job['suite'] in ('memoryagentbench','longmemeval_s'), 'Choose a prepared MAB or LME job')
    directory = root/args.job
    dataset = directory/'qualification/dataset'
    fixture = json.loads((directory/'fixture.json').read_text())
    report = evaluation.validate_answers(dataset,fixture,allow_partial=args.allow_partial)
    # Import the frozen job's native parser for LME accepted-output auditing.
    sys.path.insert(0,str(directory/'runtime/src'))
    module = importlib.import_module('analyze_memory' if job['suite']=='memoryagentbench' else 'analyze_longmemeval')
    module.FROZEN = root/'evaluation/evaluators'
    if job['suite'] == 'memoryagentbench':
        module.PYTHON = identity['python']
        module.OFFICIAL_REPO = root/'evaluation/official'
    module.analyze(SimpleNamespace(campaign=dataset,config=directory/'profile.json',external=args.external))
    (dataset/'scoring/v2-analysis-audit.json').write_text(json.dumps(dict(report,
        external_requested=args.external,evaluation_sha256=manifest['evaluation_sha256']),indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--job',required=True,help='memoryagentbench-A/C/D or longmemeval_s-A/C/D')
    parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--external',action='store_true')
    analyze(parser.parse_args())


if __name__ == '__main__':
    main()
