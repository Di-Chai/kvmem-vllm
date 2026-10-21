"""Run the existing serving launcher with a startup-only Inductor audit.

This file belongs to a frozen benchmark, not the supported serving interface.
All profile validation, compatibility setup and launch recording remain in
vllm_kvmem.serve. The diagnostic worker is the sole argv addition.
"""
from pathlib import Path


WORKER_CLASS = 'prefill_inductor_worker.SeededAuditWorker'
RUN_DIRECTORY_ENV = 'PREFILL_INDUCTOR_RUN_DIR'


def audited_build_launch(original):
    def build_launch(profile, base, run_dir, **kwargs):
        plan = original(profile, base, run_dir, **kwargs)
        argv = plan['vllm_argv']
        if any(arg == '--worker-cls' or arg.startswith('--worker-cls=') for arg in argv):
            raise ValueError('Inductor diagnostic launcher cannot replace an explicit worker class')
        # The native launcher owns run/server. The controller owns its parent,
        # where per-worker receipts sit next to server.log and resources.jsonl.
        output = Path(run_dir).expanduser().resolve().parent
        return {**plan, 'vllm_argv': [*argv, '--worker-cls', WORKER_CLASS],
                'environment': {**plan['environment'], RUN_DIRECTORY_ENV: str(output)}}

    return build_launch


def main(argv=None):
    from vllm_kvmem import serve

    original = serve.build_launch
    serve.build_launch = audited_build_launch(original)
    try:
        return serve.main(argv)
    finally:
        serve.build_launch = original


if __name__ == '__main__':
    main()
