"""Frozen experiment-only explicit launch environment."""
import hashlib
import json
import os
from pathlib import Path
from vllm_kvmem import serve

def main():
    if os.environ.get('QUERY_STAGE_NATIVE_BINARY_JSON'):
        from vllm_kvmem import _native
        expected = json.loads(os.environ['QUERY_STAGE_NATIVE_BINARY_JSON'])
        actual = Path(_native.__file__).resolve()
        digest = hashlib.sha256(actual.read_bytes()).hexdigest()
        if actual != Path(expected['path']).resolve() or digest != expected['sha256']:
            raise RuntimeError('Frozen native extension was shadowed or changed')
        with Path(os.environ['QUERY_STAGE_NATIVE_AUDIT_OUTPUT']).open('x') as stream:
            json.dump(dict(path=str(actual), sha256=digest, passed=True,
                scope='Actual server-process import before model initialization'), stream)
    if os.environ.get('QUERY_STAGE_PREFILL_STATE_BINARY_JSON'):
        from vllm_kvmem import _prefill_state_native
        expected = json.loads(os.environ['QUERY_STAGE_PREFILL_STATE_BINARY_JSON'])
        actual = Path(_prefill_state_native.__file__).resolve()
        digest = hashlib.sha256(actual.read_bytes()).hexdigest()
        if actual != Path(expected['path']).resolve() or digest != expected['sha256']:
            raise RuntimeError('Frozen prefill state extension was shadowed or changed')
        with Path(os.environ['QUERY_STAGE_PREFILL_STATE_AUDIT_OUTPUT']).open('x') as stream:
            json.dump(dict(path=str(actual), sha256=digest, passed=True,
                scope='Actual server-process import before model initialization'), stream)
    original = serve.build_launch
    def build(*args, **kwargs):
        plan = original(*args, **kwargs)
        plan['environment'].update(json.loads(os.environ['QUERY_STAGE_EXECUTION_JSON']))
        return plan
    serve.build_launch = build
    serve.main()

if __name__ == '__main__':
    main()
