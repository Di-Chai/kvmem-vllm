"""Exercise native tool-call JSON and timing before starting official tasks.

No tool is executed. This separate short API diagnostic is not an agent score
or a performance sample; it does not alter official task generation settings.
"""
import argparse
import json
from pathlib import Path
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-base', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    payload = {'model': 'qualification', 'messages': [
        {'role': 'user', 'content': 'Call the bash tool with command printf kvmem-tool-smoke. Do not explain.'}],
        'tools': [{'type': 'function', 'function': {'name': 'bash', 'description': 'Run a shell command',
            'parameters': {'type': 'object', 'properties': {'command': {'type': 'string'}}, 'required': ['command']}}}],
        'tool_choice': 'auto', 'temperature': 0, 'max_tokens': 128,
        'chat_template_kwargs': {'enable_thinking': False}}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(args.api_base.rstrip('/') + '/chat/completions',
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json', 'x-kvmem-task-id': 'api-preflight'})
    report = {'scope': 'short non-thinking tool/metrics API diagnostic; no executed tool or task score',
              'request': payload, 'status': 'failed'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError('Preflight evidence must not be overwritten')
    try:
        with opener.open(request, timeout=180) as response:
            result = json.load(response)
        report['response'] = result
        choice = result['choices'][0]
        calls = choice['message'].get('tool_calls') or []
        if len(calls) != 1 or calls[0]['function']['name'] != 'bash':
            raise RuntimeError('Native tool parser did not produce one bash call')
        arguments = json.loads(calls[0]['function']['arguments'])
        if arguments != {'command': 'printf kvmem-tool-smoke'}:
            raise RuntimeError('Native tool parser changed tool arguments')
        metrics = result.get('metrics') or {}
        if metrics.get('generation_time_ms') is None or metrics.get('speculative_decoding') is None:
            raise RuntimeError('Native generation or speculative metrics are missing')
        report['status'] = 'passed'
    except Exception as error:
        report['error'] = repr(error)
        if isinstance(error, urllib.error.HTTPError):
            report['http_status'] = error.code
            report['http_body'] = error.read().decode(errors='replace')
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'usage': result['usage'], 'metrics': result['metrics']}))


if __name__ == '__main__':
    main()
