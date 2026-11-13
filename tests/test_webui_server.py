"""Exercise the actual HTTP boundary without importing a GPU runtime."""
import http.client
import json
import threading
import time

import pytest

from vllm_kvmem.apps.webui.server import ConsoleServer
from vllm_kvmem.apps.webui.telemetry import Telemetry, parse_metrics


class Manager:
    active_profile = {'model': '/models/example', 'max_model_len': 16384,
                      'kvmem': {'select_budget': 2048, 'gen_budget': 2048}}
    base_url = 'http://127.0.0.1:1'
    backend_headers = {}
    def draft(self):
        return self.active_profile
    def status(self):
        return {'state': 'stopped', 'instance': 'instance-1', 'active_profile': self.active_profile}
    def preview(self, profile):
        return {'profile': profile, 'vllm_argv': ['serve'], 'environment': {}}
    def save(self, profile):
        return profile
    def start(self, profile):
        return {'state': 'loading'}
    def stop(self):
        return {'state': 'stopped'}
    def close(self):
        pass
    def logs(self, revision=None):
        return {'protocol': 1, 'text': 'engine log', 'revision': '1'}


class Chat:
    def __init__(self):
        self.cancelled = threading.Event()
        self.block = False
        self.last_client = None
    def context(self):
        return {'backend': 'vllm', 'instance': 'instance-1'}
    def stream(self, payload, request_id, instance):
        if instance != 'instance-1':
            raise ValueError('Service instance changed')
        yield b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        if self.block:
            assert self.cancelled.wait(5), 'disconnect did not cancel prefill'
        else:
            yield b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
        yield b'data: [DONE]\n\n'
    def record(self, request_id, *, client_id=None):
        self.last_client = client_id
        return {'id': request_id}
    def cancel(self, request_id, *, client_id=None):
        self.last_client = client_id
        self.cancelled.set()
        return {'cancelled': True}
    def release(self, conversation_id, *, client_id=None):
        self.last_client = client_id
        return {'released': True}
    def cancel_all(self):
        self.cancelled.set()
    def snapshot(self):
        return {'active': [], 'recent': []}


@pytest.fixture
def console(tmp_path):
    assets = tmp_path / 'static'
    (assets / 'monitor').mkdir(parents=True)
    (assets / 'monitor/index.html').write_text('<html>console</html>')
    (assets / 'chat/view').mkdir(parents=True)
    (assets / 'chat/view/index.html').write_text('<html>chat</html>')
    (tmp_path / 'private.txt').write_text('not an asset')
    manager, chat = Manager(), Chat()
    server = ConsoleServer(('127.0.0.1', 0), manager, chat=chat, static_dir=assets)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.close()
    thread.join(timeout=2)


def request(console, method, path, payload=None, headers=None):
    connection = http.client.HTTPConnection('127.0.0.1', console.server_port, timeout=5)
    values = {'Content-Type': 'application/json'}
    values.update(headers or {})
    body = json.dumps(payload) if payload is not None else None
    connection.request(method, path, body=body, headers=values)
    response = connection.getresponse()
    result = response.status, response.read(), dict(response.getheaders())
    connection.close()
    return result


def test_static_and_info_without_engine(console):
    assert request(console, 'GET', '/')[1] == b'<html>console</html>'
    assert request(console, 'GET', '/chat/view/')[1] == b'<html>chat</html>'
    code, body, headers = request(console, 'GET', '/api/launcher/info')
    info = json.loads(body)
    assert code == 200 and info['token'] == console.token
    assert info['status']['state'] == 'stopped'
    assert headers['Cache-Control'] == 'no-store'
    assert request(console, 'GET', '/api/monitor/snapshot')[0] == 200


@pytest.mark.parametrize('headers', [
    {'Host': 'attacker.example'},
    {'Origin': 'https://attacker.example'},
    {'Sec-Fetch-Site': 'cross-site'},
    {'Host': '127.0.0.1:12345'},
])
def test_reject_cross_origin_read(console, headers):
    assert request(console, 'GET', '/api/launcher/info', headers=headers)[0] == 403


def test_mutation_requires_token_and_profile(console):
    assert request(console, 'POST', '/api/launcher/start', {'profile': {}})[0] == 403
    assert request(console, 'POST', '/api/launcher/start', {'profile': {}},
                   {'X-KVMem-Token': '\u00e9'})[0] == 403
    headers = {'X-KVMem-Token': console.token}
    assert request(console, 'POST', '/api/launcher/start', {}, headers)[0] == 400
    assert request(console, 'POST', '/api/launcher/start',
                   {'profile': {'port': console.server_port}}, headers)[0] == 400
    code, body, _ = request(console, 'POST', '/api/launcher/preview',
                            {'profile': {'model': '/m'}}, headers)
    assert code == 200 and json.loads(body)['profile']['model'] == '/m'


@pytest.mark.parametrize('path', ['/monitor/../../private.txt', '/monitor/%2e%2e/%2e%2e/private.txt',
                                 '/chat/view/../../../../etc/passwd', '/.git/config', '/api/unknown'])
def test_static_paths_cannot_escape(console, path):
    assert request(console, 'GET', path)[0] == 404


def test_client_identity_is_forwarded_and_conflicts_rejected(console):
    headers = {'X-KVMem-Token': console.token, 'X-KVMem-Client': 'browser-a'}
    assert request(console, 'POST', '/api/chat/release',
                   {'conversation_id': 'conversation'}, headers)[0] == 200
    assert console.chat.last_client == 'browser-a'
    assert request(console, 'POST', '/api/chat/cancel',
                   {'id': 'request', 'client_id': 'browser-b'}, headers)[0] == 400
    assert request(console, 'GET', '/api/chat/requests/request', headers=headers)[0] == 200


def test_sse_and_instance_validation(console):
    headers = {'X-KVMem-Token': console.token, 'X-KVMem-Client': 'browser-a',
               'X-KVMem-Instance': 'instance-1', 'X-KVMem-Request': 'request-1'}
    code, body, response_headers = request(console, 'POST', '/api/chat/completions',
                                         {'conversation_id': 'conversation'}, headers)
    assert code == 200 and b'hello' in body and body.endswith(b'data: [DONE]\n\n')
    assert response_headers['Content-Type'].startswith('text/event-stream')
    headers['X-KVMem-Instance'] = 'old-instance'
    assert request(console, 'POST', '/api/chat/completions', {}, headers)[0] == 400


def test_disconnect_cancels_during_prefill(console):
    console.chat.block = True
    connection = http.client.HTTPConnection('127.0.0.1', console.server_port, timeout=3)
    connection.request('POST', '/api/chat/completions', json.dumps({'client_id': 'browser-a'}), {
        'Content-Type': 'application/json', 'X-KVMem-Token': console.token,
        'X-KVMem-Instance': 'instance-1', 'X-KVMem-Request': 'request-1'})
    response = connection.getresponse()
    assert response.readline().startswith(b'data: ')
    response.close()
    connection.close()
    assert console.chat.cancelled.wait(3)


def test_metrics_keep_unknown_values_unknown():
    assert parse_metrics('# no measurements') == {}
    assert parse_metrics('vllm:num_requests_running{model_name="m",engine="0"} 2\n'
                         'vllm:num_requests_running{model_name="m",engine="1"} 3\n'
                         'vllm:num_requests_waiting NaN\nother 100') == {'vllm:num_requests_running': 5}
    assert parse_metrics('vllm:kv_cache_usage_perc{engine="0"} 0.7\n'
                         'vllm:kv_cache_usage_perc{engine="1"} 0.7') == {}


def test_telemetry_never_presents_budgets_as_measured_usage():
    telemetry = Telemetry(Manager(), Chat())
    snapshot = telemetry.snapshot()
    assert snapshot['kvmem'] == {}
    assert snapshot['queued_count'] is None
    assert snapshot['native_metrics']['values'] == {}
    assert 'monitor_devices' not in telemetry.props()['execution']
    assert telemetry.props()['kvmem']['select_budget'] == 2048


def test_telemetry_expires_engine_data_on_stop():
    telemetry = Telemetry(Manager(), Chat())
    telemetry._instance = 'instance-1'
    telemetry._metrics_at = time.time() * 1000
    telemetry._metrics = {'vllm:num_requests_waiting': 8}
    telemetry._hardware = {'sampled_at_ms': time.time() * 1000}
    telemetry._devices = [{'uuid': 'previous-gpu'}]
    assert telemetry.snapshot()['queued_count'] is None
    assert 'monitor_devices' not in telemetry.props()['execution']


def test_finished_request_duration_does_not_keep_growing():
    chat = Chat()
    chat.snapshot = lambda: {'active': [], 'recent': [{'id': 'one', 'started_at_ms': 1000,
        'finished_at_ms': 2200, 'state': 'completed'}]}
    telemetry = Telemetry(Manager(), chat)
    assert telemetry.snapshot()['recent'][0]['elapsed_ms'] == 1200
