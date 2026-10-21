"""Validate source identity without changing or re-encoding model messages."""
import hashlib
import json


def apply_query_provenance(body, prepared, *, required=False):
    proof = body.get('kvmem_query_provenance')
    if proof is None:
        if required:
            raise ValueError('DSH Query source metadata is required for this route')
        return {'status': 'legacy-role-query'}
    if not isinstance(proof, dict) or proof.get('schema_version') != 1:
        raise ValueError('Invalid DSH Query source metadata')
    if proof.get('session_id') != body.get('prompt_cache_key'):
        raise ValueError('DSH Query source belongs to another session')
    users = [(i, m) for i, m in enumerate(body['messages']) if m['role'] == 'user']
    hashes = []
    for _, message in users:
        text = message.get('content') or ''
        if isinstance(text, list):
            text = ''.join(b['text'] for b in text if b['type'] == 'text')
        hashes.append(hashlib.sha256(text.encode()).hexdigest())
    digest = hashlib.sha256(json.dumps(hashes, separators=(',', ':')).encode()).hexdigest()
    if digest != proof.get('user_messages_sha256'):
        raise ValueError('DSH Query source differs from wire user messages')
    ordinal, identity = proof.get('query_user_ordinal'), proof.get('query_message_id')
    plan, rendered, locations = prepared[1], prepared[2], prepared[6]
    if ordinal is None:
        if identity is not None:
            raise ValueError('DSH Query identity has no user message')
        plan.pop('kvmem_query_span', None)
        plan['kvmem_reselect'] = 'off'
    else:
        if type(ordinal) is not int or not 0 <= ordinal < len(users) or not isinstance(identity, str) or not identity:
            raise ValueError('Invalid DSH Query user ordinal or identity')
        wire_index = users[ordinal][0]
        index = sum(m['role'] not in ('system', 'developer') for m in body['messages'][:wire_index])
        # The frozen Qwen renderer prepends its merged system/control message.
        count = sum(m['role'] not in ('system', 'developer') for m in body['messages'])
        prefix = len(rendered['messages']) - count
        if prefix < 0 or any(m['role'] != 'system' for m in rendered['messages'][:prefix]):
            raise ValueError('DSH Query renderer changed the message index domain')
        index += prefix
        span = next((s for s in rendered['spans'] if s['index'] == index and s['role'] == 'user'), None)
        if span is None:
            raise ValueError('Real user Query is missing from the exact rendered message spans')
        keys = [(index, span[p] - span['begin']) for p in ('content_begin', 'content_end')]
        if any(k not in locations for k in keys):
            raise ValueError('Real user Query has no verified token coordinates')
        plan['kvmem_query_span'] = [locations[k] for k in keys]
    plan['kvmem_query_mode'] = 'agent'
    plan['kvmem_query_id'] = identity
    return dict(status='verified-user-source', message_id=identity, user_ordinal=ordinal,
                query_span=plan.get('kvmem_query_span'))
