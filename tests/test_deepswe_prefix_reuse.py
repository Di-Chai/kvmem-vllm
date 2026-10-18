"""Tool transport equivalence must preserve tokens without masking real edits."""
import copy
import importlib.util
from pathlib import Path
import sys

import pytest


spec = importlib.util.spec_from_file_location('deepswe_prefix_bridge',
    Path(__file__).resolve().parents[1] / 'benchmarks/deepswe/claude_bridge.py')
bridge = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bridge
spec.loader.exec_module(bridge)


def message(arguments, name='Edit', call_id='call-1'):
    return {'role': 'assistant', 'content': '', 'reasoning_content': 'reason',
            'tool_calls': [{'id': call_id, 'type': 'function',
                           'function': {'name': name, 'arguments': arguments}}]}


@pytest.mark.parametrize('name,left,right', [
    ('Edit', '{"old_string":"x","new_string":"y"}',
             '{"replace_all":false,"new_string":"y","old_string":"x"}'),
    ('Bash', '{"command":"true", "timeout":120000.0}',
             '{"timeout":1.2e5,"command":"true"}'),
    ('TaskUpdate', '{"taskId":"1", "status":"completed"}',
                   '{"status":"completed","taskId":"1"}'),
    ('Bash', '{"x":0.0}', '{"x":-0}'),
])
def test_equivalent_tool_transports_have_identical_identity_without_mutation(name, left, right):
    original = message(left, name)
    saved = copy.deepcopy(original)
    assert bridge.message_identity(original) == bridge.message_identity(message(right, name))
    assert original == saved


@pytest.mark.parametrize('name,left,right', [
    ('Edit', '{"replace_all":false}', '{"replace_all":true}'),
    ('Edit', '{}', '{"replace_all":0}'),
    ('Other', '{}', '{"replace_all":false}'),
    ('Edit', '{"old_string":"\\\\u0000"}', '{"old_string":"\\u0000"}'),
    ('Bash', '{"command":" x"}', '{"command":"x"}'),
    ('Bash', '{"x":9007199254740993}', '{"x":9007199254740992.0}'),
    ('Bash', '{"x":1,"x":2}', '{"x":2}'),
    ('Bash', '{"x":NaN}', '{ "x":NaN}'),
    ('Bash', '{"x":[1,2]}', '{"x":[2,1]}'),
])
def test_actual_edits_and_ambiguous_json_remain_distinct(name, left, right):
    assert bridge.message_identity(message(left, name)) != bridge.message_identity(message(right, name))


def test_call_identity_reasoning_and_content_remain_part_of_match():
    first = message('{}')
    for change in (message('{}', call_id='other'), first | {'reasoning_content': 'changed'},
                   first | {'content': 'changed'}, message('{}', name='Bash')):
        assert bridge.message_identity(first) != bridge.message_identity(change)


def test_planner_appends_only_new_tool_result_and_retains_exact_output_ids():
    # Character tokenizer makes any accidental old-prefix re-encoding visible.
    class Tokenizer:
        def convert_tokens_to_ids(self, token): return 999
        def encode(self, text, **kw): return list(text.encode())
    old = {'role': 'user', 'content': 'question'}
    assistant = message('{"old_string":"x"}')
    transported = message('{"old_string":"x","replace_all":false}')
    rendered = {'prompt': 'PREFIX-NEW', 'control_end': 0, 'tools': [],
                'messages': [old, transported, {'role':'tool','content':'NEW'}],
                'spans': [{'index':2, 'role':'tool', 'begin':7, 'end':10,
                           'content_begin':7, 'content_end':10}], 'pins': [], 'query_index':-1}
    planner = bridge.Planner(lambda **kw: rendered, Tokenizer())
    state = bridge.Trajectory(messages=[bridge.message_identity(old)],
        control=bridge.canonical({'prefix':'', 'tools':[]}), tokens=[100, 101],
        output=[901, 902, 999], assistant=bridge.message_identity(assistant))
    prepared = planner.prepare({}, state)
    assert prepared[5] is True
    assert prepared[0] == [100, 101, 901, 902, 999] + list(b'\nNEW')
    rendered['messages'][1] = message('{"old_string":"different"}')
    changed = planner.prepare({}, state)
    assert changed[5] is False and changed[0] == list(b'PREFIX-NEW')


@pytest.fixture(params=('none','NFC'))
def boundary_tokenizer(request):
    from tokenizers import AddedToken, Tokenizer, models, normalizers, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    backend=Tokenizer(models.BPE(unk_token='<unk>'))
    if request.param=='NFC':backend.normalizer=normalizers.NFC()
    backend.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    special=[AddedToken(s,special=True,normalized=False) for s in ('<unk>','<|im_start|>','<|im_end|>')]
    backend.train_from_iterator(['hello hell helping testing tested test 测试汉字🙂 café aaaaa ababab '*10],
        trainers.BpeTrainer(vocab_size=400,special_tokens=special,initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return PreTrainedTokenizerFast(tokenizer_object=backend)


@pytest.mark.parametrize('text',[
    'hello helping tested café 🙂',
    '<|im_start|>user\n测试汉字🙂 café cafe\u0301 helping a\u0308<|im_end|>\n<|im_start|>assistant\n',
    'hello<|im_start|><|im_end|><|im_start|>user\n<|im_end|>\n',
    'hello<|im_start|>user\n<|im_en',
    '',
])
def test_every_character_boundary_matches_exact_prefix_tokenization(boundary_tokenizer,text):
    encoded=bridge.PrefixTokens(boundary_tokenizer,text)
    assert encoded.ids==boundary_tokenizer.encode(text,add_special_tokens=False)
    for end in range(len(text)+1):
        prefix=text[:end]
        assert encoded.count(len(prefix.encode()))==len(boundary_tokenizer.encode(prefix,add_special_tokens=False))


def test_delimiters_with_stripping_or_normalization_are_not_safe_anchors(boundary_tokenizer):
    from tokenizers import AddedToken
    for marker in ('<|im_start|>','<|im_end|>'):
        boundary_tokenizer.add_tokens([AddedToken(marker,special=True,normalized=True,lstrip=True)])
    text='hello <|im_start|>user\nhello <|im_end|>\n'
    encoded=bridge.PrefixTokens(boundary_tokenizer,text)
    assert encoded.positions==[0]
    for end in range(len(text)+1):
        assert encoded.count(end)==len(boundary_tokenizer.encode(text[:end],add_special_tokens=False))


def test_message_coordinate_work_is_linear_and_duplicate_points_are_cached(boundary_tokenizer):
    class CountingTokenizer:
        def __init__(self, tokenizer):self.tokenizer=tokenizer;self.characters=0
        def __getattr__(self,name):return getattr(self.tokenizer,name)
        def __call__(self,text,**kw):self.characters+=len(text);return self.tokenizer(text,**kw)
        def encode(self,text,**kw):self.characters+=len(text);return self.tokenizer.encode(text,**kw)
    tokenizer=CountingTokenizer(boundary_tokenizer)
    parts=['<|im_start|>user\n'+('hello test café 🙂 '*80)+'<|im_end|>\n' for _ in range(80)]
    text=''.join(parts);encoded=bridge.PrefixTokens(tokenizer,text)
    begin=0
    for part in parts:
        for end in (0,len('<|im_start|>user\n'.encode()),len(part.encode())-len('<|im_end|>\n'),len(part.encode())):
            point=begin+end
            value=encoded.count(point);cost=tokenizer.characters
            assert encoded.count(point)==value and tokenizer.characters==cost
        begin+=len(part.encode())
    assert tokenizer.characters < 3*len(text)
