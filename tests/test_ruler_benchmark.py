"""Public question boundaries must survive chat wrapping and tokenization."""
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def benchmark(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('benchmark_ruler')


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert messages[0]['role'] == 'user' and len(messages) == 1
        assert kwargs['enable_thinking'] is True
        return '<user>\n' + messages[0]['content'] + '\n</user><assistant><think>\n'

    def __call__(self, text, **kwargs):
        assert kwargs == dict(add_special_tokens=False, return_offsets_mapping=True)
        return dict(input_ids=[ord(c) for c in text],
                    offset_mapping=[(i, i + 1) for i in range(len(text))])

    def decode(self, ids, **kwargs):
        return ''.join(chr(i) for i in ids)


def test_full_context_is_preserved_and_query_excludes_history(benchmark):
    original = 'question-like history\nQuestion: old question?\nMore history\nQuestion: actual question?'
    tokenizer = CharacterTokenizer()
    ids, query, decoded, rendered = benchmark.render_case(tokenizer, 'qa_1', original)
    assert original in rendered
    assert decoded == 'Question: actual question?'
    assert tokenizer.decode(ids[slice(*query)]) == decoded
    assert query[0] > rendered.index('More history')
    assert query[1] < rendered.index('</user>')


def test_changed_chat_input_is_rejected(benchmark):
    class TruncatingTokenizer(CharacterTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return '<user>Question: remaining question?</user><assistant><think>\n'
    with pytest.raises(ValueError, match='preserve'):
        benchmark.render_case(TruncatingTokenizer(), 'qa_1', 'full original history\nQuestion: actual question?')


def test_chat_template_must_start_in_thinking(benchmark):
    class NonThinkingTokenizer(CharacterTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]['content'] + '\n<think>\n</think>\n'
    with pytest.raises(ValueError, match='open-thinking'):
        benchmark.render_case(NonThinkingTokenizer(), 'qa_1', 'history\nQuestion: actual question?')


def test_token_boundary_cannot_silently_include_history(benchmark):
    class BrokenTokenizer(CharacterTokenizer):
        def __call__(self, text, **kwargs):
            encoded = super().__call__(text, **kwargs)
            at = text.index('\nQuestion:') - 1
            encoded['offset_mapping'][at] = (at, len(text))
            return encoded
    with pytest.raises(ValueError, match='include history'):
        benchmark.render_case(BrokenTokenizer(), 'qa_1', 'historyX\nQuestion: actual question?')


def test_result_handles_pressure_reports_without_query_differences(benchmark):
    row = dict(id='qa_1-0000', task='qa_1', index=0, served_sha256='frozen', outputs=['Paris'])
    response = dict(text='thinking\n</think>\nParis', output_ids=[1, 2, 3], seconds=12,
                    ttft_seconds=11, finish_reason='stop', response_id='response-1')
    pressure = dict(phase='pressure', propagation_applied=False)
    evidence = dict(pressure=[pressure, pressure],
                    graph=[pressure, pressure, dict(phase='query', added_blocks=191)])
    def metric(predictions, references):
        assert predictions == ['Paris'] and references == [['Paris']]
        return 100.0
    scorer = ({'qa': dict(metric_fn=metric)}, lambda text, config: text.strip())
    result = benchmark.summarize_result(row, 'owner', 262100, response, evidence,
                                       scorer, starts_in_thinking=True)
    assert result['graph_added_blocks'] == 191
    assert result['pressure_count'] == 2
    assert result['official_case_score'] == 100.0
    assert result['completion_tokens'] == 3 and result['thinking_complete'] is True
    # Do not hide a genuinely incomplete semantic Query report with a default.
    del evidence['graph'][-1]['added_blocks']
    with pytest.raises(KeyError, match='added_blocks'):
        benchmark.summarize_result(row, 'owner', 262100, response, evidence,
                                   scorer, starts_in_thinking=True)
