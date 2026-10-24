import pytest


def test_parser_service_refuses_visible_cuda_before_importing_any_model_dependency(monkeypatch):
    from diagnostics.dsh_parser_service import create_app
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    with pytest.raises(RuntimeError,match='hide CUDA'):
        create_app('/nonexistent/no-model-is-loaded')


def test_parser_payload_is_bounded_and_rejects_invalid_tool_shapes():
    from diagnostics.dsh_parser_service import ParseInput
    from pydantic import ValidationError
    for value in [dict(text='x'*2097153,tools=[]),dict(text='',tools=[{}]*65),dict(text='',tools=['invalid'])]:
        with pytest.raises(ValidationError):ParseInput.model_validate(value)
