"""Append the frozen upstream answer prefix after a Qwen assistant template."""
from array import array
import sys

POLICY = 'official_after_assistant_template_v1'
ASSISTANT = '<|im_start|>assistant\n<think>\n\n</think>\n\n'


def append_official_prefix(tokens, prefix, tokenizer):
    """Preserve every public token and retokenize only trailing whitespace.

    Like upstream RULER's model_template(task_input) + answer_prefix, this
    prefills the assistant answer. The prefix is never a generated answer or
    part of the public Query. No reference labels are accepted by this helper.
    """
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError('Missing frozen official answer_prefix')
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids
    tail, whitespace = encode(ASSISTANT), encode('\n\n')
    if not tail or tokens[-len(tail):] != tail or not whitespace or tail[-len(whitespace):] != whitespace:
        raise ValueError('Expected the unprefilled non-thinking assistant template')
    keep = len(tokens)-len(whitespace)
    suffix = encode('\n\n'+prefix)
    if tokenizer.decode(suffix, skip_special_tokens=False) != '\n\n'+prefix:
        raise ValueError('Official prefix did not survive tokenization')
    return list(tokens[:keep])+suffix, dict(policy=POLICY, retained_tokens=keep,
        replaced_whitespace_tokens=len(whitespace), appended_tokens=len(suffix)-len(whitespace))


def token_bytes(tokens):
    values = array('i', tokens)
    if sys.byteorder != 'little':
        values.byteswap()
    return values.tobytes()
