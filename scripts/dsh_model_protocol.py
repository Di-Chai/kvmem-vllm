"""Verified Qwen3.8 tokenizer-template protocol for the text-only DSH route.

Rendering always comes from the supplied tokenizer. Byte spans are accepted only
for the audited template and after the complete rendered framing is verified.
No model, CUDA context, frozen protocol library or independent text renderer is
loaded. The official installed vLLM Qwen parser handles complete tool arguments.
"""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import re
from threading import Lock
from types import SimpleNamespace


QWEN38_TEMPLATE_SHA256 = 'c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041'
_DELIMITERS = ('<|im_start|>', '<|im_end|>', '<|endoftext|>')
_FRAME = re.compile(rb'<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>\n', re.S)
_CALL = re.compile(r'<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>', re.S)
_PARAM = re.compile(r'<parameter=([^>\n]+)>(.*?)</parameter>', re.S)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _text(value):
    if isinstance(value, str):
        return value
    if value is None:
        return ''
    if isinstance(value, list) and all(isinstance(p, dict) and p.get('type') == 'text' for p in value):
        return ''.join(p['text'] for p in value)
    raise ValueError('Tokenizer protocol accepts text-only content')


def _convert(payload):
    """Preserve one rendered message index per wire message, including tools."""
    messages = [dict(role='system', content=_text(payload.get('system', '')))]
    for message in payload['messages']:
        role = message['role']
        if role not in ('user', 'assistant'):
            raise ValueError('Unsupported protocol message role')
        content = message.get('content', '')
        blocks = [dict(type='text', text=content)] if isinstance(content, str) else content
        if not isinstance(blocks, list):
            raise ValueError('Message content must be text or blocks')
        results = [b for b in blocks if b.get('type') == 'tool_result']
        if results:
            if role != 'user' or len(results) != len(blocks) or len(results) != 1:
                raise ValueError('Each wire tool result must retain its own message index')
            result = results[0]
            messages.append(dict(role='tool', tool_call_id=result['tool_use_id'],
                                 content=_text(result.get('content', ''))))
            continue
        text, thinking, calls = [], [], []
        for block in blocks:
            kind = block.get('type')
            if kind == 'text':
                text.append(block['text'])
            elif kind == 'thinking' and role == 'assistant':
                thinking.append(block['thinking'])
            elif kind == 'tool_use' and role == 'assistant':
                if not isinstance(block['input'], dict):
                    raise ValueError('Tool arguments must be an object')
                calls.append(dict(id=block['id'], type='function', function=dict(
                    name=block['name'], arguments=block['input'])))
            else:
                raise ValueError('Unsupported protocol content block')
        row = dict(role=role, content=''.join(text))
        if thinking:
            row['reasoning_content'] = ''.join(thinking)
        if calls:
            row['tool_calls'] = calls
        messages.append(row)
    tools = [dict(type='function', function=dict(name=t['name'],
             description=t.get('description', ''),
             parameters=t.get('input_schema', dict(type='object', properties={}))))
             for t in payload.get('tools', [])]
    # Delimiters in source text can impersonate complete template frames. Reject
    # these requests rather than inventing ambiguous provenance coordinates.
    serialized = _json([messages, tools])
    if any(marker in serialized for marker in _DELIMITERS):
        raise ValueError('Source content contains reserved Qwen message delimiters')
    return messages, tools


class TokenizerProtocol:
    implicit_tool_reasoning_end = True

    def __init__(self, tokenizer, directory, *, reasoning_effort='low'):
        directory = Path(directory)
        if reasoning_effort not in ('low', 'medium', 'xhigh'):
            raise ValueError('Unsupported Qwen3.8 reasoning effort')
        files = {name: (directory/name).read_bytes() for name in (
            'chat_template.jinja', 'tokenizer.json', 'tokenizer_config.json')}
        fingerprints = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
        template = files['chat_template.jinja'].decode()
        if fingerprints['chat_template.jinja'] != QWEN38_TEMPLATE_SHA256:
            raise ValueError('Unaudited Qwen3.8 chat template; refusing inferred message spans')
        if tokenizer.chat_template != template:
            raise ValueError('Loaded tokenizer template differs from the fingerprinted model files')
        # The tokenizer is loaded from these local assets by the bridge entrypoint.
        # Compare the actual loaded vocabulary with the file, not just its name.
        from tokenizers import Tokenizer
        source_vocab = Tokenizer.from_str(files['tokenizer.json'].decode()).get_vocab()
        if tokenizer.get_vocab() != source_vocab:
            raise ValueError('Loaded tokenizer vocabulary differs from the fingerprinted model files')
        self.tokenizer = tokenizer
        self.kwargs = dict(enable_thinking=True, preserve_thinking=True,
                           reasoning_effort=reasoning_effort)
        self.provenance = dict(schema_version=1, source='tokenizer.apply_chat_template',
                               files=fingerprints, chat_template_kwargs=self.kwargs,
                               tool_parser='vllm.parser.qwen3.Qwen3Parser')
        self.template = 'qwen38_tokenizer:'+hashlib.sha256(_json(self.provenance).encode()).hexdigest()
        self._parsers, self._parser_lock = OrderedDict(), Lock()

    def __call__(self, **payload):
        if payload.get('op', 'render') == 'parse':
            return self.parse(payload['text'], payload.get('tools', []))
        messages, tools = _convert(payload['request'])
        prompt = self.tokenizer.apply_chat_template(
            messages, tools=tools or None, tokenize=False,
            add_generation_prompt=True, **self.kwargs)
        data = prompt.encode()
        header = b'<|im_start|>assistant\n<think>\n'
        if not data.endswith(header):
            raise ValueError('Qwen generation header differs from the audited template')
        stop = len(data)-len(header)
        frames, cursor = [], 0
        while cursor < stop:
            match = _FRAME.match(data, cursor)
            if match is None or match.end() > stop:
                raise ValueError('Rendered message framing does not match audited template')
            frames.append(match)
            cursor = match.end()
        if cursor != stop or not frames:
            raise ValueError('Rendered system/control framing is incomplete')
        control_end = frames[0].end() if frames[0].group(1) == b'system' else 0
        if not control_end:
            # The audited template omits an empty system message for medium
            # effort without tools. Do not invent a frame or shift wire indices.
            if tools or messages[0]['content'].strip() or self.kwargs['reasoning_effort'] != 'medium':
                raise ValueError('Rendered system/control framing is incomplete')
            messages = messages[1:]
        spans, index = [], 0
        for frame in frames:
            role = frame.group(1).decode()
            if index >= len(messages):
                raise ValueError('Unexpected rendered message frame')
            message = messages[index]
            if message['role'] == 'tool':
                if role != 'user':
                    raise ValueError('Tool results lack the audited user wrapper')
                # Consecutive tool results share a user frame in this template;
                # each source message still gets its own non-overlapping span.
                position = frame.start()+len(b'<|im_start|>user')
                first = True
                while index < len(messages) and messages[index]['role'] == 'tool':
                    body = messages[index]['content'].strip().encode()
                    prefix = b'\n<tool_response>\n'
                    chunk = prefix+body+b'\n</tool_response>'
                    if data[position:position+len(chunk)] != chunk:
                        raise ValueError('Tool-result bytes differ from their rendered source')
                    begin = frame.start() if first else position
                    content_begin = position+len(prefix)
                    position += len(chunk)
                    last = index+1 == len(messages) or messages[index+1]['role'] != 'tool'
                    end = frame.end() if last else position
                    spans.append(dict(index=index, role='tool', begin=begin, end=end,
                                      content_begin=content_begin,
                                      content_end=content_begin+len(body)))
                    index += 1
                    first = False
                if data[position:frame.end()] != b'<|im_end|>\n':
                    raise ValueError('Tool-result frame has unaccounted bytes')
            else:
                if message['role'] != role:
                    raise ValueError('Rendered message roles differ from source indices')
                if role == 'user' and frame.group(2) != message['content'].strip().encode():
                    raise ValueError('User content differs from its exact rendered span')
                spans.append(dict(index=index, role=role, begin=frame.start(),
                                  end=frame.end(), content_begin=frame.start(2),
                                  content_end=frame.end(2)))
                index += 1
        if index != len(messages):
            raise ValueError('Rendered message index domain is incomplete')
        query = next((i for i in range(len(messages)-1, -1, -1)
                      if messages[i]['role'] == 'user'), None)
        return dict(prompt=prompt, messages=messages, tools=tools, spans=spans,
                    control_end=control_end, pins=[], query_index=query)

    def parse(self, text, tools):
        """Use exact reasoning boundaries and the official typed tool parser."""
        close, first_tool = text.find('</think>'), text.find('<tool_call>')
        if first_tool >= 0 and (close < 0 or first_tool < close):
            # Match the official Qwen grammar: a tool frame can end reasoning
            # implicitly, even when the model omits the explicit closing tag.
            reasoning, content = text[:first_tool], text[first_tool:]
        elif close >= 0:
            reasoning, content = text[:close], text[close+len('</think>'):]
        else:
            return dict(reasoning=text, content='', tool_calls=[])
        reasoning = reasoning.removesuffix('\n')
        content = content.lstrip('\r\n')
        if '<tool_call>' not in content:
            return dict(reasoning=reasoning, content=content, tool_calls=[])
        prefix, body = content.split('<tool_call>', 1)
        body = '<tool_call>'+body
        matches, cursor = [], 0
        while cursor < len(body):
            match = _CALL.match(body, cursor)
            if match is None:
                raise ValueError('Incomplete or malformed Qwen tool call; no tool was executed')
            arguments, position, names = match[2], 0, set()
            for argument in _PARAM.finditer(arguments):
                if arguments[position:argument.start()].strip() or argument[1] in names:
                    raise ValueError('Malformed or duplicate Qwen tool argument')
                names.add(argument[1])
                position = argument.end()
            if arguments[position:].strip():
                raise ValueError('Incomplete Qwen tool argument')
            matches.append(match)
            cursor = match.end()
            while cursor < len(body) and body[cursor].isspace():
                cursor += 1
        by_name = {t['function']['name']: t['function'] for t in tools}
        if any(m[1] not in by_name for m in matches):
            raise ValueError('Qwen emitted an unknown tool')
        key = _json(tools)
        with self._parser_lock:
            if key not in self._parsers:
                from vllm.parser.qwen3 import Qwen3Parser
                from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam
                typed = [ChatCompletionToolsParam.model_validate(tool) for tool in tools]
                parser = Qwen3Parser(self.tokenizer, typed,
                                    chat_template_kwargs=dict(enable_thinking=False))
                self._parsers[key] = (parser, typed)
                if len(self._parsers) > 4:
                    self._parsers.popitem(last=False)
            self._parsers.move_to_end(key)
            parser, typed = self._parsers[key]
            extracted = parser.extract_tool_calls_from_content(
                body, SimpleNamespace(tools=typed, tool_choice='auto'))
            calls = [call.model_dump() for call in extracted.tool_calls]
        if len(calls) != len(matches):
            raise ValueError('Official parser did not accept every complete Qwen tool frame')
        from jsonschema import SchemaError, ValidationError, validate
        for call, match in zip(calls, matches):
            if call['function']['name'] != match[1]:
                raise ValueError('Official parser changed Qwen tool order')
            arguments = json.loads(call['function']['arguments'])
            try:
                validate(arguments, by_name[match[1]].get('parameters', {}))
            except (SchemaError, ValidationError) as error:
                raise ValueError('Qwen tool arguments do not satisfy the declared schema') from error
        return dict(reasoning=reasoning, content=prefix, tool_calls=calls)
