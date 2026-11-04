"""Task-source Query provenance for the pinned Harbor Terminus adapter.

Only the trusted agent loop may classify its observations as terminal feedback.
Message text, roles and model tokens are never rewritten. Unknown transcript
edits fail closed; a real user update needs an explicit source identity.
"""
import hashlib
import inspect
import json
from pathlib import Path


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def user_hashes(messages):
    result = []
    for message in messages:
        if message['role'] != 'user':
            continue
        content = message.get('content') or ''
        if isinstance(content, list):
            if any(part.get('type') != 'text' for part in content):
                raise ValueError('Terminal Query requires exact textual user messages')
            content = ''.join(part['text'] for part in content)
        if not isinstance(content, str):
            raise ValueError('Terminal Query requires exact textual user messages')
        result.append(digest(content))
    return result


class TaskQuery:
    def __init__(self, session_id, task_id, instruction, initial_prompt):
        if not all(isinstance(x, str) and x for x in (session_id, task_id, instruction, initial_prompt)):
            raise ValueError('Terminal Query needs an explicit task source and session')
        if instruction not in initial_prompt:
            raise ValueError('Initial prompt does not contain the verified task instruction')
        self.session_id = session_id
        self.task_id = task_id
        self.instruction_sha256 = digest(instruction)
        self.initial_hash = digest(initial_prompt)
        self.seen = []
        self.query_ordinal = None
        self.query_id = None
        self.pending_user = None

    def real_user(self, text, source_id):
        """Called by a trusted human-message producer, never inferred from role."""
        if not isinstance(text, str) or not text or not isinstance(source_id, str) or not source_id:
            raise ValueError('A real user update needs text and a distinct source identity')
        if self.pending_user is not None:
            raise ValueError('A real user update is already pending')
        identity = digest(self.session_id + '\0user\0' + source_id)
        if identity == self.query_id:
            raise ValueError('A new user update must have a new source identity')
        self.pending_user = digest(text), identity

    def proof(self, messages):
        hashes = user_hashes(messages)
        if not hashes:
            raise ValueError('Missing terminal task message')
        # Retries may resend the last exact wire request. No implicit rewinds,
        # unknown human insertions or changed task messages are permitted.
        if hashes == self.seen:
            pass
        elif hashes[:-1] != self.seen:
            raise ValueError('Terminal Query history changed without source metadata')
        elif not self.seen:
            if hashes[0] != self.initial_hash:
                raise ValueError('Wire task message differs from verified initial prompt')
            self.query_ordinal = 0
            self.query_id = digest(self.session_id + '\0task\0' + self.task_id + '\0' + self.instruction_sha256)
        elif self.pending_user is not None:
            expected, identity = self.pending_user
            if hashes[-1] != expected:
                raise ValueError('Real user update differs from its source metadata')
            self.query_ordinal, self.query_id = len(hashes) - 1, identity
            self.pending_user = None
        self.seen = hashes
        return dict(schema_version=1, session_id=self.session_id,
                    user_messages_sha256=digest(json.dumps(hashes, separators=(',', ':'))),
                    query_user_ordinal=self.query_ordinal, query_message_id=self.query_id,
                    source='terminalbench-task', task_id=self.task_id,
                    instruction_sha256=self.instruction_sha256)


def register_user_query(chat, text, source_id):
    """Register a genuine new human message before passing it to Chat.chat()."""
    source = getattr(chat, '_kvmem_task_query', None)
    if source is None:
        raise ValueError('Chat has no verified Terminal-Bench task source')
    source.real_user(text, source_id)


def install(task_sources, audit_path, expected_sources=None):
    from harbor.agents.terminus_2.terminus_2 import Terminus2
    from harbor.llms.chat import Chat
    from harbor.llms.base import OutputLengthExceededError
    from harbor.models.task.task import strip_canary

    actual = {Path(inspect.getfile(cls)).name: digest(Path(inspect.getfile(cls)).read_text())
              for cls in (Terminus2, Chat)}
    if expected_sources is not None and actual != expected_sources:
        raise ValueError('Pinned Harbor Query adapter sources changed')
    if getattr(Chat, '_kvmem_task_query_installed', False):
        raise RuntimeError('Terminal Query adapter installed twice')
    Chat._kvmem_task_query_installed = True
    sources = {str(Path(path).resolve()): value for path, value in task_sources.items()}
    audit_path = Path(audit_path)

    def audit(**value):
        with audit_path.open('a') as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + '\n')

    original_loop, original_chat = Terminus2._run_agent_loop, Chat.chat

    async def loop(self, initial_prompt, chat, original_instruction=''):
        directory = Path(self._session.environment.environment_dir).resolve().parent
        source = sources.get(str(directory))
        path = directory / 'instruction.md'
        if source is None or digest(path.read_text()) != source['instruction_sha256']:
            raise ValueError('Terminal task is absent from the frozen instruction manifest')
        if original_instruction != strip_canary(path.read_text()):
            raise ValueError('Agent instruction differs from the frozen task source')
        if self.options.enable_summarize:
            raise ValueError('Terminal Query provenance requires complete unsummarized history')
        chat._kvmem_task_query = TaskQuery(self._session_id, source['task_id'], original_instruction, initial_prompt)
        audit(event='task_query_bound', session_id=self._session_id, **source)
        return await original_loop(self, initial_prompt, chat, original_instruction)

    async def call(self, prompt, logging_path=None, **kwargs):
        source = getattr(self, '_kvmem_task_query', None)
        if source is not None:
            checkpoint = (source.seen, source.query_ordinal, source.query_id, source.pending_user)
            messages = [*self.messages, dict(role='user', content=prompt)]
            proof = source.proof(messages)
            expected_history = source.seen[:-1]
            try:
                extra = dict(kwargs.get('extra_body') or {})
                if 'kvmem_query_provenance' in extra or ('prompt_cache_key' in extra and extra['prompt_cache_key'] != source.session_id):
                    raise ValueError('Conflicting Terminal Query transport metadata')
                extra.update(prompt_cache_key=source.session_id, kvmem_query_provenance=proof)
                kwargs['extra_body'] = extra
                audit(event='query_source', session_id=source.session_id, **{
                    key: proof[key] for key in ('query_user_ordinal', 'query_message_id', 'user_messages_sha256')})
                return await original_chat(self, prompt, logging_path=logging_path, **kwargs)
            except OutputLengthExceededError:
                # The pinned Terminus handler commits this prompt and its
                # truncated answer before asking for a shorter response.
                raise
            except BaseException:
                # Chat commits messages only after a successful model call.
                # A failed nested recovery call must not become a phantom
                # user message when Terminus returns to the outer retry.
                if user_hashes(self.messages) != expected_history:
                    raise ValueError('Terminal Query failed call changed committed history')
                source.seen, source.query_ordinal, source.query_id, source.pending_user = checkpoint
                audit(event='query_attempt_rolled_back', session_id=source.session_id)
                raise
        return await original_chat(self, prompt, logging_path=logging_path, **kwargs)

    Terminus2._run_agent_loop, Chat.chat = loop, call
    audit(event='installed', policy='verified-task-query-v1', sources=actual)
    return actual
