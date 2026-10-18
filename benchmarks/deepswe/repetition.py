"""Conservative, opt-in termination evidence for identical Bash loops."""
from collections import deque
import hashlib
import json


def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',', ':'))


def expected_stop_cancellation(event, stopped):
    if not stopped or event.get('at',0)<stopped['at']:return False
    if event.get('event')=='cancelled' and event.get('error_type')=='CancelledError':return True
    # Compatibility with the original bridge's untyped exception record.
    error=event.get('error')
    return error=='' or (isinstance(error,str) and error.startswith(('Cancelled by cancel scope','Cancelled via cancel scope')))


class BashRepeatGuard:
    def __init__(self, threshold=32):
        if threshold < 2:raise ValueError('Repetition threshold must be at least two')
        self.threshold=threshold
        self.calls=deque()
        self.results={}
        self.seen=set()
        self.signature=None
        self.streak=0
        self.first_trigger=None

    def line(self, line):
        record=json.loads(line)
        for block in record.get('message',{}).get('content',[]):
            if not isinstance(block,dict):continue
            if record.get('type')=='assistant' and block.get('type')=='tool_use':
                if block['id'] not in self.seen:
                    self.seen.add(block['id']);self.calls.append(block)
            elif record.get('type')=='user' and block.get('type')=='tool_result':
                self.results[block['tool_use_id']]=block
        # Consume in call order, even when parallel tool results arrive in a
        # different order. Every intervening call breaks a Bash-only streak.
        while self.calls and self.calls[0]['id'] in self.results:
            call=self.calls.popleft();result=self.results.pop(call['id'])
            eligible=call['name']=='Bash' and not result.get('is_error') and bool(result.get('content'))
            signature=(canonical(call['input']),canonical(result['content'])) if eligible else None
            self.streak=self.streak+1 if signature is not None and signature==self.signature else int(eligible)
            self.signature=signature
            if self.streak>=self.threshold and self.first_trigger is None:
                self.first_trigger={'tool':'Bash','consecutive_identical_successful_calls':self.streak,
                    'last_tool_use_id':call['id'],
                    'input_sha256':hashlib.sha256(signature[0].encode()).hexdigest(),
                    'result_sha256':hashlib.sha256(signature[1].encode()).hexdigest()}
