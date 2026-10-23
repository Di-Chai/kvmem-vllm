"""CPU-only post-hoc evidence audit; gold never enters a model request.

Round-trip the exact submitted IDs before mapping character matches to token
positions. Selection coverage describes stored KV positions, not a new raw-text
forward; complete coverage alone does not prove preserved hidden-state quality.
"""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import time
import unicodedata

import numpy as np
from tokenizers import Tokenizer

import ruler_protocol as protocol


class PromptOffsets:
    """Map original byte-BPE IDs without re-encoding Unicode-normalized text."""
    def __init__(self, tokenizer):
        self.tokenizer=tokenizer
        values=list(range(33,127))+list(range(161,173))+list(range(174,256))
        codes=list(values)
        missing=0
        for value in range(256):
            if value not in values:
                values.append(value)
                codes.append(256+missing)
                missing+=1
        # GPT-2's byte alphabet has 188 directly represented bytes.
        self.alphabet={chr(code):value for value,code in zip(values,codes)}
        self.added=tokenizer.get_added_tokens_decoder()
        self.cache={}

    def piece(self, token):
        if token not in self.cache:
            self.cache[token]=(self.tokenizer.decode([token],skip_special_tokens=False).encode('utf-8')
                if token in self.added else bytes(self.alphabet[c] for c in self.tokenizer.id_to_token(token)))
        return self.cache[token]

    def positions(self, ids, text):
        pieces=[self.piece(token) for token in ids]
        if b''.join(pieces)!=text.encode('utf-8'):
            raise ValueError('Original ID byte stream does not match decoded prompt')
        starts=[];ends=[];offset=0
        for piece in pieces:
            starts.append(offset);offset+=len(piece);ends.append(offset)
        ascii_text=text.isascii()
        def tokens(begin,end):
            lo=begin if ascii_text else len(text[:begin].encode('utf-8'))
            hi=end if ascii_text else len(text[:end].encode('utf-8'))
            return bisect_right(ends,lo),bisect_left(starts,hi)
        return tokens


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_ranges(ranges):
    result = []
    for start, length in sorted(ranges):
        end = start + length
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def coverage(begin, end, ranges):
    length = sum(max(0, min(end, hi) - max(begin, lo)) for lo, hi in compact_ranges(ranges))
    return dict(tokens=end-begin, selected_tokens=length,
                state='full' if length == end-begin else 'partial' if length else 'none')


def occurrences(text, value, *, word=False):
    pattern = (r'(?<!\w)' + re.escape(value) + r'(?!\w)') if word else re.escape(value)
    return [(m.start(), m.end()) for m in re.finditer(pattern, text, re.IGNORECASE)]


def support_sources(upstream, rows):
    directory=upstream/'scripts/data/synthetic/json'
    result={}
    wanted={(row['task'],row['question'].removeprefix('Question: ')) for row in rows}
    for article in read(directory/'squad.json')['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                if ('qa_1',qa['question']) not in wanted:
                    continue
                result.setdefault(('qa_1',qa['question']),[]).append(dict(
                    title=article['title'],text=paragraph['context'],sentences=[]))
    for qa in read(directory/'hotpotqa.json'):
        if ('qa_2',qa['question']) not in wanted:
            continue
        documents={title:sentences for title,sentences in qa['context']}
        titles=list(dict.fromkeys(title for title,index in qa['supporting_facts']))
        result[('qa_2',qa['question'])]=[dict(title=title,text=''.join(documents[title]),
            sentences=[documents[title][index] for source,index in qa['supporting_facts']
                       if source==title and 0<=index<len(documents[title])])
            for title in titles]
    return result


def audit_case(row, result, evidence, ids, tokenizer, mapper, supports):
    text = tokenizer.decode(ids, skip_special_tokens=False)
    tokens=mapper.positions(ids,text)

    query = evidence['query']
    ranges = query['selected_history_ranges']
    merged = compact_ranges(ranges)
    selected_ids = [token for begin, end in merged for token in ids[begin:end]]
    if len(selected_ids) != query['selected_tokens']:
        raise ValueError('Selected history token count mismatch')
    selected_texts = [tokenizer.decode(ids[begin:end], skip_special_tokens=False)
                      for begin, end in merged]
    unique_refs = list(dict.fromkeys(row['outputs']))
    refs = []
    for gold in unique_refs:
        found = occurrences(text, gold, word=row['task'] == 'cwe')
        spots = []
        for begin, end in found:
            lo, hi = tokens(begin, end)
            item = dict(char_span=[begin,end], token_span=[lo,hi],
                        query=coverage(lo,hi,ranges))
            if row['task'].startswith('niah_'):
                line_begin = text.rfind('One of the special magic ', max(0,begin-220), begin)
                if line_begin < 0:
                    raise ValueError('Missing NIAH key/value sentence: ' + row['id'])
                prefix = text[line_begin:begin]
                match = re.fullmatch(r'One of the special magic (?:numbers|uuids) for (.+?) is: ', prefix)
                if not match:
                    raise ValueError('Unexpected NIAH key/value prefix: ' + repr(prefix))
                kb, ke = tokens(line_begin+match.start(1),line_begin+match.end(1))
                line_end = end + int(text[end:end+1] == '.')
                sb, se = tokens(line_begin,line_end)
                pieces = [tokenizer.decode(ids[max(sb,a):min(se,b)],skip_special_tokens=False)
                          for a,b in merged if max(sb,a)<min(se,b)]
                item.update(key=match[1], key_token_span=[kb,ke],
                    key_query=coverage(kb,ke,ranges), sentence_token_span=[sb,se],
                    sentence_query=coverage(sb,se,ranges), sentence=text[line_begin:line_end],
                    selected_sentence_pieces=pieces,
                    pressure=[dict(at=e['history_begin'],value=coverage(lo,hi,e['selected_history_ranges'])['state'],
                        key=coverage(kb,ke,e['selected_history_ranges'])['state'])
                        for e in evidence['pressure'] if e['history_begin']>=hi])
            elif row['task'].startswith('qa_'):
                item['context'] = text[max(0,begin-180):min(len(text),end+180)]
            spots.append(item)
        refs.append(dict(value=gold, returned_in_final=gold.lower() in result['answer'].lower(),
            mentioned_in_thinking=gold.lower() in result['text'].split('</think>',1)[0].lower(),
            occurrences=len(found), full_selected_occurrences=sum(s['query']['state']=='full' for s in spots),
            partial_selected_occurrences=sum(s['query']['state']=='partial' for s in spots),
            occurrences_in_decoded_selected_runs=sum(len(occurrences(t,gold,word=row['task']=='cwe')) for t in selected_texts),
            spans=spots))
    graph = next(e for e in evidence['graph'] if e['phase']=='query')
    supporting=[]
    for support in supports.get((row['task'],row['question'].removeprefix('Question: ')),[]):
        def locate(source):
            variants=list(dict.fromkeys((source,unicodedata.normalize('NFC',source),
                                        unicodedata.normalize('NFKC',source))))
            for value in variants:
                found=occurrences(text,value)
                if found:
                    return [dict(token_span=list(tokens(a,b)),query=coverage(*tokens(a,b),ranges))
                            for a,b in found]
            return []
        supporting.append(dict(title=support['title'],document=locate(support['text']),
            sentences=[dict(text=sentence,locations=locate(sentence)) for sentence in support['sentences']]))
    value=dict(id=row['id'], task=row['task'], index=row['index'], score=result['official_case_score'],
        question=row['question'], input_sha256=row['served_sha256'], tokenizer_roundtrip=True,
        token_mapping='Original ID byte offsets, verified against the full decoded prompt; no retokenization.',
        result_sha256=sha(Path(result['_path'])), evidence_sha256=sha(Path(evidence['_path'])),
        completion_tokens=result['completion_tokens'], finish_reason=result['finish_reason'],
        thinking_complete=result['thinking_complete'], answer=result['answer'], text_tail=result['text'][-1600:],
        query_selected_tokens=query['selected_tokens'], query_replay_span=query['replay_span'],
        references=refs, supporting_documents=supporting, pressure_count=len(evidence['pressure']),
        graph={k:graph.get(k) for k in ('mode','alpha','backend','added_blocks','displaced_blocks',
              'locality_unknown_fraction','attention_seed_coverage','normalized_l1_change')},
        limits=['Selected KV positions do not prove their original hidden states were computed with all supporting context.',
                'No per-target pre/post-graph ranking trace was recorded; mean-K versus graph displacement cannot be attributed.'])
    if row['task']=='cwe':
        # Whole numbered entries avoid counting incidental words in instructions.
        matches=list(re.finditer(r'\b\d+\. ([A-Za-z]+)(?![A-Za-z])',text))
        main_begin=text.rfind('Below is a numbered list of words.')
        if main_begin<0:
            raise ValueError('Missing main CWE list boundary')
        original=Counter();kept=Counter();example=Counter();kept_example=Counter()
        selected_starts=[a for a,b in merged]
        for match in matches:
            word=match[1].lower();is_main=match.start()>=main_begin
            (original if is_main else example)[word]+=1
            lo,hi=tokens(*match.span(1));index=bisect_right(selected_starts,lo)-1
            if index>=0 and hi<=merged[index][1]:
                (kept if is_main else kept_example)[word]+=1
        value['word_frequencies']=dict(
            scope='Main task list only; the preceding few-shot example is counted separately.',
            original_gold={gold:original[gold.lower()] for gold in unique_refs},
            selected_gold={gold:kept[gold.lower()] for gold in unique_refs},
            max_selected_non_gold=max((count for word,count in kept.items() if word not in unique_refs),default=0),
            selected_top20=kept.most_common(20),
            selected_fewshot_top10=kept_example.most_common(10),
            fewshot_words_returned=[word for word,count in example.items()
                                   if word not in unique_refs and occurrences(result['answer'],word,word=True)])
    return value


def run(root, output, native):
    os.nice(10)
    started = time.monotonic()
    analyzer_sha256=sha(Path(__file__))
    fixture = read(root/'fixture.json')
    profile = read(root/'profile.json')
    tokenizer_path=Path(profile['tokenizer'])/'tokenizer.json'
    if sha(tokenizer_path)!=fixture['tokenizer_sha256']:
        raise ValueError('Tokenizer changed')
    tokenizer=Tokenizer.from_file(str(tokenizer_path))
    mapper=PromptOffsets(tokenizer)
    supports=support_sources(Path(fixture['upstream_path']),fixture['rows'])
    all_results=read(root/'run/answers/completed.json')['results']
    byid={r['id']:r for r in fixture['rows']}
    # Analyze all non-full scores plus successful exact-lookup controls, using
    # one CPU worker so the ongoing native GPU campaign is left undisturbed.
    selected=[r for r in all_results if r['official_case_score']<100 or
              r['task'] in ('niah_multikey_2','niah_multikey_3')]
    analyzed=[]
    for result in sorted(selected,key=lambda r:(r['task'],r['index'])):
        row=byid[result['id']]
        path=Path(fixture['input_directory'])/(row['id']+'.i32')
        if sha(path)!=row['served_sha256']:
            raise ValueError('Frozen input changed')
        ids=np.fromfile(path,dtype='<i4').tolist()
        if len(ids)!=row['served_input_tokens']:
            raise ValueError('Input length changed')
        case=root/'run/answers/cases'/row['id']
        evidence=read(case/'evidence.json'); evidence['_path']=str(case/'evidence.json')
        value=audit_case(row,dict(result,_path=str(case/'result.json')),evidence,ids,tokenizer,mapper,supports)
        other=native/'run/answers/cases'/row['id']/'result.json' if native else None
        value['native_result']=(dict(score=read(other)['official_case_score'],path=str(other),sha256=sha(other))
                                if other and other.exists() else None)
        analyzed.append(value)
        print(json.dumps(dict(id=row['id'],score=value['score'],references=[
            dict(value=r['value'],occurrences=r['occurrences'],selected=r['full_selected_occurrences'],
                 partial=r['partial_selected_occurrences']) for r in value['references']])),flush=True)
    report=dict(kind='ruler-v1-posthoc-failure-evidence', campaign=str(root.resolve()),
        at=time.time(),cpu_seconds_wall=time.monotonic()-started,fixture_sha256=sha(root/'fixture.json'),
        analyzer_sha256=analyzer_sha256,
        failed_cases=sum(r['official_case_score']<100 for r in all_results),analyzed=len(analyzed),
        failure_tasks=dict(Counter(r['task'] for r in all_results if r['official_case_score']<100)),
        rows=analyzed)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--native',type=Path)
    args=parser.parse_args()
    run(args.campaign.resolve(),args.output.resolve(),args.native.resolve() if args.native else None)
