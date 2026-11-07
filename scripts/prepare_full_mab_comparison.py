"""Freeze the 2,071-question MAB paper cohort, preserving historical prompt bytes.

Inputs are factored into one immutable prefix per context and small suffixes;
no answers are used to choose inputs or influence retrieval.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time

REPO=Path(__file__).resolve().parents[1]
OLD=REPO/'artifacts/mab-pressure-multiowner-20261105-r10/formal'
INPUTS=Path('/home/chaidi/qw3/benchmark/qw3_memory_comparison_20261004/inputs/memoryagentbench')
RENDERER=Path('/home/chaidi/qw3/scripts/kvmem_eval/prepare_memoryagentbench_archive_row.py')
CAPS={'ruler':50,'eventqa':40,'longmemeval':50,'fact':10,'infbench':1200,'detective':2000,'icl':20,'recsys':300}
CLOSING='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def token_sha(t):return digest(json.dumps(t,separators=(',',':')))
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def gzsave(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(p,'wt',compresslevel=1) as f:json.dump(v,f,ensure_ascii=False,separators=(',',':'))
def lcp(a,b):
    for i,(x,y) in enumerate(zip(a,b)):
        if x!=y:return i
    return min(len(a),len(b))


def prepare(root):
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    assert not root.exists(),'New data directory required'
    root.mkdir(parents=True)
    spec=importlib.util.spec_from_file_location('mab_renderer',RENDERER)
    render=importlib.util.module_from_spec(spec);spec.loader.exec_module(render)
    inventory=read(REPO/'artifacts/mab-official-full-inventory-20261105.json')
    for path,h in inventory['files'].items():assert sha(path)==h
    tables={Path(f).name.split('-00000')[0]:pq.read_table(f).to_pylist() for f in inventory['files']}
    historical={}
    for p in INPUTS.glob('*/row_summary.json'):
        meta=read(p);historical[(meta['split'],meta['dataset_row'])]=p.parent
    old=read(OLD/'fixture.json');oldrows={r['id']:r for r in old['rows']}
    oldbindings={r['id']:r for r in old['bindings']}
    sampler=old['bindings'][0]['sampling']
    assert all(b['sampling']==sampler and b['thinking_token_budget']==0 for b in oldbindings.values())
    tokenizer_path=Path('/home/chaidi/qw3/models/Qwen3.8-27B-BF16/tokenizer.json')
    tok=Tokenizer.from_file(str(tokenizer_path));encode=lambda s:tok.encode(s,add_special_tokens=False).ids
    rows=[];bindings=[];groups=[];overlap=[]
    for n,ctx in enumerate(inventory['contexts']):
        split,index=ctx['split'],ctx['dataset_row'];data=tables[split][index];meta=data['metadata'];kind=render.family(meta['source'])
        assert meta['source']==ctx['source'] and len(data['questions'])==ctx['questions']
        prior=historical.get((split,index))
        if prior:
            group=prior.name
            prefix=(prior/'archive_prefix.qwen-chat-open-user.txt').read_text()+'\n'
        else:
            group=f'full_{split}_r{index:03d}_{ctx["source"].replace("*", "star")}'
            chunks=render.official_chunks(data['context'])
            memorized='\n'.join(render.MEMORIZE[kind].format(context=c,time_stamp='2026-08-02 00:00:00') for c in chunks).strip()
            prefix=f'<|im_start|>system\n{render.SYSTEM_MESSAGE}<|im_end|>\n<|im_start|>user\n'+memorized+'\n'
        preids=encode(prefix);prepath=root/'prefixes'/(group+'.json.gz')
        gzsave(prepath,dict(full=prefix,tokens=preids,prompt_sha256=digest(prefix),token_sha256=token_sha(preids)))
        prefix_hash=sha(prepath);ids=[]
        for qi,(question,answer) in enumerate(zip(data['questions'],data['answers'])):
            ident=group+f'_q{qi:04d}';formatted=render.QUERY[kind].format(question=question,label='{label}')
            suffix=formatted+CLOSING;tail=encode(suffix);full=prefix+suffix;tokens=preids+tail
            # Verify factorization against unfactored tokenization at both ends
            # of every context. The common boundary is newline + task template.
            if qi in (0,len(data['questions'])-1):assert tokens==encode(full),(ident,'token boundary')
            pos=suffix.rfind(str(question));assert pos>=0 and str(question)
            a=encode(suffix[:pos]);b=encode(suffix[:pos+len(str(question))])
            begin=len(preids)+lcp(a,tail);hi=lcp(b,tail);end=len(preids)+hi+(hi<len(b))
            info=dict(split=split,dataset_row=index,source=meta['source'],family=kind,gold_answer=answer,
                question_index=qi,qa_pair_id=render.indexed(meta,'qa_pair_ids',qi),
                question_type=render.indexed(meta,'question_types',qi),question_date=render.indexed(meta,'question_dates',qi),
                question_id=render.indexed(meta,'question_ids',qi),previous_event=render.indexed(meta,'previous_events',qi),
                keypoints=meta.get('keypoints'),raw_question=question,formatted_question=formatted)
            path=root/'questions'/(ident+'.json')
            sample=dict(schema='factored-mab-v1',prefix=str(prepath),prefix_sha256=prefix_hash,
                suffix=suffix,suffix_tokens=tail,query=str(question),query_begin=begin,query_end=end,
                max_tokens=CAPS[kind],thinking=False,info=info)
            save(path,sample)
            row=dict(suite='memoryagentbench',id=ident,input_tokens=len(tokens),source=str(path),
                source_cache_sha256=sha(path),prompt_sha256=digest(full),token_sha256=token_sha(tokens),
                query_sha256=digest(str(question)),query_begin=begin,query_end=end)
            if ident in oldrows:
                original=oldrows[ident]
                for key in ('input_tokens','prompt_sha256','token_sha256','query_sha256','query_begin','query_end'):
                    assert row[key]==original[key],(ident,key,row[key],original[key])
                assert CAPS[kind]==oldbindings[ident]['max_tokens']
                overlap.append(ident)
            binding=dict(id=ident,sampling=sampler,max_tokens=CAPS[kind],thinking_token_budget=0,
                sha256=digest(json.dumps(dict(sampling=sampler,max_tokens=CAPS[kind],thinking_token_budget=0),sort_keys=True)))
            rows.append(row);bindings.append(binding);ids.append(ident)
        groups.append(dict(id=group,split=split,source=meta['source'],dataset_row=index,cases=ids,
            prefix_tokens=len(preids),prefix=str(prepath),sha256=prefix_hash))
        save(root/'progress.json',dict(contexts=n+1,cases=len(rows),at=time.time()))
        print(json.dumps(dict(contexts=n+1,cases=len(rows),source=ctx['source'],prefix_tokens=len(preids))),flush=True)
    assert len(rows)==2071 and len(groups)==130 and len(overlap)==1116
    save(root/'cohort.json',dict(rows=rows,bindings=bindings,groups=groups,cases=2071,contexts=130,
        max_input_tokens=max(r['input_tokens'] for r in rows),sampling=sampler,thinking=False,
        historical_overlap_count=len(overlap),historical_inputs_byte_identical=True,
        historical_fixture=str(OLD/'fixture.json'),historical_fixture_sha256=sha(OLD/'fixture.json'),
        selection='Paper main 2071; full EventQA and 262k FactConsolidation only. No answer-based sampling.',
        renderer=str(RENDERER),renderer_sha256=sha(RENDERER),tokenizer_sha256=sha(tokenizer_path),data_files=inventory['files']))
    shutil.copy2(RENDERER,root/'renderer.py')
    save(root/'proof.json',{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='progress.json'})
    save(root/'status.json',dict(status='completed',cases=2071,contexts=130,at=time.time()))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    prepare(parser.parse_args().root.resolve())
