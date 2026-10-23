"""One real fixed-Q/K component: original indices, scoring and picker equality.

This intentionally retains the original qualification procedure. It does not
start a model; archive read_region coverage is a separate component test.
Outputs use exclusive creation so a previous measurement is never replaced.
"""
from pathlib import Path
import argparse
import json,time,hashlib
import numpy as np
import torch
from vllm_kvmem.query_key_fps import select_key_representatives_optimized,key_fps_scores
from analyze_query_key_fps import native_selections
r=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--case',default='niah_multikey_2-0011')
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.output.exists():raise FileExistsError('Refusing to replace a previous component measurement')
s=r/'artifacts/ruler-retrieval-diagnostic-20261023-r4';c=args.case;p=s/'compressed/cases'/c/'diagnostic'
oldroot=r/'artifacts/query-key-fps-fixed-operands-20261023-r1'/c
meta=json.loads((p/'archive.json').read_text());keys=np.load(p/'raw_keys.npy',mmap_mode='r');q=torch.load(p/'queries.pt',weights_only=True)
old=np.load(oldroot/'representative-token-offsets.npy');oldsummary=json.loads((oldroot/'summary.json').read_text())
L,N,H,D=keys.shape;B=len(meta['counts']); reps=torch.empty((L,B,4,H,D),dtype=torch.float16);ids=torch.empty((L,B,4,H),dtype=torch.int8)
torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.init();torch.cuda.reset_peak_memory_stats()
read=select=write=0.;fallback=0;groups=0;diff=0;keydiff=0;start=time.perf_counter()
for b in range(0,B,32):
 e=min(B,b+32);ns=meta['counts'][b:e];n=sum(ns);beg=time.perf_counter()
 source=torch.from_numpy(np.array(keys[:,b*32:b*32+n],copy=True)).view(torch.float8_e4m3fn)
 packed=torch.zeros((e-b,L,32,H,D),dtype=torch.float8_e4m3fn,pin_memory=True)
 if all(x==32 for x in ns):packed.copy_(source.reshape(L,e-b,32,H,D).permute(1,0,2,3,4))
 else:
  off=0
  for i,x in enumerate(ns):packed[i,:,:x].copy_(source[:,off:off+x]);off+=x
 raw=packed.permute(1,0,2,3,4);now=time.perf_counter();read+=now-beg
 x,i,stat=select_key_representatives_optimized(raw,ns,device='cuda');done=time.perf_counter();select+=done-now
 expected=torch.from_numpy(old[:,b:e]).long();diff+=int((i!=expected).sum())
 original=raw.view(torch.uint8).gather(2,expected.clamp_min(0)[...,None].expand(L,e-b,4,H,D))
 keydiff+=int((x.view(torch.uint8)!=original).sum())
 reps[:,b:e].copy_(x.half());ids[:,b:e].copy_(i.to(torch.int8));write+=time.perf_counter()-done
 fallback+=stat['fps_cpu_fallback_groups'];groups+=stat['fps_gpu_groups']
 if b%1024==0: print(json.dumps(dict(block=b,seconds=time.perf_counter()-start,diff=diff,fallback=fallback)),flush=True)
end=time.perf_counter();buildpeak=torch.cuda.max_memory_allocated();q=q.cuda();mask=torch.tensor(meta['score_mask'],dtype=torch.bool,device='cuda')
torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter();a=key_fps_scores(q,reps,ids,mask,query_half=True);torch.cuda.synchronize();oldtime=time.perf_counter()-begin;oldpeak=torch.cuda.max_memory_allocated()
torch.cuda.reset_peak_memory_stats();begin=time.perf_counter();z=key_fps_scores(q,reps,ids,mask,query_half=True,optimized=True);torch.cuda.synchronize();newtime=time.perf_counter()-begin;newpeak=torch.cuda.max_memory_allocated()
profile=json.loads((s/'compressed/profile.json').read_text());v={'captured_mean_k':np.asarray(meta['mean_scores']),'old_scoring':a.cpu().numpy(),'optimized_scoring':z.cpu().numpy()};selected=native_selections(profile,meta,v)
report=dict(case=c,input_source=str(p),old_reference=str(oldroot),indices_equal=(diff==0),index_differences=diff,representative_bytes_equal=(keydiff==0),key_byte_differences=keydiff,representative_offsets=old.size,groups=groups,cpu_fallback_groups=fallback,cpu_fallback_fraction=fallback/groups,score_bitwise_equal=torch.equal(a,z),score_max_abs_difference=(a-z).abs().max().item(),native_picker_equal=selected['old_scoring']==selected['optimized_scoring'],score_mass=float(z.sum()),cpu_threads=torch.get_num_threads(),raw_read_pack_seconds=read,gpu_select_with_cpu_guard_seconds=select,representative_publish_and_validation_seconds=write,build_seconds=end-start,old_score_seconds=oldtime,optimized_score_seconds=newtime,build_peak_cuda_allocated=buildpeak,old_score_peak_cuda_allocated=oldpeak,optimized_score_peak_cuda_allocated=newpeak,reference_original_capture_hashes=oldsummary['source_sha256'],scope='Fixed real FP8 K/Q; mmap read+packing+GPU construction+guard+CPU cache+scoring+native picker; archive region API covered by component tests, not live model serving. Publication timing also includes explicit original-index and Key-byte validation. Original CPU indices already saved; no CPU double-computation except guarded groups.')
report.update(torch_version=torch.__version__,torch_cuda_version=torch.version.cuda,
 gpu_model=torch.cuda.get_device_name(),benchmark_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 measured_source_sha256={name:hashlib.sha256((r/name).read_bytes()).hexdigest() for name in
 ('src/vllm_kvmem/query_key_fps.py','src/vllm_kvmem/authority.py','src/vllm_kvmem/storage.py',
  'src/vllm_kvmem/memory_storage.py','scripts/analyze_query_key_fps.py')})
out=args.output
with out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(report),flush=True)
