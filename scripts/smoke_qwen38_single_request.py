def main():
    import dataclasses,json,os,pathlib,time
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    model=os.environ["GC_SMOKE_MODEL"]
    out=pathlib.Path(os.environ["GC_SMOKE_OUT"])
    prompt="计算 37 × 19，只用一句中文给出结果。"
    t=time.monotonic()
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    formatted=tokenizer.apply_chat_template([{"role":"user","content":prompt}], tokenize=False,add_generation_prompt=True,enable_thinking=False)
    args=dict(model=model,tokenizer=model,dtype="bfloat16",load_format="safetensors",safetensors_load_strategy="eager",language_model_only=True,kv_cache_dtype="fp8_e4m3",max_model_len=2048,max_num_seqs=1,max_num_batched_tokens=128,gpu_memory_utilization=0.93,enforce_eager=True,enable_prefix_caching=False,async_scheduling=False,disable_log_stats=False,seed=20261023)
    (out/"settings.json").write_text(json.dumps({"model":model,"args":args,"prompt":prompt,"num_requests":1,"mtp":False},ensure_ascii=False,indent=2))
    llm=LLM(**args)
    loaded=time.monotonic()
    sampling=SamplingParams(temperature=0,max_tokens=128)
    outputs=llm.generate([formatted],sampling,use_tqdm=False)
    ended=time.monotonic()
    results=[]
    for r in outputs:
     metrics=vars(r.metrics) if r.metrics is not None and hasattr(r.metrics,"__dict__") else str(r.metrics)
     results.append({"prompt_tokens":len(r.prompt_token_ids),"metrics":metrics,"completions":[{"text":o.text,"token_ids":o.token_ids,"finish_reason":o.finish_reason} for o in r.outputs]})
    payload={"load_s":loaded-t,"generation_s":ended-loaded,"requests":results,"elapsed_s":ended-t}
    (out/"result.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str))
    print("GC_SMOKE_RESULT",json.dumps(payload,ensure_ascii=False,default=str),flush=True)

if __name__ == '__main__':
    main()
