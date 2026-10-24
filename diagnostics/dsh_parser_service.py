"""Owned loopback CPU service for the exact installed vLLM Qwen tool parser."""
import argparse
from pathlib import Path
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class ParseInput(BaseModel):
    text: str = Field(max_length=2097152)
    tools: list[dict] = Field(max_length=64)


def create_app(tokenizer_path):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('The parser service must explicitly hide CUDA devices')
    from transformers import AutoTokenizer
    # Fail during setup if official parsing dependencies are absent.
    from vllm.parser.qwen3 import Qwen3Parser  # noqa: F401
    from scripts.dsh_model_protocol import TokenizerProtocol
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
    protocol=TokenizerProtocol(tokenizer,tokenizer_path,enable_thinking=False)
    app=FastAPI()

    @app.get('/health')
    def health():
        return dict(cpu_only=True,provenance=protocol.provenance)

    @app.post('/parse')
    def parse(value: ParseInput):
        try:
            return protocol.parse(value.text,value.tools)
        except (ValueError,KeyError,TypeError) as error:
            raise HTTPException(status_code=422,detail=str(error)) from error

    return app


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer',type=Path,required=True)
    parser.add_argument('--port',type=int,default=18389)
    args=parser.parse_args()
    import uvicorn
    uvicorn.run(create_app(args.tokenizer),host='127.0.0.1',port=args.port)


if __name__=='__main__':
    main()
