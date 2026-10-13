"""Explicit profile serving and version-checked native vLLM compatibility entry."""
import argparse
import os
import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'elastic-serve':
        from .elastic_serve import main as elastic_main
        return elastic_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == 'serve':
        from .serve import main as serve_main
        return serve_main(sys.argv[2:])
    parser = argparse.ArgumentParser(epilog='Experimental profile serving: vllm-kvmem serve --help')
    parser.add_argument("--draft-load-compat", action="store_true")
    parser.add_argument("command", choices=["vllm"])
    args, remaining = parser.parse_known_args()
    if args.draft_load_compat:
        os.environ["KVMEM_DRAFT_LOAD_COMPAT"] = "1"
        from .compat import enable_draft_load_config_fix
        enable_draft_load_config_fix()
    if "VLLM_PLUGINS" in os.environ:
        enabled = os.environ["VLLM_PLUGINS"].split(",")
        if "kvmem" not in enabled:
            raise SystemExit("VLLM_PLUGINS must include kvmem for worker compatibility hooks")
    from vllm.entrypoints.cli.main import main as vllm_main
    sys.argv = ["vllm", *remaining]
    vllm_main()


if __name__ == "__main__":
    main()
