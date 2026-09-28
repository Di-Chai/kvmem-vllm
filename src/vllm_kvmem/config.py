from dataclasses import dataclass


@dataclass(frozen=True)
class KVMemConfig:
    block_tokens: int = 128
    select_budget: int = 131072
    gen_budget: int = 65536
    repository_blocks: int = 4096
    sink_blocks: int | None = None
    recent_blocks: int | None = None
    kv_dtype: str = "fp8_e4m3"
    speculation: bool = True

    def __post_init__(self) -> None:
        for key in ("block_tokens", "select_budget", "gen_budget", "repository_blocks"):
            value = getattr(self, key)
            if type(value) is not int or not 0 < value <= 2**32 - 1:
                raise ValueError(f"{key} must be a positive uint32")
        if self.select_budget % self.block_tokens or self.gen_budget % self.block_tokens:
            raise ValueError("K/G must be divisible by block_tokens")
        if self.repository_blocks * self.block_tokens >= 2**31:
            raise ValueError("Physical addressing exceeds signed 32-bit kernel limit")
        if self.repository_blocks * self.block_tokens < self.select_budget + self.gen_budget:
            raise ValueError("Repository must fit K + G")
        for key in ("sink_blocks", "recent_blocks"):
            value = getattr(self, key)
            if value is not None and (type(value) is not int or not 0 <= value <= 2**32 - 1):
                raise ValueError(f"{key} must be a nonnegative integer")
        from . import _native
        try:
            sink, recent = _native.resolve_keep_bands(
                self.block_tokens, self.select_budget,
                -1 if self.sink_blocks is None else self.sink_blocks,
                -1 if self.recent_blocks is None else self.recent_blocks)
        except RuntimeError as error:
            raise ValueError(str(error)) from error
        object.__setattr__(self, "sink_blocks", sink)
        object.__setattr__(self, "recent_blocks", recent)
        if self.kv_dtype not in ("fp8_e4m3", "float16", "bfloat16"):
            raise ValueError(f"KV codec not qualified: {self.kv_dtype}")

    def native(self):
        from . import _native
        cfg = _native.StoreConfig()
        for key in ("block_tokens", "select_budget", "gen_budget", "repository_blocks",
                    "sink_blocks", "recent_blocks"):
            setattr(cfg, key, getattr(self, key))
        cfg.prefill_budget = self.select_budget
        cfg.immutable_source_k = True
        cfg.immutable_refresh_abs_delta_tokens = 1 if self.kv_dtype.startswith("fp8") else 262144
        cfg.mtp_enabled = self.speculation
        return cfg
