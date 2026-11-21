"""Probe: count trainable weights left bit-identical after each AdamW step.

Put this directory on PYTHONPATH; Python imports `sitecustomize` at startup and the
patch below wraps torch's AdamW without touching the LeRobot source tree.

Env vars:
  PROBE_LOG    output jsonl path (required)
  PROBE_EVERY  probe interval in optimizer steps (default 100)
"""
import json
import os

import torch

_LOG = os.environ.get("PROBE_LOG", "probe.jsonl")
_EVERY = int(os.environ.get("PROBE_EVERY", "100"))
_orig_step = torch.optim.AdamW.step
_state = {"n": 0, "censused": False}


def _probe_step(self, *args, **kwargs):
    n = _state["n"]
    params = [p for g in self.param_groups for p in g["params"] if p.requires_grad]
    snap = [p.detach().clone() for p in params] if (n % _EVERY == 0 and params) else None

    out = _orig_step(self, *args, **kwargs)

    if snap is not None:
        bf16_numel = same = n2d = same_2d = 0
        for p, before in zip(params, snap):
            if p.dtype is not torch.bfloat16:
                continue
            eq = p.detach() == before
            bf16_numel += eq.numel()
            same += int(eq.sum())
            if p.detach().ndim == 2:
                n2d += eq.numel()
                same_2d += int(eq.sum())
        rec = {
            "step": n,
            "lr": self.param_groups[0].get("lr"),
            "bf16_numel": bf16_numel,
            "unchanged_share_all": (same / bf16_numel) if bf16_numel else None,
            "unchanged_share_2d": (same_2d / n2d) if n2d else None,
        }
        if not _state["censused"] and params:
            census = {}
            for p in params:
                k = str(p.dtype).replace("torch.", "")
                census[k] = census.get(k, 0) + p.numel()
            rec["trainable_param_dtype_numel"] = census
            rec["trainable_total_numel"] = sum(p.numel() for p in params)
            _state["censused"] = True
        del snap
        with open(_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[probe] step={n} lr={rec['lr']} unchanged_2d={rec['unchanged_share_2d']}", flush=True)

    _state["n"] = n + 1
    return out


torch.optim.AdamW.step = _probe_step
