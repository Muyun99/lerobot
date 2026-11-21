"""Summarise a SmolVLA bf16-vs-fp32 run and draw the 2x2 figure.

Expected layout (produced by train.sh and eval.sh):

  <root>/bf16/probe.jsonl   <root>/bf16/train.log   <root>/eval-bf16/eval_info.json
  <root>/fp32/probe.jsonl   <root>/fp32/train.log   <root>/eval-fp32/eval_info.json

Usage: python analyze.py --root repro-out [--figure figure.png]
"""
import argparse
import json
import math
import random
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C = {"bf16": "#d95f02", "fp32": "#1b9e77"}
LABEL = {"bf16": "bf16 expert storage (LeRobot default)", "fp32": "fp32 master copy"}
LOSS_RE = re.compile(r"step:([\d.]+[KM]?) smpl:[\d.]+[KM]? ep:[\d.]+[KM]? epch:[\d.]+ loss:([0-9.]+)")


def _num(tok):
    return int(float(tok[:-1] if tok[-1] in "KM" else tok) * {"K": 1e3, "M": 1e6}.get(tok[-1], 1))


def load_probe(p):
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def load_loss(p):
    if not p.exists():
        return []
    return [(_num(s), float(l)) for s, l in LOSS_RE.findall(p.read_text(errors="ignore"))]


def ema(points, a=0.25):
    xs, ys, e = [], [], None
    for x, y in points:
        e = y if e is None else a * y + (1 - a) * e
        xs.append(x)
        ys.append(e)
    return xs, ys


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * (c - r) / d, 100 * (c + r) / d


def mcnemar(a, b):
    """Exact two-sided binomial test on discordant pairs."""
    b01 = sum(1 for x, y in zip(a, b) if not x and y)
    c10 = sum(1 for x, y in zip(a, b) if x and not y)
    m = b01 + c10
    if not m:
        return b01, c10, 1.0
    k = min(b01, c10)
    return b01, c10, min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2**m)


def flat(info):
    return [s for t in info["per_task"] for s in t["metrics"]["successes"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="repro-out")
    ap.add_argument("--figure", default=None)
    args = ap.parse_args()
    root = Path(args.root)

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("SmolVLA on LIBERO-Object — same checkpoint, data and schedule; "
                 "the action expert is stored in bf16 or fp32", fontsize=11)
    summary = {}

    for arm in ("bf16", "fp32"):
        probe = load_probe(root / arm / "probe.jsonl")
        if probe:
            pts = [(r["step"], 100 * r["unchanged_share_2d"])
                   for r in probe if r.get("unchanged_share_2d") is not None]
            tail = [(s, v) for s, v in pts if s >= 200]
            summary[arm] = {
                "trainable_param_dtype_numel": probe[0].get("trainable_param_dtype_numel"),
                "unchanged_mean_pct": round(sum(v for _, v in tail) / len(tail), 1) if tail else None,
                "unchanged_min_pct": round(min(tail, key=lambda t: t[1])[1], 1) if tail else None,
                "unchanged_min_step": min(tail, key=lambda t: t[1])[0] if tail else None,
                "unchanged_max_pct": round(max(tail, key=lambda t: t[1])[1], 1) if tail else None,
                "unchanged_max_step": max(tail, key=lambda t: t[1])[0] if tail else None,
            }
            if pts:
                ax[0][0].plot([p[0] for p in pts], [p[1] for p in pts], color=C[arm], lw=1.5, label=LABEL[arm])
        loss = load_loss(root / arm / "train.log")
        if loss:
            x, y = ema(loss)
            ax[0][1].plot(x, y, color=C[arm], lw=1.5, label=LABEL[arm])
            summary.setdefault(arm, {})["final_loss_ema"] = round(y[-1], 4)

    ax[0][0].set(xlabel="optimizer step", ylabel="% of bf16 weights unchanged after 1 step",
                 title="(a) Weights unchanged after one AdamW step")
    ax[0][0].set_ylim(0, 102)
    ax[0][0].legend(fontsize=8)
    ax[0][0].grid(alpha=0.25)
    ax[0][1].set(xlabel="optimizer step", ylabel="training loss (EMA)",
                 title="(b) Training loss")
    ax[0][1].legend(fontsize=8)
    ax[0][1].grid(alpha=0.25)

    ev = {}
    for arm, d in (("bf16", "eval-bf16"), ("fp32", "eval-fp32")):
        p = root / d / "eval_info.json"
        if p.exists():
            ev[arm] = json.load(open(p))

    if len(ev) == 2:
        for i, arm in enumerate(("bf16", "fp32")):
            info = ev[arm]
            pct = 100 * info["overall"]["n_success"] / info["overall"]["n_episodes"]
            lo, hi = wilson(info["overall"]["n_success"], info["overall"]["n_episodes"])
            ax[1][0].bar(i, pct, color=C[arm], alpha=0.88, width=0.55)
            ax[1][0].errorbar(i, pct, yerr=[[pct - lo], [hi - pct]], fmt="none", ecolor="black", capsize=5)
            ax[1][0].text(i, pct + 3, f"{pct:.1f}%\n({info['overall']['n_success']}/{info['overall']['n_episodes']})",
                          ha="center", fontsize=9)
            summary[arm]["success_pct"] = round(pct, 2)
            summary[arm]["n_success"] = info["overall"]["n_success"]
            summary[arm]["n_episodes"] = info["overall"]["n_episodes"]

        a, b = flat(ev["bf16"]), flat(ev["fp32"])
        b01, c10, p = mcnemar(a, b)
        random.seed(0)
        diffs = []
        for _ in range(20000):
            idx = [random.randrange(len(a)) for _ in a]
            diffs.append(100 * (sum(b[i] - a[i] for i in idx)) / len(a))
        diffs.sort()
        ci = (diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))])
        summary["paired"] = {"discordant_fp32_only": b01, "discordant_bf16_only": c10,
                             "mcnemar_p": round(p, 4), "bootstrap_ci95_pct": [round(x, 2) for x in ci]}

        tids = [t["task_id"] for t in ev["bf16"]["per_task"]]
        for arm in ("bf16", "fp32"):
            ax[1][1].plot(tids, [100 * t["n_success"] / t["n_episodes"] for t in ev[arm]["per_task"]],
                          "o-", color=C[arm], lw=1.5, ms=6, label=LABEL[arm])
        ax[1][1].legend(fontsize=8)
        ax[1][0].set_xticks([0, 1])
        ax[1][0].set_xticklabels(["bf16", "fp32"], fontsize=9)
        ax[1][0].set(ylabel="LIBERO-Object success (%)", title=f"(c) Paired rollouts, McNemar p = {p:.2f}")
        ax[1][0].set_ylim(0, 105)
        ax[1][0].grid(alpha=0.25, axis="y")
        ax[1][1].set(xlabel="LIBERO-Object task id", ylabel="success (%)", title="(d) Per task")
        ax[1][1].set_ylim(-5, 105)
        ax[1][1].grid(alpha=0.25)

    print(json.dumps(summary, indent=2))
    if args.figure:
        fig.tight_layout()
        fig.savefig(args.figure, dpi=130)
        print("wrote", args.figure)


if __name__ == "__main__":
    main()
