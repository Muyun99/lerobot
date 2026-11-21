# Reproducing the SmolVLA bf16 vs fp32 evidence

Everything needed to regenerate the numbers and the figure in the PR description and in
[issue #5099](https://github.com/huggingface/lerobot/issues/5099#issuecomment-7061066012).

## Setup

```bash
git clone https://github.com/huggingface/lerobot && cd lerobot
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[smolvla,dataset,libero]"
```

`hf-libero` pulls `robomimic -> egl-probe`, which needs EGL headers, and cmake 4 rejects
older `cmake_minimum_required`. On a headless box:

```bash
apt-get download libegl-dev && dpkg-deb -x libegl-dev_*.deb /tmp/egl && cp -r /tmp/egl/usr/include/EGL /usr/include/
ln -sf /usr/lib/x86_64-linux-gnu/libEGL.so.1  /usr/lib/x86_64-linux-gnu/libEGL.so
export CMAKE_POLICY_VERSION_MINIMUM=3.5
```

If `torchcodec` fails to load with your torch build, `pip uninstall torchcodec` — LeRobot
falls back to PyAV.

## Run

```bash
cd evidence

ARM=bf16 STEPS=10000 ./train.sh                 # LeRobot default: expert in bf16
ARM=fp32 STEPS=10000 ./train.sh                 # same run with the one-line fp32 cast

CKPT=repro-out/bf16/checkpoints/last/pretrained_model OUT=eval-bf16 ./eval.sh
CKPT=repro-out/fp32/checkpoints/last/pretrained_model OUT=eval-fp32 ./eval.sh

python analyze.py --root repro-out --figure figure.png
```

`train.sh` applies the paper's one-line change (`apply_fp32_cast.py on`) for the fp32 arm
and restores the file on exit, so one checkout serves both arms. The probe
(`probe_sitecustomize.py`) is injected through `PYTHONPATH` and only observes — it does not
change training.

## What is measured

| file | meaning |
|---|---|
| `repro-out/*/probe.jsonl` | every 100 steps: share of bf16 trainable weights bit-identical before/after one AdamW step, plus the step-0 dtype census |
| `repro-out/*/train.log` | training loss and the optimizer schedule |
| `eval-*/eval_info.json` | 500 LIBERO-Object tries (10 tasks x 50 fixed initial states), 50-action chunks open loop |

`analyze.py` prints the summary and redraws the 2x2 figure: unchanged weights, loss, paired
success, per-task success. Paired statistics use exact McNemar on discordant pairs plus a
percentile bootstrap over the 500 matched rollouts (seed 0).

Expected wall clock on one 140 GB-class GPU: ~40 min per training arm, ~40 min per 500-try
evaluation when run alone.
