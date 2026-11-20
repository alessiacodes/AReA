"""Inference-only AReA (FLUX.1) run with latency tracking.

Loads the model and calls the pipeline exactly as generate.py does (FLUX.1-Krea-dev,
bf16, VAE tiling, CPU offload below 60 GB of GPU memory, AReA with its default
options and seed = generation seed, generator on the execution device,
28 steps, guidance 4.5). The pipeline is loaded once and reused across runs, so
run 1 includes CUDA/kernel warm-up and run 2 does not.

Recorded per run (all CUDA-synchronized), by wrapping functions at runtime
(no repo code is modified):
  - text encoding (CLIP + T5)
  - per denoising step: total, transformer forward, AReA phase and size,
    AReA overhead (per-step setup, Phase II window layout, bridge token selection)
  - final VAE decode, end-to-end time, image save time
  - peak GPU memory (PyTorch allocator)
plus import time and model load time.

AReA has no low-res stage. For the luma comparison, the same prompt and seed are
also generated at 1024x1024 with the same pipeline; at the native resolution AReA
keeps the original attention, so this is plain FLUX.1-Krea-dev. It is saved as
lowres_1024x1024.jpg next to the 4K image.

Run on a GPU node from the repo root:
    python infer_timed.py --prompt "..." --out_dir zrun/infer4k
"""
import argparse
import json
import os
import socket
import statistics
import sys
import time

T_SCRIPT_START = time.perf_counter()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from diffusers import FluxPipeline  # noqa: E402
from diffusers.models import AutoencoderKL, FluxTransformer2DModel  # noqa: E402
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler  # noqa: E402

import generate as area_generate  # noqa: E402
from area import AReA, bridge  # noqa: E402
from area.backbones.flux import FluxBackbone  # noqa: E402

T_IMPORTS_DONE = time.perf_counter()
GB = 1024 ** 3

# Same values as generate.py for --model flux1
MODEL_ID = area_generate.MODELS["flux1"]["checkpoint"]
GUIDANCE = area_generate.MODELS["flux1"]["guidance"]
STEPS = area_generate.MODELS["flux1"]["steps"]
SIZE = 4096
REF_SIZE = 1024


def now() -> float:
    return time.perf_counter()


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompt", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--runs", type=int, default=2, help="1 = cold run only; 2 = cold + warm run")
    p.add_argument("--time_budget_s", type=float, default=None,
                   help="skip further runs if they would likely exceed this many seconds since script start")
    p.add_argument("--no_hooks", action="store_true",
                   help="record only end-to-end times and memory (no per-call syncs), to check the hooks' own cost")
    return p.parse_args()


class Recorder:
    def __init__(self):
        self.reset()

    def reset(self):
        self.events = []
        self.phases = []


REC = Recorder()


def _record(label, fn, *args, **kwargs):
    sync()
    t0 = now()
    out = fn(*args, **kwargs)
    sync()
    REC.events.append({"label": label, "start": t0, "end": now()})
    return out


def install_hooks():
    def wrap_method(cls, attr, label):
        orig = getattr(cls, attr)

        def wrapper(self, *args, **kwargs):
            return _record(label, orig, self, *args, **kwargs)

        setattr(cls, attr, wrapper)

    def wrap_func(module, attr, label):
        orig = getattr(module, attr)

        def wrapper(*args, **kwargs):
            return _record(label, orig, *args, **kwargs)

        setattr(module, attr, wrapper)

    # Patched on the classes before the model is built and AReA attaches, so accelerate's
    # offload hooks and AReA's forward hooks pick up the wrapped methods.
    wrap_method(FluxPipeline, "encode_prompt", "text_encode")
    wrap_method(FluxTransformer2DModel, "forward", "transformer_forward")
    wrap_method(AutoencoderKL, "decode", "final_decode")
    wrap_method(FlowMatchEulerDiscreteScheduler, "step", "scheduler_step")
    wrap_method(FluxBackbone, "_after_embedding", "area_layout")
    # area/attention.py calls these as bridge.<name>.
    wrap_func(bridge, "token_importance", "area_bridge")
    wrap_func(bridge, "select_registers", "area_bridge")

    orig_begin = FluxBackbone._begin_step

    def begin_step(self, *args, **kwargs):
        out = _record("area_step_setup", orig_begin, self, *args, **kwargs)
        st = self.state
        REC.phases.append({"phase": st.phase or "plain", "size": st.size if st.phase else None})
        return out

    FluxBackbone._begin_step = begin_step


def summarize_run(t_gen0, t_gen1, peaks) -> dict:
    ev = REC.events

    def total(label):
        return sum(e["end"] - e["start"] for e in ev if e["label"] == label)

    step_ends = [e["end"] for e in ev if e["label"] == "scheduler_step"]
    enc_end = max((e["end"] for e in ev if e["label"] == "text_encode"), default=t_gen0)
    steps, prev = [], enc_end
    for i, end in enumerate(step_ends):
        in_step = [e for e in ev if prev <= e["start"] < end]

        def s(*labels, in_step=in_step):
            return sum(e["end"] - e["start"] for e in in_step if e["label"] in labels)

        ph = REC.phases[i] if i < len(REC.phases) else {"phase": None, "size": None}
        steps.append({
            "total": end - prev,
            "phase": ph["phase"],
            "size": ph["size"],
            "transformer_forward": s("transformer_forward"),
            "area_overhead": s("area_step_setup", "area_layout", "area_bridge"),
            "area_bridge": s("area_bridge"),
            "scheduler_step": s("scheduler_step"),
        })
        prev = end
    totals = [st["total"] for st in steps]

    def phase_sum(name):
        return sum(st["total"] for st in steps if st["phase"] == name)

    return {
        "text_encode": total("text_encode"),
        "denoise": (step_ends[-1] - enc_end) if step_ends else None,
        "steps": steps,
        "step_mean_excl_first": statistics.mean(totals[1:]) if len(totals) > 1 else None,
        "step_min": min(totals) if totals else None,
        "step_max": max(totals) if totals else None,
        "phase1_total": phase_sum("coarse"),
        "phase2_total": phase_sum("fine"),
        "area_overhead_total": sum(st["area_overhead"] for st in steps),
        "final_decode": total("final_decode"),
        "generate_total": t_gen1 - t_gen0,
        **peaks,
    }


def luma_stats(im: Image.Image) -> dict:
    a = np.asarray(im.convert("RGB")).astype(np.float32)
    y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    return {"mean_Y": float(y.mean()), "std_Y": float(y.std()),
            "rgb_mean": [float(a[..., c].mean()) for c in range(3)]}


def fmt(x, unit="s"):
    return f"{x:9.2f} {unit}"


def render_summary(meta: dict, runs: list, ref: dict) -> str:
    L = ["=== AReA (FLUX.1) inference timing ===",
         f"host {meta['host']}  job {meta['job']}  GPU {meta['gpu']} ({meta['gpu_mem_gb']:.1f} GiB)",
         f"torch {meta['torch']}  diffusers {meta['diffusers']}  transformers {meta['transformers']}",
         f"model {meta['model']}  (bf16, {meta['placement']})",
         f"output {SIZE}x{SIZE}  steps {STEPS}  guidance {GUIDANCE}  seed {meta['seed']}  AReA {meta['area_config']}",
         f"prompt: {meta['prompt']}",
         "",
         f"{'python imports':<36}{fmt(meta['imports_s'])}",
         f"{'model load (from_pretrained + move)':<36}{fmt(meta['load_s'])}",
         ""]
    head = f"{'':<36}" + "".join(f"{('run %d (%s)' % (i + 1, 'cold' if i == 0 else 'warm')):>22}" for i in range(len(runs)))
    L.append(head)

    def row(label, fn, unit="s"):
        vals = []
        for r in runs:
            try:
                v = fn(r)
            except (IndexError, KeyError, TypeError, statistics.StatisticsError):
                v = None
            vals.append(f"{'-':>22}" if v is None else f"{v:>20.2f} {unit}")
        L.append(f"{label:<36}" + "".join(vals))

    row("text encoding (CLIP + T5)", lambda r: r["text_encode"])
    row(f"denoise ({STEPS} steps)", lambda r: r["denoise"])
    row("  Phase I steps (coarse), total", lambda r: r["phase1_total"])
    row("  Phase II steps (windowed), total", lambda r: r["phase2_total"])
    row("  step 1", lambda r: r["steps"][0]["total"])
    row("  mean step (steps 2-28)", lambda r: r["step_mean_excl_first"])
    row("  mean Phase I step (excl. step 1)",
        lambda r: statistics.mean(st["total"] for st in r["steps"][1:] if st["phase"] == "coarse"))
    row("  mean Phase II step",
        lambda r: statistics.mean(st["total"] for st in r["steps"] if st["phase"] == "fine"))
    row("  min / max step", lambda r: r["step_min"])
    row("", lambda r: r["step_max"])
    row("  AReA overhead, all steps", lambda r: r["area_overhead_total"])
    row("    of which bridge selection", lambda r: sum(st["area_bridge"] for st in r["steps"]))
    row("final VAE decode (4096, tiled)", lambda r: r["final_decode"])
    row("generate total", lambda r: r["generate_total"])
    row("image save (PNG+JPG)", lambda r: r["save_s"])
    row("peak GPU alloc, whole run", lambda r: r["peak_alloc_gb"], unit="GiB")
    row("peak GPU reserved, whole run", lambda r: r["peak_reserved_gb"], unit="GiB")
    L.append("")
    L.append("per-step totals (s) [phase size]:")
    for i, r in enumerate(runs):
        L.append(f"  run {i + 1}: " + " ".join(
            f"{st['total']:.1f}[{(st['phase'] or '?')[0]}{st['size'] or ''}]" for st in r["steps"]))
    L.append("")
    if ref and ref["denoise"] is not None:
        L.append(f"1024 reference (same pipeline/seed, AReA inactive): generate {ref['generate_total']:.2f} s, "
                 f"denoise {ref['denoise']:.2f} s, mean step {ref['step_mean_excl_first']:.2f} s")
    elif ref:
        L.append(f"1024 reference (same pipeline/seed, AReA inactive): generate {ref['generate_total']:.2f} s (no hooks)")
    for i, r in enumerate(runs):
        if ref:
            lr, hr = ref["luma"], r["luma_4k"]
            L.append(f"run {i + 1} luma (BT.601, 0-255): 1024 reference mean {lr['mean_Y']:.1f} std {lr['std_Y']:.1f}  |  "
                     f"4K (downscaled to 1024) mean {hr['mean_Y']:.1f} std {hr['std_Y']:.1f}")
        L.append(f"run {i + 1} image: {r['image']}")
    if len(runs) > 1:
        L.append(f"run 1 vs run 2 images identical: {meta.get('runs_identical')}")
    L.append(f"script wall time so far: {now() - T_SCRIPT_START:.1f} s")
    return "\n".join(L)


def generate(pipe, prompt, size, seed):
    """Same call as generate.py (non-bootstrap path)."""
    generator = torch.Generator(pipe._execution_device).manual_seed(seed)
    REC.reset()
    torch.cuda.reset_peak_memory_stats()
    sync()
    t0 = now()
    image = pipe(prompt, generator=generator, height=size, width=size, num_inference_steps=STEPS,
                 guidance_scale=GUIDANCE).images[0]
    sync()
    t1 = now()
    peaks = {"peak_alloc_gb": torch.cuda.max_memory_allocated() / GB,
             "peak_reserved_gb": torch.cuda.max_memory_reserved() / GB}
    return image, summarize_run(t0, t1, peaks)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        sys.exit("requires a CUDA GPU")
    os.makedirs(args.out_dir, exist_ok=True)
    if not args.no_hooks:
        install_hooks()

    import diffusers
    import transformers
    props = torch.cuda.get_device_properties(0)
    meta = {
        "host": socket.gethostname(), "job": os.environ.get("SLURM_JOB_ID", "-"),
        "gpu": props.name, "gpu_mem_gb": props.total_memory / GB,
        "torch": torch.__version__, "diffusers": diffusers.__version__, "transformers": transformers.__version__,
        "model": MODEL_ID, "seed": args.seed, "prompt": args.prompt,
        "imports_s": T_IMPORTS_DONE - T_SCRIPT_START,
    }

    # Same loading as generate.py.
    t0 = now()
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.vae.enable_tiling()
    gpu_gb = torch.cuda.get_device_properties(0).total_memory / GB
    if gpu_gb < area_generate.OFFLOAD_BELOW_GB:
        pipe.enable_model_cpu_offload()
        meta["placement"] = "model CPU offload (GPU < 60 GB, as in generate.py)"
    else:
        pipe.to("cuda")
        meta["placement"] = "all on GPU (GPU >= 60 GB, as in generate.py)"
    # generate.py defaults: pooling subsample, phase1_steps per model, halo 8, bridge_ratio 0.5, seed = generation seed
    area = AReA(pipe, pooling="subsample", phase1_steps=None, halo=8, bridge_ratio=0.5, seed=args.seed)
    meta["area_config"] = repr(area.config)
    sync()
    meta["load_s"] = now() - t0
    print(f"model load {meta['load_s']:.1f} s ({meta['placement']})", flush=True)

    runs, ref = [], None
    for i in range(args.runs):
        if i > 0 and args.time_budget_s:
            last = runs[-1]["generate_total"] + runs[-1]["save_s"]
            if now() - T_SCRIPT_START + last * 1.2 > args.time_budget_s:
                print(f"skipping run {i + 1}: would exceed time budget", flush=True)
                break
        run_dir = os.path.join(args.out_dir, f"run{i + 1}")
        os.makedirs(run_dir, exist_ok=True)
        image, r = generate(pipe, args.prompt, SIZE, args.seed)

        t_s0 = now()
        png = os.path.join(run_dir, f"area_{SIZE}x{SIZE}.png")
        image.save(png)
        preview = image.resize((REF_SIZE, REF_SIZE), Image.LANCZOS)
        preview.save(os.path.join(run_dir, f"area_preview_{REF_SIZE}.jpg"), quality=95)
        r["save_s"] = now() - t_s0
        r["image"] = png
        r["luma_4k"] = luma_stats(preview)
        runs.append(r)
        print(f"run {i + 1}: generate {r['generate_total']:.1f} s", flush=True)

        if len(runs) == 2:
            a = np.asarray(Image.open(runs[0]["image"]))
            b = np.asarray(Image.open(runs[1]["image"]))
            meta["runs_identical"] = bool(np.array_equal(a, b))

        if ref is None:
            ref_im, ref = generate(pipe, args.prompt, REF_SIZE, args.seed)
            ref_im.save(os.path.join(args.out_dir, f"lowres_{REF_SIZE}x{REF_SIZE}.jpg"), quality=95)
            ref["luma"] = luma_stats(ref_im)
            print(f"1024 reference: generate {ref['generate_total']:.1f} s", flush=True)

        with open(os.path.join(args.out_dir, "timings.json"), "w") as f:
            json.dump({"meta": meta, "runs": runs, "ref_1024": ref}, f, indent=2)
        summary = render_summary(meta, runs, ref)
        with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
            f.write(summary + "\n")
        print("\n" + summary, flush=True)


if __name__ == "__main__":
    main()
