#!/usr/bin/env python3
"""Figures for WRITEUP.md -> docs/figures/*.png (light surface, 2x resolution).

Numbers are read from the judged runs in data/04_eval_cache (the same files scripts/writeup_tables.py uses), except the
weight-space figures, which come from the §6.4 weight table.
  python3 scripts/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
E = ROOT / "data" / "04_eval_cache"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1 and 2 (validated pair)
BASE, STAGE1, FINAL = "base_q4_v2", "joint_all_v3_lr5e-5_e2", "joint_all_v3_mtopd_ck50"
SURF = ["en", "hi", "te", "ja", "hinglish", "hi_en_mix", "te_en_mix", "ja_en_mix"]
LABEL = {"en": "English", "hi": "Hindi", "te": "Telugu", "ja": "Japanese", "hinglish": "Hinglish (Latin script)",
         "hi_en_mix": "Hindi–English mix", "te_en_mix": "Telugu–English mix", "ja_en_mix": "Japanese–English mix"}
HELDOUT = {"strata_heldout", "human"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def stance(run):
    return {d["id"]: d for d in map(json.loads, open(E / run / "stance_decider.jsonl"))}


def per_surface(run, ids, prof):
    S = stance(run)
    return {s: 100 * sum(S[i]["complied"] for i in ids if prof[i] == s) / sum(1 for i in ids if prof[i] == s) for s in SURF}


def style(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def fig_compliance():
    b = stance(BASE)
    ids = [i for i, d in b.items() if d["source"] in HELDOUT]
    prof = {i: b[i]["profile"] for i in ids}
    pb, pf = per_surface(BASE, ids, prof), per_surface(FINAL, ids, prof)
    tb = 100 * sum(b[i]["complied"] for i in ids) / len(ids)
    F = stance(FINAL)
    tf = 100 * sum(F[i]["complied"] for i in ids) / len(ids)
    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=200)
    rows = list(reversed(SURF))
    for y, s in enumerate(rows):
        ax.plot([pf[s], pb[s]], [y, y], color=GRID, lw=2, zorder=1, solid_capstyle="round")
        ax.scatter(pb[s], y, s=70, color=ORANGE, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax.scatter(pf[s], y, s=70, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax.text(pb[s] + 1.6, y, f"{pb[s]:.0f}%", va="center", color=INK2, fontsize=9.5)
        ax.text(pf[s] - 1.6, y, f"{pf[s]:.0f}%", va="center", ha="right", color=INK2, fontsize=9.5)
    ax.set_yticks(range(len(rows)), [LABEL[s] for s in rows])
    ax.set_xlim(-6, 92)
    ax.set_xticks([0, 20, 40, 60, 80], ["0%", "20%", "40%", "60%", "80%"])
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    style(ax)
    ax.set_xlabel("Held-out multi-turn jailbreak conversations the model went along with")
    ax.scatter([], [], s=70, color=ORANGE, label=f"Tiny Aya Global, base ({tb:.1f}% overall)")
    ax.scatter([], [], s=70, color=BLUE, label=f"Final model ({tf:.1f}% overall)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, 1.10), ncol=2, frameon=False, fontsize=9.5)
    fig.suptitle("Jailbreak compliance by language surface (n = 382, lower is safer)", x=0.02, ha="left",
                 y=1.03, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "compliance_by_surface.png", bbox_inches="tight")
    plt.close(fig)


def fig_flores():
    cap = {r: json.load(open(E / r / "capability_summary.json"))["summary"]["flores_chrf_pp"] for r in (BASE, STAGE1, FINAL)}
    dirs = ["en-hi", "en-te", "en-ja", "hi-en", "te-en", "ja-en"]
    d1 = [cap[STAGE1][d] - cap[BASE][d] for d in dirs]
    df = [cap[FINAL][d] - cap[BASE][d] for d in dirs]
    fig, ax = plt.subplots(figsize=(8.2, 3.9), dpi=200)
    x = range(len(dirs))
    w = 0.36
    ax.bar([i - w / 2 - 0.01 for i in x], d1, w, color=ORANGE, label="Stage 1 only (safety DPO)", zorder=2)
    ax.bar([i + w / 2 + 0.01 for i in x], df, w, color=BLUE, label="Final model (+ anchored distillation)", zorder=2)
    for i, (a, b) in enumerate(zip(d1, df)):
        if abs(a) > 0.5:
            ax.text(i - w / 2, a + (0.06 if a > 0 else -0.06), f"{a:+.2f}", ha="center",
                    va="bottom" if a > 0 else "top", fontsize=9, color=INK2)
        ax.text(i + w / 2, b + (0.06 if b >= 0 else -0.06), f"{b:+.2f}", ha="center",
                va="bottom" if b >= 0 else "top", fontsize=9, color=INK2)
    ax.axhline(0, color=MUTED, lw=1, zorder=1)
    ax.set_xticks(list(x), [d.replace("-", " → ") for d in dirs])
    ax.set_ylabel("chrF++ change vs base")
    ax.set_ylim(-1.45, 1.45)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    style(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, 1.11), ncol=2, frameon=False, fontsize=9.5)
    fig.suptitle("Translation (FLORES+ devtest) vs the base model: the final model is not dumber", x=0.02, ha="left",
                 y=1.03, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "flores_vs_base.png", bbox_inches="tight")
    plt.close(fig)


def fig_direction():
    # §6.4 weight table: share of the stage-1 safety update a repair reverses, and its compliance change vs stage 1
    runs = [("Retention DPO", 0.3, 2.1), ("Translation-only OPD, step 50", 1.4, 4.2),
            ("Translation-only OPD, step 100", 2.1, 4.7), ("OPD, step 50", 2.9, 19.9), ("OPD, step 100", 4.6, 32.2),
            ("Translation-only OPD, projected", 0.0, 2.1), ("Multi-teacher OPD + anchor, step 50 (final)", -0.1, -0.3),
            ("Multi-teacher OPD + anchor, step 72", -0.1, 1.0)]
    # label positions in data coordinates (leader lines for the crowded cluster near the origin)
    place = {"Retention DPO": (0.75, 7.0), "Translation-only OPD, step 50": (1.6, 1.4),
             "Translation-only OPD, step 100": (2.3, 4.7), "OPD, step 50": (3.05, 19.9), "OPD, step 100": (4.45, 32.2),
             "Translation-only OPD, projected": (0.75, 11.0), "Multi-teacher OPD + anchor, step 50 (final)": (0.35, -4.7),
             "Multi-teacher OPD + anchor, step 72": (0.35, -2.2)}
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=200)
    for name, u, c in runs:
        final = "final" in name
        ax.scatter(u, c, s=80 if final else 60, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        tx, ty = place[name]
        right = name == "OPD, step 100"
        far = abs(tx - u) > 0.3 or abs(ty - c) > 1.5
        ax.annotate(name, (u, c), xytext=(tx, ty), textcoords="data", fontsize=9,
                    color=INK if final else INK2, ha="right" if right else "left", va="center",
                    fontweight="bold" if final else "normal",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7, shrinkA=0, shrinkB=4, relpos=(0, 0.5)) if far else None)
    ax.axhline(0, color=MUTED, lw=1, zorder=1)
    ax.set_xlim(-0.6, 5.2)
    ax.set_ylim(-6, 36)
    ax.set_xlabel("Share of the stage-1 safety update the repair reverses (weight space, %)")
    ax.set_ylabel("Compliance change vs stage 1 (points)")
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    style(ax)
    ax.spines["bottom"].set_visible(False)
    fig.suptitle("A repair's direction, not its size, decides how much safety it costs", x=0.02, ha="left", y=1.0,
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "direction_vs_safety_cost.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_compliance()
    fig_flores()
    fig_direction()
    print("saved", sorted(p.name for p in OUT.glob("*.png")))
