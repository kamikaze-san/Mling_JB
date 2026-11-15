#!/usr/bin/env python3
"""Figures for WRITEUP.md -> docs/figures/*.svg (hand-built SVG: crisp at any size, renders on GitHub).

Numbers are read from the judged runs in data/04_eval_cache (the files scripts/writeup_tables.py uses).
  python3 scripts/make_figures.py
"""
import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
E = ROOT / "data" / "04_eval_cache"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BG, INK, INK2, MUTED, RULE = "#ffffff", "#1a1a19", "#52514e", "#8a8984", "#e8e7e3"
ACCENT, BEFORE = "#2a78d6", "#b9b7b0"   # final model (blue) vs base / stage 1 (warm grey)
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
BASE, STAGE1, FINAL = "base_q4_v2", "joint_all_v3_lr5e-5_e2", "joint_all_v3_mtopd_ck50"
SURF = ["en", "hi", "te", "ja", "hinglish", "hi_en_mix", "te_en_mix", "ja_en_mix"]
LABEL = {"en": "English", "hi": "Hindi", "te": "Telugu", "ja": "Japanese", "hinglish": "Hinglish (Latin script)",
         "hi_en_mix": "Hindi–English mix", "te_en_mix": "Telugu–English mix", "ja_en_mix": "Japanese–English mix"}
HELDOUT = {"strata_heldout", "human"}


def t(x, y, s, size=13, color=INK, weight=400, anchor="start"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" font-weight="{weight}" '
            f'text-anchor="{anchor}">{escape(s)}</text>')


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'font-family="{FONT}" role="img" aria-label="{escape(title)}">'
            f'<rect width="{w}" height="{h}" rx="10" fill="{BG}"/>' + "".join(body) + "</svg>")


def stance(run):
    return {d["id"]: d for d in map(json.loads, open(E / run / "stance_decider.jsonl"))}


def fig_compliance():
    b, f = stance(BASE), stance(FINAL)
    ids = [i for i, d in b.items() if d["source"] in HELDOUT]
    rate = lambda S, sub: 100 * sum(S[i]["complied"] for i in sub) / len(sub)
    tb, tf = rate(b, ids), rate(f, ids)
    W, left, right, top, rowh = 760, 196, 712, 150, 38
    x = lambda v: left + (right - left) * v / 100
    H = top + rowh * len(SURF) + 52
    body = [t(32, 42, "Jailbreak compliance by language, before and after", 19, weight=650),
            t(32, 66, "Share of 382 held-out multi-turn attack conversations the model went along with. Lower is safer.", 13, INK2),
            t(32, 112, f"{tb:.0f}%", 34, MUTED, 650), t(112, 110, "→", 28, MUTED),
            t(146, 112, f"{tf:.0f}%", 34, ACCENT, 700), t(222, 98, "all eight surfaces,", 13, INK2),
            t(222, 115, "base → final model", 13, INK2),
            f'<circle cx="{W - 250}" cy="100" r="6" fill="{BEFORE}"/>', t(W - 238, 104.5, "Tiny Aya Global (base)", 12.5, INK2),
            f'<circle cx="{W - 250}" cy="121" r="6" fill="{ACCENT}"/>', t(W - 238, 125.5, "Final model", 12.5, INK2)]
    for v in (0, 25, 50, 75, 100):
        body.append(f'<line x1="{x(v):.1f}" y1="{top - 10}" x2="{x(v):.1f}" y2="{top + rowh * len(SURF) - 14}" stroke="{RULE}" stroke-width="1"/>')
        body.append(t(x(v), top + rowh * len(SURF) + 4, f"{v}%", 11.5, MUTED, anchor="middle"))
    for r, s in enumerate(SURF):
        sub = [i for i in ids if b[i]["profile"] == s]
        vb, vf = rate(b, sub), rate(f, sub)
        y = top + rowh * r + 8
        body.append(t(left - 16, y + 4.5, LABEL[s], 13, INK, anchor="end"))
        body.append(f'<line x1="{x(vf) + 15:.1f}" y1="{y}" x2="{x(vb) - 8:.1f}" y2="{y}" stroke="{ACCENT}" stroke-opacity="0.35" stroke-width="3" stroke-linecap="round"/>')
        body.append(f'<path d="M{x(vf) + 9:.1f},{y} l9,-5 v10 z" fill="{ACCENT}" fill-opacity="0.6"/>')
        body.append(f'<circle cx="{x(vb):.1f}" cy="{y}" r="7" fill="{BEFORE}"/>')
        body.append(f'<circle cx="{x(vf):.1f}" cy="{y}" r="7.5" fill="{ACCENT}" stroke="{BG}" stroke-width="2"/>')
        body.append(t(x(vb) + 13, y + 4.5, f"{vb:.0f}%", 12, MUTED))
        body.append(t(x(vf) - 13, y + 4.5, f"{vf:.0f}%", 12.5, ACCENT, 650, "end"))
    (OUT / "compliance_by_surface.svg").write_text(svg(W, H, body, "Jailbreak compliance by language, base vs final model"), encoding="utf-8")


def fig_flores():
    cap = {r: json.load(open(E / r / "capability_summary.json"))["summary"]["flores_chrf_pp"] for r in (BASE, STAGE1, FINAL)}
    dirs = ["en-hi", "en-te", "en-ja", "hi-en", "te-en", "ja-en"]
    names = {"en-hi": "English → Hindi", "en-te": "English → Telugu", "en-ja": "English → Japanese",
             "hi-en": "Hindi → English", "te-en": "Telugu → English", "ja-en": "Japanese → English"}
    d1 = {d: cap[STAGE1][d] - cap[BASE][d] for d in dirs}
    df = {d: cap[FINAL][d] - cap[BASE][d] for d in dirs}
    W, left, right, top, rowh, lim = 760, 196, 700, 128, 50, 1.5
    zero = (left + right) / 2
    x = lambda v: zero + (right - left) / 2 * v / lim
    H = top + rowh * len(dirs) + 44
    body = [t(32, 42, "Translation quality compared with the base model", 19, weight=650),
            t(32, 66, "Change in FLORES+ chrF++ against Tiny Aya Global. Right of the centre line is better than the base model.", 13, INK2),
            f'<rect x="32" y="92" width="14" height="9" rx="2" fill="{BEFORE}"/>', t(52, 101, "Safety training only (stage 1)", 12.5, INK2),
            f'<rect x="262" y="92" width="14" height="9" rx="2" fill="{ACCENT}"/>', t(282, 101, "Final model (stage 1 + anchored distillation)", 12.5, INK2)]
    for v in (-1.5, -1, -0.5, 0.5, 1, 1.5):
        body.append(f'<line x1="{x(v):.1f}" y1="{top - 12}" x2="{x(v):.1f}" y2="{top + rowh * len(dirs) - 12}" stroke="{RULE}" stroke-width="1"/>')
        body.append(t(x(v), top + rowh * len(dirs) + 4, f"{v:+.1f}".replace("-", "−"), 11.5, MUTED, anchor="middle"))
    body.append(f'<line x1="{zero}" y1="{top - 16}" x2="{zero}" y2="{top + rowh * len(dirs) - 8}" stroke="{INK2}" stroke-width="1.4"/>')
    body.append(t(zero, top + rowh * len(dirs) + 4, "base", 11.5, INK2, 600, "middle"))

    def bar(v, y, color):
        x0, x1 = sorted((zero, x(v)))
        return f'<rect x="{x0:.1f}" y="{y:.1f}" width="{max(x1 - x0, 1.5):.1f}" height="12" rx="3" fill="{color}"/>'
    for r, d in enumerate(dirs):
        y = top + rowh * r
        body.append(t(left - 16, y + 12, names[d], 13, INK, anchor="end"))
        body.append(bar(d1[d], y - 4, BEFORE))
        body.append(bar(df[d], y + 11, ACCENT))
        for v, yy, col, w in ((d1[d], y + 6, MUTED, 400), (df[d], y + 21, ACCENT, 650)):
            if col == MUTED and abs(v) < 0.5:
                continue
            body.append(t(x(v) + (7 if v >= 0 else -7), yy, f"{v:+.2f}".replace("-", "−"), 11.5, col, w, "start" if v >= 0 else "end"))
    (OUT / "flores_vs_base.svg").write_text(svg(W, H, body, "Translation quality vs base: stage 1 and final model"), encoding="utf-8")


if __name__ == "__main__":
    fig_compliance()
    fig_flores()
    print("saved", sorted(p.name for p in OUT.glob("*.svg")))
