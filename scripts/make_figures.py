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
    W, left, right, top, rowh = 720, 176, 680, 92, 32
    x = lambda v: left + (right - left) * v / 100
    rows = [("Overall", ids)] + [(LABEL[s], [i for i in ids if b[i]["profile"] == s]) for s in SURF]
    gap = 18                                   # space between the overall row and the surfaces
    ybot = top + rowh * (len(rows) - 1) + gap
    H = ybot + 46
    body = [f'<text x="28" y="44" font-size="16" fill="{INK}" font-weight="650">Jailbreak compliance '
            f'<tspan font-size="13" fill="{INK2}" font-weight="400">(lower is safer)</tspan></text>',
            f'<circle cx="{W - 236}" cy="40" r="5" fill="{BEFORE}"/>', t(W - 226, 44, "Tiny Aya Global (base)", 11.5, INK2),
            f'<circle cx="{W - 236}" cy="58" r="5" fill="{ACCENT}"/>', t(W - 226, 62, "Final model", 11.5, INK2),
            # highlight band behind the overall row
            f'<rect x="20" y="{top - 15}" width="{W - 40}" height="30" rx="6" fill="{ACCENT}" fill-opacity="0.07"/>']
    for v in (0, 25, 50, 75, 100):
        body.append(f'<line x1="{x(v):.1f}" y1="{top - 15}" x2="{x(v):.1f}" y2="{ybot + 10}" stroke="{RULE}" stroke-width="1"/>')
        body.append(t(x(v), ybot + 28, f"{v}%", 10.5, MUTED, anchor="middle"))
    for r, (name, sub) in enumerate(rows):
        overall = r == 0
        y = top if overall else top + rowh * r + gap
        vb, vf = rate(b, sub), rate(f, sub)
        rb, rf, lw = (8, 8.5, 3.5) if overall else (6, 6.5, 2.5)
        body.append(t(left - 14, y + 4, name, 12.5 if overall else 12, INK, 700 if overall else 400, "end"))
        body.append(f'<line x1="{x(vf) + 14:.1f}" y1="{y}" x2="{x(vb) - rb - 1:.1f}" y2="{y}" stroke="{ACCENT}" '
                    f'stroke-opacity="{0.55 if overall else 0.3}" stroke-width="{lw}" stroke-linecap="round"/>')
        body.append(f'<path d="M{x(vf) + rf + 1:.1f},{y} l8,-4.5 v9 z" fill="{ACCENT}" fill-opacity="{0.8 if overall else 0.55}"/>')
        body.append(f'<circle cx="{x(vb):.1f}" cy="{y}" r="{rb}" fill="{BEFORE}"/>')
        body.append(f'<circle cx="{x(vf):.1f}" cy="{y}" r="{rf}" fill="{ACCENT}" stroke="{BG}" stroke-width="2"/>')
        fmt = (lambda v: f"{v:.1f}%") if overall else (lambda v: f"{v:.0f}%")
        body.append(t(x(vb) + rb + 6, y + 4, fmt(vb), 11.5 if overall else 11, INK2 if overall else MUTED, 600 if overall else 400))
        body.append(t(x(vf) - rf - 6, y + 4, fmt(vf), 12 if overall else 11.5, ACCENT, 700 if overall else 600, "end"))
    (OUT / "compliance_by_surface.svg").write_text(svg(W, H, body, "Jailbreak compliance, base vs final model, overall and by language"), encoding="utf-8")


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
