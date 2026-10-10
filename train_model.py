"""Learn the demand model that recommender.py and index.html use.

  python train_model.py                      make synthetic history, train, save
  python train_model.py --data ridership.csv train on real data instead

Real data needs these CSV columns, one row per day and time band:
  date,band,off,light,heavy,hot,ev_in,ev_out,demand
  band 0-4 (see BANDS); off/light/heavy/hot are 0 or 1; ev_in/ev_out are event
  crowd points in that band; demand is the demand index (100 = typical weekday peak).

Writes model.json, ridership_synthetic.csv (synthetic mode) and embeds the
model into index.html so the app works offline from a single file.
Python 3.8+, no dependencies.
"""
import csv
import json
import math
import os
import random
import re
import sys
from datetime import date, timedelta

from recommender import BANDS, CROWD, FEATURES, OFFDAY, TYPICAL, event_loads

HERE = os.path.dirname(os.path.abspath(__file__))
rng = random.Random(2026)

# The synthetic world. These effects are hidden from the model, which must learn them.
PULL_IN = [.8, .9, .9, 1.0, .9]        # share of inbound crowd that rides, per band
PULL_OUT = [1.0, 1.0, 1.1, 1.1, 1.3]   # outbound crowds ride more, esp. late at night
WX_UP = {"clear": [0] * 5, "light": [.08] * 5, "heavy": [.12, .18, .12, .18, .12], "hot": [0, 0, .06, 0, 0]}


def _hhmm(h):
    return "%02d:%02d" % (int(h) % 24, round((h - int(h)) * 60))


def make_day(d):
    holiday = d.weekday() < 5 and rng.random() < .04
    off = holiday or d.weekday() >= 5
    wx = rng.choices(["clear", "hot", "light", "heavy"], [68, 8, 16, 8])[0]
    events = []
    if rng.random() < .22:
        t = rng.choice(["cricket", "concert", "festival", "exam"])
        start = {"cricket": rng.choice([14.5, 19.0]), "concert": rng.choice([19.0, 20.0]),
                 "festival": rng.choice([18.0, 19.5]), "exam": rng.choice([10.0, 14.0])}[t]
        dur = {"cricket": 3.5, "concert": 3.0, "festival": rng.choice([4.0, 5.0]), "exam": 3.0}[t]
        events = [{"type": t, "crowd": rng.choices(list(CROWD), [30, 35, 25, 10])[0],
                   "start": _hhmm(start), "end": _hhmm(start + dur)}]
    rows = []
    for i, (a, b, _) in enumerate(BANDS):
        loads = event_loads(events, a, b)
        ev_in = sum(p for p, k, _e in loads if k == "in")
        ev_out = sum(p for p, k, _e in loads if k == "out")
        base = (OFFDAY if off else TYPICAL)[i]
        truth = (base + PULL_IN[i] * ev_in + PULL_OUT[i] * ev_out) * (1 + WX_UP[wx][i])
        rows.append({"date": d.isoformat(), "band": i, "off": int(off), "light": int(wx == "light"),
                     "heavy": int(wx == "heavy"), "hot": int(wx == "hot"), "ev_in": round(ev_in, 2),
                     "ev_out": round(ev_out, 2), "demand": round(truth * math.exp(rng.gauss(0, .06)), 2)})
    return rows


def make_data(days=365):
    start = date(2025, 1, 1)
    return [r for k in range(days) for r in make_day(start + timedelta(k))]


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [{**r, **{k: float(r[k]) for k in FEATURES + ["demand"]}, "band": int(r["band"])}
                for r in csv.DictReader(f)]


def _solve(A, y):
    n = len(A)
    M = [row[:] + [y[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [x - f * z for x, z in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def fit(X, y, lam=1e-3):
    """Ridge regression by the normal equations."""
    k = len(X[0])
    A = [[sum(r[i] * r[j] for r in X) + (lam if i == j else 0) for j in range(k)] for i in range(k)]
    return _solve(A, [sum(r[i] * t for r, t in zip(X, y)) for i in range(k)])


def _x(r):
    return [1] + [r[f] for f in FEATURES]


def train(rows, source):
    cut = sorted({r["date"] for r in rows})[int(len({r["date"] for r in rows}) * .8)]
    err, act = [], []
    for i in range(len(BANDS)):
        sub = [r for r in rows if r["band"] == i]
        tr, te = [r for r in sub if r["date"] < cut], [r for r in sub if r["date"] >= cut]
        w = fit([_x(r) for r in tr], [r["demand"] for r in tr])
        err += [sum(a * b for a, b in zip(w, _x(r))) - r["demand"] for r in te]
        act += [r["demand"] for r in te]
    mean = sum(act) / len(act)
    r2 = 1 - sum(e * e for e in err) / sum((a - mean) ** 2 for a in act)
    intercept, coef = [], []
    for i in range(len(BANDS)):
        sub = [r for r in rows if r["band"] == i]
        w = fit([_x(r) for r in sub], [r["demand"] for r in sub])
        intercept.append(round(w[0], 2))
        coef.append([round(v, 3) for v in w[1:]])
    return {"source": source, "rows": len(rows), "features": FEATURES, "intercept": intercept, "coef": coef,
            "r2": round(r2, 3), "mae": round(sum(abs(e) for e in err) / len(err), 1)}


def main(argv):
    if "--data" in argv:
        rows, source = read_csv(argv[argv.index("--data") + 1]), "real"
    else:
        rows, source = make_data(), "synthetic"
        with open(os.path.join(HERE, "ridership_synthetic.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["date", "band"] + FEATURES + ["demand"])
            w.writeheader()
            w.writerows({k: r[k] for k in w.fieldnames} for r in rows)
    model = train(rows, source)
    with open(os.path.join(HERE, "model.json"), "w") as f:
        json.dump(model, f, indent=1)
    page = os.path.join(HERE, "index.html")
    html = open(page, encoding="utf-8").read()
    html = re.sub(r"/\*MODEL\*/.*?/\*END\*/", lambda m: "/*MODEL*/window.MODEL=%s;/*END*/" % json.dumps(model),
                  html, flags=re.S)
    open(page, "w", encoding="utf-8").write(html)
    print("Trained on %d %s rows. Held-out days: R2 %.3f, typical error %.1f demand points." % (
        len(rows), source, model["r2"], model["mae"]))
    for i, (_, _, name) in enumerate(BANDS):
        print("  %-13s weekday base %5.1f | " % (name, model["intercept"][i]) +
              ", ".join("%s %+.2f" % (f, v) for f, v in zip(FEATURES, model["coef"][i])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
