"""Headway recommender for PS-04, "No Day Is an Average Day".

Input:  next day's calendar (date, holiday flag, weather, events)
Output: a headway (minutes between trains) for each time band, with a
        plain-language reason a control room operator can act on.

Demand comes from a small linear model (model.json, learned by train_model.py).
If model.json is missing, built-in rules of the same shape are used.

Usage:  python recommender.py cricket_night.json [--json] [--rules]
Python 3.8+, no dependencies. index.html uses the same model and logic.
"""
import json
import math
import os
import sys
from datetime import date

BANDS = [(6, 8, "Early morning"), (8, 11, "Morning peak"), (11, 16, "Midday"),
         (16, 20, "Evening peak"), (20, 24, "Late evening")]
HOURS = [b - a for a, b, _ in BANDS]
TYPICAL = [55, 100, 40, 100, 45]   # demand index on a typical weekday (100 = peak)
OFFDAY = [20, 40, 60, 65, 55]      # weekends and public holidays
PLANNED = [10, 6, 10, 6, 10]       # planned headway in minutes (the "average day" plan)
CROWD = {"small": 10, "medium": 25, "large": 45, "mega": 70}
FEATURES = ["off", "light", "heavy", "hot", "ev_in", "ev_out"]
MIN_HEADWAY = 3                    # shortest safe gap, minutes
MAX_STRETCH = 1.6                  # never stretch beyond 1.6x the plan
OVER = 1.15                        # "over capacity" = demand 15% above what the plan carries
WEATHER_TEXT = {
    "heavy": "heavy rain pushes riders off roads and onto the metro",
    "light": "light rain nudges more people onto the metro",
    "hot": "the afternoon heat sends people to air-conditioned trains",
}
# Fallback when no trained model exists: the same linear shape, with hand-set effects.
DEFAULT_MODEL = {
    "source": "built-in rules", "features": FEATURES, "intercept": list(TYPICAL),
    "coef": [[OFFDAY[i] - t, .08 * t, (.18 if i in (1, 3) else .12) * t,
              .06 * t if i == 2 else 0, 1, 1] for i, t in enumerate(TYPICAL)],
}


def load_model(path=None):
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.json")
    try:
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        assert m["features"] == FEATURES and len(m["coef"]) == len(BANDS)
        return m
    except (OSError, ValueError, KeyError, AssertionError):
        return DEFAULT_MODEL


def _round(x):
    return math.floor(x + 0.5)


def _count(n, word):
    return "%d %s%s" % (n, word, "" if n == 1 else "s")


def _hours(t):
    h, m = t.split(":")
    return int(h) + int(m) / 60


def _clock(h):
    h %= 24
    a = int(h)
    m = _round((h - a) * 60)
    if m == 60:
        a, m = a + 1, 0
    return "%02d:%02d" % (a % 24, m)


def _windows(e):
    """Time windows when an event moves riders: (from, to, weight, direction)."""
    s, end = _hours(e["start"]), _hours(e["end"])
    if end <= s:
        end += 24
    if e["type"] == "festival":
        return [(s, end, .7, "in"), (end, end + 1.5, .3, "out")]
    if e["type"] == "exam":
        return [(s - 1.5, s, .5, "in"), (end, end + 1.5, .5, "out")]
    return [(s - 3, s, .4, "in"), (end, end + 1.5, .6, "out")]  # cricket, concert


def event_loads(events, a, b):
    """Crowd points that events put on the band [a, b): list of (points, direction, event)."""
    out = []
    for e in events:
        for w0, w1, wt, kind in _windows(e):
            overlap = max(0, min(w1, b) - max(w0, a))
            if overlap > 0:
                out.append((CROWD[e["crowd"]] * wt * overlap / (w1 - w0), kind, e))
    return out


def _event_text(e, kind):
    n = (e.get("name") or "").strip() or "the event"
    if kind == "in":
        if e["type"] == "festival":
            return "crowds at " + n
        who = "students" if e["type"] == "exam" else "people"
        return "%s heading to %s (starts %s)" % (who, n, e["start"])
    if e["type"] == "exam":
        return "%s finishes at %s and everyone leaves together" % (n, e["end"])
    return "%s ends around %s and the crowd heads home" % (n, e["end"])


def recommend(day, model=None):
    model = model or load_model()
    d = date.fromisoformat(day["date"])
    holiday = bool(day.get("holiday"))
    off = holiday or d.weekday() >= 5
    label = "public holiday" if holiday else "weekend"
    wx = day.get("weather", "clear")
    events = day.get("events", [])
    bands, notes, rows = [], [], []

    for i, (a, b, name) in enumerate(BANDS):
        c = model["coef"][i]
        demand = model["intercept"][i]
        why = []
        if off:
            demand += c[0]
            if abs(c[0]) >= 5:
                why.append((c[0], label + (" commuting is lighter than the weekday plan assumes" if c[0] < 0
                                          else " leisure trips run higher than a normal weekday at this hour")))
        wi = {"light": 1, "heavy": 2, "hot": 3}.get(wx)
        if wi:
            demand += c[wi]
            if c[wi] >= 3:
                why.append((c[wi], WEATHER_TEXT[wx]))
        for pts, kind, e in event_loads(events, a, b):
            p = c[4 if kind == "in" else 5] * pts
            demand += p
            why.append((p, _event_text(e, kind)))
        demand = max(demand, 5.0)

        ratio = demand / model["intercept"][i]
        planned = PLANNED[i]
        h = planned
        if ratio >= 1.1 or ratio <= .75:
            h = _round(min(max(planned / ratio, MIN_HEADWAY), planned * MAX_STRETCH) * 2) / 2
        action = "shorten" if h < planned else "lengthen" if h > planned else "keep"
        pct = _round((ratio - 1) * 100)
        if action == "shorten":
            todo = "Run a train every %g min instead of %g. That adds about %s an hour." % (
                h, planned, _count(_round(60 / h - 60 / planned), "train"))
        elif action == "lengthen":
            todo = "Stretch to a train every %g min instead of %g. That frees about %s an hour with little crowding risk." % (
                h, planned, _count(_round(60 / planned - 60 / h), "trip"))
        else:
            todo = "Keep the planned %g-minute gap. Demand is close to what the plan assumes." % planned
        why.sort(key=lambda x: -abs(x[0]))
        top = [t for p, t in why if abs(p) >= 3][:2]
        if top:
            reason = "Expect %d%% %s riders than planned: %s." % (
                abs(pct), "more" if pct >= 0 else "fewer", ", and ".join(top))
        else:
            reason = "Nothing on the calendar moves demand much in this band."
        rows.append((i, demand, ratio, h, planned))
        bands.append({
            "band": "%s-%s" % (_clock(a), _clock(b)), "name": name,
            "planned_min": planned, "recommended_min": h, "action": action,
            "demand_vs_plan_pct": pct, "instruction": todo,
            "reason": reason[0].upper() + reason[1:],
        })

    for e in events:
        n = (e.get("name") or "").strip() or "the event"
        if any(w[1] > 24.01 and w[3] == "out" for w in _windows(e)):
            notes.append("Crowd from %s is still leaving after midnight. Consider a late special service." % n)
    return {"date": day["date"], "day_type": label if off else "weekday", "weather": wx,
            "model": model["source"], "bands": bands, "notes": notes, "impact": _impact(rows)}


def _impact(rows):
    """Day-level effect of following the recommendations, from the model's own demand."""
    tp = tr = wp = wr = den = 0.0
    over_p = over_r = 0
    for i, demand, ratio, h, planned in rows:
        w = demand * HOURS[i]
        tp += HOURS[i] * 60 / planned
        tr += HOURS[i] * 60 / h
        den += w
        wp += w * planned / 2   # average wait is half the gap
        wr += w * h / 2
        over_p += ratio > OVER
        over_r += ratio * h / planned > OVER
    return {"trains_planned": _round(tp), "trains_recommended": _round(tr),
            "avg_wait_planned_min": round(wp / den, 1), "avg_wait_recommended_min": round(wr / den, 1),
            "over_capacity_planned": over_p, "over_capacity_recommended": over_r}


def format_text(res):
    out = ["Headway plan for %s (%s, weather: %s, demand model: %s)" % (
        res["date"], res["day_type"], res["weather"], res["model"]), ""]
    for b in res["bands"]:
        out.append("%s  %-13s  %4g -> %-4g min  [%s]" % (
            b["band"], b["name"], b["planned_min"], b["recommended_min"], b["action"].upper()))
        out.append("    " + b["instruction"])
        out.append("    " + b["reason"])
    m = res["impact"]
    out += ["", "Day total: %d -> %d trains, average wait %.1f -> %.1f min, bands over capacity %d -> %d" % (
        m["trains_planned"], m["trains_recommended"], m["avg_wait_planned_min"],
        m["avg_wait_recommended_min"], m["over_capacity_planned"], m["over_capacity_recommended"])]
    out += ["", *("NOTE: " + n for n in res["notes"])]
    return "\n".join(out)


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    with open(argv[1], encoding="utf-8") as f:
        res = recommend(json.load(f), DEFAULT_MODEL if "--rules" in argv else None)
    print(json.dumps(res, indent=2) if "--json" in argv else format_text(res))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
