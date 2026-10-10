# Metro Headway Recommender

**PS-04 · No Day Is an Average Day** (DevFest Gandhinagar, Metro Hackathon, Metro Operations track)
Built by Chirag.

## Problem

Train headways are planned for a typical day. Exams, festivals, cricket matches and rain move demand a lot, so some days trains are overcrowded and others run nearly empty.

## What it does

Give it tomorrow's calendar: date, holiday, weather forecast and events. It returns a recommended headway for five time bands, each with a plain-language reason and an instruction a control room operator can act on. It also totals the effect of following the advice: trains in the day, average passenger wait, and how many time bands would be over capacity.

Example (`cricket_night.json`: cricket match 19:00–22:30, heavy rain, Friday; last band and day total shown):

```
20:00-00:00  Late evening     10 -> 4 min  [SHORTEN]
    Run a train every 4 min instead of 10. That adds about 9 trains an hour.
    Expect 143% more riders than planned: Cricket match, Narendra Modi Stadium ends around 22:30 and the crowd heads home, and heavy rain pushes riders off roads and onto the metro.

Day total: 136 -> 201 trains, average wait 3.9 -> 2.6 min, bands over capacity 3 -> 0
```

It is for metro staff (control room and planners), not passengers.

## Run it

Nothing needs installing.

1. **Double-click `index.html`.** Works in any browser, with no network.
2. **`python run.py`** starts a local server and opens the app.
3. **Command line:** `python recommender.py cricket_night.json` (`--json` for machine-readable output, `--rules` to use the built-in rules instead of the trained model).

Tests: `python -m unittest -v`

## How it works

1. **Demand model.** For each time band, a linear model predicts the demand index (100 = a typical weekday peak) from: weekend or holiday, light rain, heavy rain, heat, and the crowd points that events add on the way in and on the way out. `train_model.py` fits it by ridge regression (pure Python) and saves `model.json`; the same numbers are embedded in `index.html`.
2. **Events.** Each event has an inbound window (people heading there) and an outbound window (the crowd leaving), scaled by crowd size and split across the bands those windows touch.
3. **Headway** = planned headway ÷ (predicted demand ÷ typical demand), rounded to 0.5 min, never below 3 min, never stretched past 1.6× the plan. Demand within about −25% to +10% of normal keeps the plan.
4. **Explanation.** Because the model is linear, every feature's contribution is known exactly. The two biggest become the sentence the operator reads. Nothing is a black box.

## The data (read this)

No ridership data was provided, so `ridership_synthetic.csv` is **synthetic**: one year of days × 5 bands, generated from stated assumptions (weekday and weekend shapes, event pull per band, weather effects, 6% noise). The model never sees those assumptions; it has to learn them, and it is scored on the last 20% of days it did not train on (R² 0.98, typical error 2.9 demand points). Because the world is simulated, a good score shows the pipeline works, not that the numbers match Gujarat Metro.

To use real data, make a CSV with the columns listed at the top of `train_model.py`, then run `python train_model.py --data ridership.csv`. The app and the command line pick up the new model with no other change.

## Works with poor or no network

- One HTML file, no external requests, no fonts or scripts to download.
- The app shows an online/offline badge and keeps working either way.
- Your last inputs are remembered on the device, so a refresh or a dropped connection loses nothing.
- If the trained model is missing, the app falls back to built-in rules and says so on screen.
- Weather and events are typed in, so no forecast feed is needed. A feed can be added later without changing the flow.
- What needs the network: nothing in the core flow. Cloning the repo needs internet once, and a hosted demo link (if any) needs internet. The copy on your laptop runs fully offline.

## Files

| Path | Purpose |
|---|---|
| `index.html` | Web app (single file, works offline) |
| `recommender.py` | Same engine in Python, with a command-line interface |
| `train_model.py` | Builds synthetic history, trains the model, or trains on real data |
| `model.json`, `ridership_synthetic.csv` | Trained model and the synthetic dataset |
| `run.py` | One-command local launcher |
| `cricket_night.json`, `exam_day.json`, `festival_night.json` | Three example days as JSON |
| `test_recommender.py` | Unit tests |
| `PITCH.md` | Pitch, PPP model, demo script and likely judge questions |

## Built before the ride

Everything in this repository was prepared before the ride, with help from Claude (an AI assistant): the web app, the Python engine, the training script, the synthetic dataset, the tests and the docs.

## Limits and next steps

- The model is trained on synthetic data. Next: calibrate on the operator's ticketing and passenger-count data, in shadow mode first.
- Planned headways (10 min off-peak, 6 min peak) and the 3 min minimum are placeholders for the real timetable and signalling limits.
- Weather and events are typed in. Next: a forecast feed and a venue calendar.
- Recommendations are advice for a human operator, not automatic timetable changes.
