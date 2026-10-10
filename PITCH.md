# Pitch: No Day Is an Average Day

## In 30 seconds

The metro runs one timetable for very different days. On a cricket night in the rain the last trains overflow; on a quiet weekend they run nearly empty. Our tool takes tomorrow's calendar (events, weather, holiday) and tells the control room operator, for each time band, how far to tighten or stretch the gap between trains, and why, in one plain sentence.

## Why an operator can use it

- Output is a headway per time band, the exact thing a control room changes.
- Every recommendation names its cause ("the match ends around 22:30 and the crowd heads home").
- It warns when a crowd will still be leaving after the last train.
- Advice only. A human decides, and the 3 min safety minimum is built in.

## Impact the tool measures for any day

For each day the app shows three numbers, plan versus recommended: trains in the day, average passenger wait, and time bands over capacity. For the cricket-night sample: 136 → 201 trains, wait 3.9 → 2.6 min, over-capacity bands 3 → 0. These come from the demand model's own predictions on synthetic data, so quote them as an illustration, not a forecast for the real network.

## Resilience

Runs from a single file with no network. It shows online/offline status, remembers the last inputs, and falls back to built-in rules if the trained model is missing. Training a new model needs nothing but Python.

## PPP: who builds, runs and pays

- **Metro authority** owns the data (ticketing, passenger counts, timetable) and the operating decision. It runs the tool in the control room.
- **Private partner** (an analytics or transport-software firm) builds and maintains the model and the data pipeline, and keeps it accurate as the network changes.
- **Who pays:** a small fixed pilot fee, then a gain-share: the partner is paid from measured improvements, such as fewer overcrowded trips, shorter average waits, and fewer unneeded trips on quiet days. The metro pays only for results it can see on its own data.
- **Rollout:** (1) shadow mode, where recommendations are logged next to what actually ran, and the model is calibrated on real data; (2) advisory use for event days only; (3) daily use, linked to the scheduling system.
- **Risks and answers:** data access (the metro keeps ownership, the partner works under agreement); model drift (retrain monthly with `train_model.py`); safety (advice only, fixed minimum headway).

## Likely questions

- **Where is the AI?** A model learned from data predicts demand for each band, and its effects are readable. For a safety-sensitive decision an operator must trust, an explainable model beats a chatbot.
- **Is the data real?** No. None was provided, so it is synthetic, from stated assumptions. The pipeline is ready to retrain on real ridership with one command.
- **What if there is no network?** Nothing changes. Everything runs on the device.
- **Why not just use last year's same day?** Events and weather don't repeat on the same calendar day. The model reacts to the cause.

## 3-minute demo

1. (20 s) State the problem: one timetable, very different days.
2. (50 s) Load **Cricket night in the rain.** Point to the evening and late-evening bands tightening, read one reason aloud, then show the day totals.
3. (30 s) Load **Exam day.** Morning and midday change; the evening stays the same.
4. (30 s) Load **Festival night.** Show the warning about the crowd still leaving after midnight.
5. (20 s) Switch the weather to clear and show the numbers move.
6. (30 s) Turn off Wi-Fi, refresh, and show it still works with the offline badge. Close with the PPP model and the plan to calibrate on real ridership.
