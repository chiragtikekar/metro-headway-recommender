import unittest
from recommender import DEFAULT_MODEL, load_model, recommend

def day(**kw):
    d = {"date": "2026-10-13", "holiday": False, "weather": "clear", "events": []}
    d.update(kw)
    return d

def rules(d):
    return recommend(d, DEFAULT_MODEL)

def by_name(res):
    return {b["name"]: b for b in res["bands"]}

CRICKET = [{"name": "Match", "type": "cricket", "crowd": "mega", "start": "19:00", "end": "22:30"}]

class Rules(unittest.TestCase):
    def test_plain_weekday_keeps_the_plan(self):
        res = rules(day())
        self.assertEqual([b["recommended_min"] for b in res["bands"]], [10, 6, 10, 6, 10])
        self.assertTrue(all(b["action"] == "keep" for b in res["bands"]))

    def test_weekend_lengthens_commuter_bands(self):
        b = by_name(rules(day(date="2026-10-17")))
        self.assertEqual(b["Early morning"]["action"], "lengthen")
        self.assertEqual(b["Early morning"]["recommended_min"], 16)
        self.assertEqual(b["Midday"]["action"], "shorten")

    def test_cricket_night_in_rain(self):
        b = by_name(rules(day(date="2026-10-16", weather="heavy", events=CRICKET)))
        self.assertEqual(b["Evening peak"]["action"], "shorten")
        self.assertLessEqual(b["Evening peak"]["recommended_min"], 4.5)
        self.assertLessEqual(b["Late evening"]["recommended_min"], 5)
        self.assertIn("Match", b["Late evening"]["reason"])

    def test_exam_day(self):
        ev = [{"name": "Exams", "type": "exam", "crowd": "large", "start": "10:00", "end": "13:00"}]
        b = by_name(rules(day(weather="light", events=ev)))
        self.assertLessEqual(b["Morning peak"]["recommended_min"], 5)
        self.assertLessEqual(b["Midday"]["recommended_min"], 6.5)

    def test_late_festival_flags_after_midnight_crowd(self):
        ev = [{"name": "Garba night", "type": "festival", "crowd": "large", "start": "19:30", "end": "23:30"}]
        self.assertTrue(any("Garba night" in n for n in rules(day(date="2026-10-17", events=ev))["notes"]))

    def test_headway_never_below_minimum(self):
        ev = [{"name": "Big", "type": "concert", "crowd": "mega", "start": "20:00", "end": "22:00"}] * 4
        self.assertTrue(all(b["recommended_min"] >= 3 for b in rules(day(weather="heavy", events=ev))["bands"]))

    def test_impact_numbers(self):
        m = rules(day(date="2026-10-16", weather="heavy", events=CRICKET))["impact"]
        self.assertGreater(m["trains_recommended"], m["trains_planned"])
        self.assertLess(m["over_capacity_recommended"], m["over_capacity_planned"])
        self.assertLess(m["avg_wait_recommended_min"], m["avg_wait_planned_min"])

class Trained(unittest.TestCase):
    def setUp(self):
        self.m = load_model()
        if self.m is DEFAULT_MODEL:
            self.skipTest("model.json not found; run python train_model.py")

    def test_learned_weekday_baseline_is_sane(self):
        for learned, typical in zip(self.m["intercept"], DEFAULT_MODEL["intercept"]):
            self.assertAlmostEqual(learned / typical, 1, delta=.1)

    def test_model_fits_unseen_days(self):
        self.assertGreater(self.m["r2"], .9)

    def test_cricket_night_with_learned_model(self):
        b = by_name(recommend(day(date="2026-10-16", weather="heavy", events=CRICKET), self.m))
        self.assertEqual(b["Late evening"]["action"], "shorten")
        self.assertLessEqual(b["Late evening"]["recommended_min"], 5)

if __name__ == "__main__":
    unittest.main()
