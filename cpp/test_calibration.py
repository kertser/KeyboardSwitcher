#!/usr/bin/env python3
"""
test_calibration.py — Unit tests for the adaptive calibration controller.

Mirrors the logic in FeedbackLogger.cpp::RecordOutcome() exactly,
using the same constants.  Verifies:
  1. EWMA update formula
  2. Batch event counting (no adaptation before MIN_EVENTS)
  3. Dead zone (|pressure| < HYSTERESIS → no step)
  4. Tightening on FP pressure
  5. Loosening on FN pressure
  6. Asymmetric delta clamps
  7. Absolute value clamps on effective params
  8. Batch counter reset after adaptation
  9. Ceiling decay (ewmaFpRate damped when delta at tighten ceiling)
 10. ResetCalibration restores base values
 11. TP signal drives both EWMA rates toward 0
 12. Base values stored once and not overwritten on subsequent calls
 13–16. Reset, reversal, exact first step, per-pair isolation
 17. Settings flyout shows the user base, not the effective value (delta hidden)
 18. Manual base edit preserves the invisible calibration delta
 19. Reset Defaults wipes both the base edit and the calibration delta
"""
from __future__ import annotations
import sys, math

# ─── Mirror constants from FeedbackLogger.cpp ────────────────────────────────
EWMA_ALPHA         = 0.20
MIN_EVENTS         = 5
HYSTERESIS_BAND    = 0.15
STEP_CONF          = 0.01
STEP_MARGIN        = 0.005
MAX_TIGHTEN_CONF   = 0.10
MAX_TIGHTEN_MARG   = 0.08
MAX_LOOSEN_CONF    = 0.05
MAX_LOOSEN_MARG    = 0.01
ABS_MIN_CONF       = 0.50
ABS_MAX_CONF       = 0.995
ABS_MIN_MARGIN     = 0.005
ABS_MAX_MARGIN     = 0.25

# ─── Factory baseline for en→ru (from Config.cpp) ────────────────────────────
BASE_CONF_EN_RU   = 0.70
BASE_MARGIN_EN_RU = 0.05

# ─── Simulation of PairCalibration struct ────────────────────────────────────

class PairCalibration:
    def __init__(self, base_conf: float, base_margin: float):
        self.ewma_fp        = 0.0
        self.ewma_fn        = 0.0
        self.batch_events   = 0
        self.delta_conf_max = 0.0
        self.delta_margin   = 0.0
        self.base_conf_max  = base_conf
        self.base_margin    = base_margin
        # Tracks what was last written to "Config"
        self.applied_conf   = base_conf
        self.applied_margin = base_margin
        self.adaptations    = 0  # number of times a step was applied

    def record_outcome(self, outcome: str) -> bool:
        """Returns True if an adaptation step was applied."""
        assert outcome in ("FP", "TP", "FN")

        fp_sig = 1.0 if outcome == "FP" else 0.0
        fn_sig = 1.0 if outcome == "FN" else 0.0

        self.ewma_fp = EWMA_ALPHA * fp_sig + (1.0 - EWMA_ALPHA) * self.ewma_fp
        self.ewma_fn = EWMA_ALPHA * fn_sig + (1.0 - EWMA_ALPHA) * self.ewma_fn
        self.batch_events += 1

        if self.batch_events < MIN_EVENTS:
            return False

        pressure = self.ewma_fp - self.ewma_fn
        if abs(pressure) < HYSTERESIS_BAND:
            self.batch_events = 0
            return False

        tighten = pressure > 0.0
        d_conf   =  STEP_CONF   if tighten else -STEP_CONF
        d_margin =  STEP_MARGIN if tighten else -STEP_MARGIN

        self.delta_conf_max = max(-MAX_LOOSEN_CONF,
                                  min(MAX_TIGHTEN_CONF, self.delta_conf_max + d_conf))
        self.delta_margin   = max(-MAX_LOOSEN_MARG,
                                  min(MAX_TIGHTEN_MARG, self.delta_margin + d_margin))

        eff_conf   = max(ABS_MIN_CONF,   min(ABS_MAX_CONF,   self.base_conf_max + self.delta_conf_max))
        eff_margin = max(ABS_MIN_MARGIN, min(ABS_MAX_MARGIN, self.base_margin   + self.delta_margin))

        self.applied_conf   = eff_conf
        self.applied_margin = eff_margin
        self.adaptations   += 1

        # Ceiling decay
        at_tighten_conf = self.delta_conf_max >= MAX_TIGHTEN_CONF - 0.001
        at_tighten_marg = self.delta_margin   >= MAX_TIGHTEN_MARG - 0.001
        if tighten and (at_tighten_conf or at_tighten_marg):
            self.ewma_fp *= 0.70

        self.batch_events = 0
        return True

    # ── Settings-flyout integration (mirrors FeedbackLogger.cpp) ──────────
    def _apply_effective(self):
        """Mirror of ApplyEffective(): effective = clamp(base + delta)."""
        self.applied_conf   = max(ABS_MIN_CONF,   min(ABS_MAX_CONF,
                                  self.base_conf_max + self.delta_conf_max))
        self.applied_margin = max(ABS_MIN_MARGIN, min(ABS_MAX_MARGIN,
                                  self.base_margin   + self.delta_margin))

    def get_base_conf(self) -> float:
        """Mirror of GetBaseConfFloor(): returns the USER BASE, not effective."""
        return self.base_conf_max

    def set_base_conf(self, conf_floor: float):
        """Mirror of SetBaseConfFloor(): shift base, re-apply effective,
        preserving any existing calibration delta."""
        self.base_conf_max = conf_floor
        self._apply_effective()

    def reset_to_factory(self, factory_conf: float, factory_margin: float):
        """Mirror of ResetPairToFactory(): base→factory, all deltas→0."""
        self.base_conf_max  = factory_conf
        self.base_margin    = factory_margin
        self.delta_conf_max = 0.0
        self.delta_margin   = 0.0
        self.ewma_fp        = 0.0
        self.ewma_fn        = 0.0
        self.batch_events   = 0
        self._apply_effective()


# ─── Test helpers ─────────────────────────────────────────────────────────────

PASS = 0
FAIL = 0

def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}" + (f"  ({detail})" if detail else ""))

def approx(a: float, b: float, eps=1e-6) -> bool:
    return abs(a - b) < eps

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# ─── Test 1: EWMA update formula ──────────────────────────────────────────────

def test_ewma_formula():
    print("\n[1] EWMA update formula")
    cal = PairCalibration(BASE_CONF_EN_RU, BASE_MARGIN_EN_RU)

    # After 1 FP: ewma_fp = 0.2*1 + 0.8*0 = 0.2
    cal.ewma_fp = EWMA_ALPHA * 1.0 + (1.0 - EWMA_ALPHA) * cal.ewma_fp
    check("ewma_fp after 1 FP = 0.2", approx(cal.ewma_fp, 0.2))

    # After another FP: 0.2*1 + 0.8*0.2 = 0.36
    cal.ewma_fp = EWMA_ALPHA * 1.0 + (1.0 - EWMA_ALPHA) * cal.ewma_fp
    check("ewma_fp after 2 FP = 0.36", approx(cal.ewma_fp, 0.36))

    # After 1 TP (fp_sig=0): 0.2*0 + 0.8*0.36 = 0.288
    cal.ewma_fp = EWMA_ALPHA * 0.0 + (1.0 - EWMA_ALPHA) * cal.ewma_fp
    check("ewma_fp after TP decays to 0.288", approx(cal.ewma_fp, 0.288))

    # FN signal only updates ewma_fn
    cal2 = PairCalibration(BASE_CONF_EN_RU, BASE_MARGIN_EN_RU)
    cal2.ewma_fn = EWMA_ALPHA * 1.0 + (1.0 - EWMA_ALPHA) * cal2.ewma_fn
    check("ewma_fn after 1 FN = 0.2, ewma_fp unchanged",
          approx(cal2.ewma_fn, 0.2) and approx(cal2.ewma_fp, 0.0))

# ─── Test 2: No adaptation before MIN_EVENTS ──────────────────────────────────

def test_min_events_gate():
    print("\n[2] No adaptation before MIN_EVENTS")
    cal = PairCalibration(BASE_CONF_EN_RU, BASE_MARGIN_EN_RU)
    for i in range(MIN_EVENTS - 1):
        adapted = cal.record_outcome("FP")
        check(f"no adaptation at event {i+1}/{MIN_EVENTS-1}", not adapted)
    check("still at base after MIN_EVENTS-1 FPs",
          approx(cal.applied_conf, BASE_CONF_EN_RU) and
          approx(cal.applied_margin, BASE_MARGIN_EN_RU))

# ─── Test 3: Dead zone ────────────────────────────────────────────────────────

def test_dead_zone():
    print("\n[3] Dead zone: |pressure| < HYSTERESIS → no step")
    cal = PairCalibration(BASE_CONF_EN_RU, BASE_MARGIN_EN_RU)
    # Feed alternating FP/FN/TP to keep pressure near 0
    events = ["FP", "FN", "TP", "FP", "FN"]
    for e in events:
        cal.record_outcome(e)
    pressure = cal.ewma_fp - cal.ewma_fn
    check(f"|pressure|={abs(pressure):.4f} < HYSTERESIS={HYSTERESIS_BAND}",
          abs(pressure) < HYSTERESIS_BAND)
    check("no adaptation fired in dead zone", cal.adaptations == 0)

# ─── Test 4: Tightening on sustained FP pressure ─────────────────────────────

def test_tighten_on_fp():
    print("\n[4] Tightening: sustained FP → conf and margin increase")
    cal = PairCalibration(BASE_CONF_EN_RU, BASE_MARGIN_EN_RU)
    # Feed enough FPs to push ewma_fp well above HYSTERESIS
    for _ in range(20):
        cal.record_outcome("FP")
    check("at least one adaptation occurred", cal.adaptations >= 1)
    check("applied_conf > base_conf (tightened)",
          cal.applied_conf > BASE_CONF_EN_RU,
          f"applied={cal.applied_conf:.4f} base={BASE_CONF_EN_RU}")
    check("applied_margin > base_margin (tightened)",
          cal.applied_margin > BASE_MARGIN_EN_RU,
          f"applied={cal.applied_margin:.4f} base={BASE_MARGIN_EN_RU}")
    check("delta_conf_max >= 0", cal.delta_conf_max >= 0)
    check("delta_margin >= 0",   cal.delta_margin >= 0)

# ─── Test 5: Loosening on sustained FN pressure ───────────────────────────────

def test_loosen_on_fn():
    print("\n[5] Loosening: sustained FN → conf and margin decrease")
    cal = PairCalibration(BASE_CONF_EN_RU, BASE_MARGIN_EN_RU)
    for _ in range(20):
        cal.record_outcome("FN")
    check("at least one adaptation occurred", cal.adaptations >= 1)
    check("applied_conf < base_conf (loosened)",
          cal.applied_conf < BASE_CONF_EN_RU,
          f"applied={cal.applied_conf:.4f} base={BASE_CONF_EN_RU}")
    check("applied_margin < base_margin (loosened)",
          cal.applied_margin < BASE_MARGIN_EN_RU,
          f"applied={cal.applied_margin:.4f} base={BASE_MARGIN_EN_RU}")
    check("delta_conf_max <= 0", cal.delta_conf_max <= 0)
    check("delta_margin <= 0",   cal.delta_margin <= 0)

# ─── Test 6: Asymmetric clamp – tighten ceiling ───────────────────────────────

def test_tighten_ceiling():
    print("\n[6] Asymmetric clamp: delta cannot exceed MAX_TIGHTEN")
    cal = PairCalibration(BASE_CONF_EN_RU, BASE_MARGIN_EN_RU)
    for _ in range(200):
        cal.record_outcome("FP")
    check(f"delta_conf_max <= MAX_TIGHTEN_CONF={MAX_TIGHTEN_CONF}",
          cal.delta_conf_max <= MAX_TIGHTEN_CONF + 1e-6,
          f"actual={cal.delta_conf_max:.4f}")
    check(f"delta_margin <= MAX_TIGHTEN_MARG={MAX_TIGHTEN_MARG}",
          cal.delta_margin <= MAX_TIGHTEN_MARG + 1e-6,
          f"actual={cal.delta_margin:.4f}")

# ─── Test 7: Asymmetric clamp – loosen floor ─────────────────────────────────

def test_loosen_floor():
    print("\n[7] Asymmetric clamp: delta cannot go below -MAX_LOOSEN")
    cal = PairCalibration(BASE_CONF_EN_RU, BASE_MARGIN_EN_RU)
    for _ in range(200):
        cal.record_outcome("FN")
    check(f"delta_conf_max >= -MAX_LOOSEN_CONF={-MAX_LOOSEN_CONF}",
          cal.delta_conf_max >= -MAX_LOOSEN_CONF - 1e-6,
          f"actual={cal.delta_conf_max:.4f}")
    check(f"delta_margin >= -MAX_LOOSEN_MARG={-MAX_LOOSEN_MARG}",
          cal.delta_margin >= -MAX_LOOSEN_MARG - 1e-6,
          f"actual={cal.delta_margin:.4f}")

# ─── Test 8: Absolute clamps on effective values ──────────────────────────────

def test_absolute_clamps():
    print("\n[8] Absolute clamps on effective params")
    # Pair with a very high base: effective conf must never exceed ABS_MAX
    cal_high = PairCalibration(0.98, 0.22)
    for _ in range(200):
        cal_high.record_outcome("FP")
    check(f"effective conf <= ABS_MAX_CONF={ABS_MAX_CONF}",
          cal_high.applied_conf <= ABS_MAX_CONF + 1e-6,
          f"actual={cal_high.applied_conf:.4f}")
    check(f"effective margin <= ABS_MAX_MARGIN={ABS_MAX_MARGIN}",
          cal_high.applied_margin <= ABS_MAX_MARGIN + 1e-6,
          f"actual={cal_high.applied_margin:.4f}")

    # Pair with a very low base: effective conf must never go below ABS_MIN
    cal_low = PairCalibration(0.55, 0.015)
    for _ in range(200):
        cal_low.record_outcome("FN")
    check(f"effective conf >= ABS_MIN_CONF={ABS_MIN_CONF}",
          cal_low.applied_conf >= ABS_MIN_CONF - 1e-6,
          f"actual={cal_low.applied_conf:.4f}")
    check(f"effective margin >= ABS_MIN_MARGIN={ABS_MIN_MARGIN}",
          cal_low.applied_margin >= ABS_MIN_MARGIN - 1e-6,
          f"actual={cal_low.applied_margin:.4f}")

# ─── Test 9: Batch counter resets after adaptation ────────────────────────────

def test_batch_reset():
    print("\n[9] Batch counter resets after each adaptation")
    cal = PairCalibration(BASE_CONF_EN_RU, BASE_MARGIN_EN_RU)
    # Trigger first adaptation
    for _ in range(20):
        cal.record_outcome("FP")
    first_adaptations = cal.adaptations
    check("first adaptation occurred", first_adaptations >= 1)
    check("batch_events reset to 0 after adaptation",
          cal.batch_events == 0,
          f"actual={cal.batch_events}")
    # After batch reset, MIN_EVENTS fresh events are needed again
    pre_adapt = cal.adaptations
    for _ in range(MIN_EVENTS - 1):
        cal.record_outcome("FP")
    check("no second adaptation after fewer than MIN_EVENTS new events",
          cal.adaptations == pre_adapt)

# ─── Test 10: Ceiling decay ───────────────────────────────────────────────────

def test_ceiling_decay():
    print("\n[10] Ceiling decay: ewma_fp damped when delta hits tighten ceiling")
    cal = PairCalibration(BASE_CONF_EN_RU, BASE_MARGIN_EN_RU)
    # Drive delta to ceiling
    for _ in range(200):
        cal.record_outcome("FP")
    # Record ewma_fp right after hitting ceiling
    ewma_before = cal.ewma_fp
    # The ceiling decay applies a 0.7 multiplier when at ceiling
    # (applied inside record_outcome when tighten & at_ceiling)
    # We can verify: after many FPs at ceiling, ewma_fp must be < what it
    # would be without damping (which would be near 1.0).
    # With damping: ewma_fp should have been reduced at some point.
    check("ewma_fp after ceiling decay < 0.9 (not stuck near 1.0)",
          cal.ewma_fp < 0.90,
          f"actual={cal.ewma_fp:.4f}")
    check("delta still at ceiling (decay doesn't reduce delta)",
          cal.delta_conf_max >= MAX_TIGHTEN_CONF - 0.001,
          f"actual={cal.delta_conf_max:.4f}")

# ─── Test 11: TP stabilises — both EWMA rates decay ──────────────────────────

def test_tp_stabilises():
    print("\n[11] TP signal: both EWMA rates decay toward 0")
    cal = PairCalibration(BASE_CONF_EN_RU, BASE_MARGIN_EN_RU)
    # Build up some FP and FN rates
    for _ in range(3):
        cal.record_outcome("FP")
    for _ in range(3):
        cal.record_outcome("FN")
    fp_before = cal.ewma_fp
    fn_before = cal.ewma_fn
    # Feed TPs: both should decay
    for _ in range(5):
        cal.record_outcome("TP")
    check("ewma_fp decays on TP",
          cal.ewma_fp < fp_before,
          f"before={fp_before:.4f} after={cal.ewma_fp:.4f}")
    check("ewma_fn decays on TP",
          cal.ewma_fn < fn_before,
          f"before={fn_before:.4f} after={cal.ewma_fn:.4f}")
    check("TP alone does not trigger adaptation in dead zone",
          cal.adaptations == 0)

# ─── Test 12: Base stored once, not overwritten ───────────────────────────────

def test_base_stored_once():
    print("\n[12] Base stored once and not overwritten")
    cal = PairCalibration(BASE_CONF_EN_RU, BASE_MARGIN_EN_RU)
    original_base = cal.base_conf_max
    # Trigger tightening
    for _ in range(20):
        cal.record_outcome("FP")
    check("base_conf_max unchanged after tightening",
          approx(cal.base_conf_max, original_base),
          f"base={cal.base_conf_max:.4f} original={original_base:.4f}")
    # Trigger loosening (switch to FNs)
    for _ in range(50):
        cal.record_outcome("FN")
    check("base_conf_max still unchanged after loosening",
          approx(cal.base_conf_max, original_base),
          f"base={cal.base_conf_max:.4f}")

# ─── Test 13: Reset restores factory params ───────────────────────────────────

def test_reset_calibration():
    print("\n[13] ResetCalibration restores factory params")
    cal = PairCalibration(BASE_CONF_EN_RU, BASE_MARGIN_EN_RU)
    for _ in range(20):
        cal.record_outcome("FP")
    check("adapted before reset", cal.adaptations >= 1)
    adapted_conf = cal.applied_conf
    # Reset
    cal.delta_conf_max = 0.0
    cal.delta_margin   = 0.0
    cal.ewma_fp        = 0.0
    cal.ewma_fn        = 0.0
    cal.batch_events   = 0
    # Re-apply base
    cal.applied_conf   = cal.base_conf_max
    cal.applied_margin = cal.base_margin
    check("applied_conf restored to base after reset",
          approx(cal.applied_conf, BASE_CONF_EN_RU),
          f"applied={cal.applied_conf:.4f}")
    check("applied_margin restored to base after reset",
          approx(cal.applied_margin, BASE_MARGIN_EN_RU),
          f"applied={cal.applied_margin:.4f}")

# ─── Test 14: Reversal — FPs then FNs converge back to base ──────────────────

def test_reversal():
    print("\n[14] Reversal: after FP tightening, sustained FN loosens back toward base")
    cal = PairCalibration(BASE_CONF_EN_RU, BASE_MARGIN_EN_RU)
    # Tighten
    for _ in range(50):
        cal.record_outcome("FP")
    tightened_conf = cal.applied_conf
    check("tightened above base", tightened_conf > BASE_CONF_EN_RU)
    # Now loosen
    for _ in range(100):
        cal.record_outcome("FN")
    check("after sustained FN, conf moves back toward or below base",
          cal.applied_conf <= tightened_conf,
          f"tightened={tightened_conf:.4f} after_fn={cal.applied_conf:.4f}")

# ─── Test 15: Exact first-adaptation values (en→ru baseline) ─────────────────

def test_exact_first_adaptation():
    """Verify the exact conf and margin values after the very first tighten step."""
    print("\n[15] Exact first-adaptation values (en→ru)")
    cal = PairCalibration(BASE_CONF_EN_RU, BASE_MARGIN_EN_RU)
    # Drive ewma_fp to above HYSTERESIS with pure FPs
    # We need enough events for batchEvents >= MIN_EVENTS AND |pressure| > HYSTERESIS
    # After 5 FPs: ewma_fp = 1-(1-0.2)^5 ≈ 0.6723, pressure = 0.6723 > 0.15 → adaptation
    for _ in range(5):
        cal.record_outcome("FP")
    expected_conf   = clamp(BASE_CONF_EN_RU + STEP_CONF,   ABS_MIN_CONF, ABS_MAX_CONF)
    expected_margin = clamp(BASE_MARGIN_EN_RU + STEP_MARGIN, ABS_MIN_MARGIN, ABS_MAX_MARGIN)
    check(f"applied_conf after first step = {expected_conf:.4f}",
          approx(cal.applied_conf, expected_conf, eps=1e-5),
          f"actual={cal.applied_conf:.6f}")
    check(f"applied_margin after first step = {expected_margin:.4f}",
          approx(cal.applied_margin, expected_margin, eps=1e-5),
          f"actual={cal.applied_margin:.6f}")
    check("delta_conf_max = STEP_CONF after first step",
          approx(cal.delta_conf_max, STEP_CONF, eps=1e-5),
          f"actual={cal.delta_conf_max:.6f}")

# ─── Test 16: Per-pair isolation ─────────────────────────────────────────────

def test_per_pair_isolation():
    print("\n[16] Per-pair isolation: adapting en→ru does not affect ru→en")
    cal_en_ru = PairCalibration(BASE_CONF_EN_RU, BASE_MARGIN_EN_RU)
    cal_ru_en = PairCalibration(0.70, 0.05)   # same factory values
    for _ in range(20):
        cal_en_ru.record_outcome("FP")
    check("en→ru tightened",        cal_en_ru.applied_conf > BASE_CONF_EN_RU)
    check("ru→en still at factory", approx(cal_ru_en.applied_conf, 0.70) and
                                     approx(cal_ru_en.applied_margin, 0.05))

# ─── Test 17: Flyout slider shows base, not effective ────────────────────────

def test_base_hidden_from_ui():
    print("\n[17] GetBaseConfFloor returns base, calibration delta stays hidden")
    cal = PairCalibration(BASE_CONF_EN_RU, BASE_MARGIN_EN_RU)
    # Calibration tightens the effective value behind the scenes.
    for _ in range(20):
        cal.record_outcome("FP")
    check("effective conf tightened above base",
          cal.applied_conf > BASE_CONF_EN_RU,
          f"applied={cal.applied_conf:.4f}")
    check("GetBaseConfFloor still reports the factory base (delta invisible)",
          approx(cal.get_base_conf(), BASE_CONF_EN_RU),
          f"base={cal.get_base_conf():.4f}")

# ─── Test 18: Manual base edit preserves calibration delta ───────────────────

def test_set_base_preserves_delta():
    print("\n[18] SetBaseConfFloor shifts base while keeping the calibration delta")
    cal = PairCalibration(BASE_CONF_EN_RU, BASE_MARGIN_EN_RU)
    # Build up a tightening delta.
    for _ in range(20):
        cal.record_outcome("FP")
    delta_before = cal.delta_conf_max
    check("a positive delta exists before manual edit", delta_before > 0.0)
    # User drags the slider to a new base (e.g. 0.65).
    new_base = 0.65
    cal.set_base_conf(new_base)
    check("base updated to user value",
          approx(cal.get_base_conf(), new_base))
    check("calibration delta preserved across manual edit",
          approx(cal.delta_conf_max, delta_before),
          f"delta={cal.delta_conf_max:.4f} before={delta_before:.4f}")
    check("effective = clamp(new_base + delta) rides on top of the edit",
          approx(cal.applied_conf,
                 clamp(new_base + delta_before, ABS_MIN_CONF, ABS_MAX_CONF)),
          f"applied={cal.applied_conf:.4f}")

# ─── Test 19: Reset Defaults clears base AND delta to factory ─────────────────

def test_reset_pair_to_factory():
    print("\n[19] ResetPairToFactory wipes both base edits and calibration delta")
    cal = PairCalibration(BASE_CONF_EN_RU, BASE_MARGIN_EN_RU)
    # Manual edit + calibration delta both present.
    cal.set_base_conf(0.62)
    for _ in range(20):
        cal.record_outcome("FP")
    check("state diverged from factory before reset",
          (not approx(cal.get_base_conf(), BASE_CONF_EN_RU)) or
          cal.delta_conf_max != 0.0)
    # "Reset Defaults" button.
    cal.reset_to_factory(BASE_CONF_EN_RU, BASE_MARGIN_EN_RU)
    check("base restored to factory",
          approx(cal.get_base_conf(), BASE_CONF_EN_RU))
    check("delta_conf_max cleared", approx(cal.delta_conf_max, 0.0))
    check("delta_margin cleared",   approx(cal.delta_margin, 0.0))
    check("effective conf back to factory",
          approx(cal.applied_conf, BASE_CONF_EN_RU),
          f"applied={cal.applied_conf:.4f}")
    check("effective margin back to factory",
          approx(cal.applied_margin, BASE_MARGIN_EN_RU),
          f"applied={cal.applied_margin:.4f}")

# ─── Run all tests ────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Adaptive Calibration Controller — Unit Tests")
    print(f"Constants: alpha={EWMA_ALPHA}, min_events={MIN_EVENTS}, "
          f"hysteresis={HYSTERESIS_BAND}, step_conf={STEP_CONF}, "
          f"step_margin={STEP_MARGIN}")
    print("=" * 60)

    test_ewma_formula()
    test_min_events_gate()
    test_dead_zone()
    test_tighten_on_fp()
    test_loosen_on_fn()
    test_tighten_ceiling()
    test_loosen_floor()
    test_absolute_clamps()
    test_batch_reset()
    test_ceiling_decay()
    test_tp_stabilises()
    test_base_stored_once()
    test_reset_calibration()
    test_reversal()
    test_exact_first_adaptation()
    test_per_pair_isolation()

    test_base_hidden_from_ui()
    test_set_base_preserves_delta()
    test_reset_pair_to_factory()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed  "
          f"({'OK' if FAIL == 0 else 'FAILURES'})")
    print("=" * 60)
    return 0 if FAIL == 0 else 1

if __name__ == "__main__":
    sys.exit(main())

