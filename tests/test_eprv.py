import importlib
import copy

import importlib.util
from pathlib import Path

# load src/predict.py as module `predict`
spec = importlib.util.spec_from_file_location("predict", Path(__file__).resolve().parents[1] / "src" / "predict.py")
predict = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predict)


def test_baseline_backoff_levels():
    # prepare a manual baseline structure covering various backoff levels
    bl = {
        '1-2': {
            'SL': {
                '5': {
                    'R': {'R': -0.012},
                    '*': -0.011
                },
                '*': -0.02
            },
            '*': -0.05
        }
    }
    predict._set_baseline_table(bl)

    # full match (count -> pitch -> zone -> p_throws -> stand)
    v_full = predict._baseline_lookup('1-2', 'SL', 5, 'R', 'R')
    assert abs(v_full - (-0.012)) < 1e-9

    # missing stand -> fallback to '*' at p_throws level
    v_nostand = predict._baseline_lookup('1-2', 'SL', 5, 'R', 'L')
    assert abs(v_nostand - (-0.011)) < 1e-9

    # missing zone-specific entries -> fallback to '*' under pitch_type
    v_dropzone = predict._baseline_lookup('1-2', 'SL', 999, 'R', 'R')
    # expected to find SL -> '*' -> -0.02
    assert abs(v_dropzone - (-0.02)) < 1e-9

    # missing pitch_type -> fallback to count global
    v_droppitch = predict._baseline_lookup('1-2', 'FOO', 1, 'R', 'R')
    assert abs(v_droppitch - (-0.05)) < 1e-9


def test_eprv_rvaa_and_selection():
    # set a tiny outcome RV table and baseline so computations are deterministic
    # outcomes: ball, called_strike, swinging_strike, foul, inplay_out, 1B,2B,3B,HR,HBP
    rv_tbl = {
        'ball': {'0-0': -0.01},
        'called_strike': {'0-0': -0.02},
        'swinging_strike': {'0-0': -0.03},
        'foul': {'0-0': -0.005},
        'inplay_out': {'0-0': -0.02},
        '1B': {'0-0': 0.2},
        '2B': {'0-0': 0.4},
        '3B': {'0-0': 0.6},
        'HR': {'0-0': 1.0},
        'HBP': {'0-0': 0.0}
    }
    predict._set_outcome_rv_table(rv_tbl)

    # baseline: make FF low (worse for pitcher), SL better
    bl = {'0-0': {'FF': {'*': -0.01}, 'SL': {'*': -0.03}, '*': -0.02}}
    predict._set_baseline_table(bl)

    # monkeypatch predict_outcome_distribution to return different distributions for FF vs SL
    def fake_predict_outcome(ctx):
        if ctx.get('pitch_type') == 'FF':
            # heavier chance of contact -> worse raw EPRV
            return {'ball': 0.2, 'called_strike':0.1, 'swinging_strike':0.05, 'foul':0.05, 'inplay_out':0.4, '1B':0.1, '2B':0.05, '3B':0.02, 'HR':0.02, 'HBP':0.01}
        else:
            # SL: more chase -> more swinging strike
            return {'ball': 0.1, 'called_strike':0.05, 'swinging_strike':0.25, 'foul':0.1, 'inplay_out':0.3, '1B':0.05, '2B':0.05, '3B':0.02, 'HR':0.05, 'HBP':0.03}

    predict.predict_outcome_distribution = fake_predict_outcome

    state = {'balls': 0, 'strikes': 0, 'p_throws': 'R', 'stand': 'R'}
    candidates = [{'pitch':'FF','zone':5},{'pitch':'SL','zone':5}]

    results = predict.compute_eprv_and_rvaa_for_candidates(state, candidates)
    # find chosen by minimal rvaa
    sorted_res = sorted(results, key=lambda r: (r['rvaa'], r['raw_eprv']))
    chosen = sorted_res[0]

    # With the fake distributions and baselines the selection should be deterministic; assert expected pitch
    assert chosen['pitch'] == 'FF'
    # rvaa should be raw - baseline
    assert abs(chosen['rvaa'] - (chosen['raw_eprv'] - bl['0-0'][chosen['pitch']]['*'])) < 1e-9


def test_deterministic_repeated_runs():
    # ensure that repeated calls produce identical results
    predict._set_outcome_rv_table({'ball':{'0-0':-0.01}})
    predict._set_baseline_table({'0-0':{'FF':{'*':-0.01}, '*':-0.01}})

    # monkeypatch to simple deterministic distribution
    predict.predict_outcome_distribution = lambda ctx: {'ball':1.0,'called_strike':0.0,'swinging_strike':0.0,'foul':0.0,'inplay_out':0.0,'1B':0.0,'2B':0.0,'3B':0.0,'HR':0.0,'HBP':0.0}

    state = {'balls':0,'strikes':0,'p_throws':'R','stand':'R'}
    candidates = [{'pitch':'FF','zone':5}]
    a = predict.compute_eprv_and_rvaa_for_candidates(state, candidates)
    b = predict.compute_eprv_and_rvaa_for_candidates(state, candidates)
    assert a == b
