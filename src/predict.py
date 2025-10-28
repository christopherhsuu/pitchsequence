import pandas as pd
import joblib
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")
ASSETS_DIR = Path("pitcher_assets")

MODEL_PATH = ARTIFACTS_DIR / "runvalue_model.pkl"
ARSENALS_PATH = ASSETS_DIR / "pitcher_arsenals.csv"
CLUSTER_PATH = Path("data/cluster_averages.csv")
CLUSTER_DF = None
PROCESSED_DIR = Path("data/processed")

# caches for EPRV computation
_OUTCOME_RV = None  # dict: outcome -> {count_key: avg_delta_re}
_BASELINE = None    # nested dict baseline[count][pitch_type][zone][p_throws][stand]
OUTCOME_CLASSIFIER_PATH = ARTIFACTS_DIR / "outcome_classifier.pkl"
_OUTCOME_CLASSIFIER = None

def load_cluster_data():
    """Load cluster-level averages into a DataFrame (cached)."""
    global CLUSTER_DF
    if CLUSTER_DF is None and CLUSTER_PATH.exists():
        df = pd.read_csv(CLUSTER_PATH)
        # ensure cluster column present and usable
        CLUSTER_DF = df
    return CLUSTER_DF

def get_cluster_features(cluster_id):
    """Return a dict of cluster features for a given cluster id (or None)."""
    df = load_cluster_data()
    if df is None:
        return None
    try:
        row = df[df['cluster'] == float(cluster_id)].iloc[0]
        return row.to_dict()
    except Exception:
        return None

def adjust_pitch_recommendation(probs: dict, cluster_features: dict):
    """Adjust pitch probability dict based on simple cluster heuristics.

    probs: mapping pitch_type -> probability (sums to 1)
    cluster_features: row from cluster_averages.csv
    Returns a new normalized probability dict.
    """
    if not cluster_features or not isinstance(probs, dict):
        return probs

    # copy
    adj = {p: float(v) for p, v in probs.items()}

    # Heuristic thresholds (tunable):
    bat_speed = float(cluster_features.get('bat_speed_mean', 0) or 0)
    attack_angle = float(cluster_features.get('attack_angle_mean', 0) or 0)
    contact_rate = float(cluster_features.get('contact_rate', 0) or 0)

    # Identify fastball-like and offspeed pitch codes heuristically
    fastball_keys = [k for k in adj.keys() if k.upper().startswith('F') or k.upper() in ('FF','FT','FA','FC','SI')]
    offspeed_keys = [k for k in adj.keys() if k.upper() in ('SL','CU','CH','KC','CT','CH')]

    # Power/pull hitters: high bat speed and flat attack angle -> reduce fastballs a bit
    if bat_speed > 72 and attack_angle < 10:
        for k in fastball_keys:
            adj[k] = adj.get(k, 0.0) * 0.92
        for k in offspeed_keys:
            adj[k] = adj.get(k, 0.0) * 1.08

    # Contact-oriented: high contact rate and low attack angle -> favor fastballs
    if contact_rate > 0.85 and attack_angle < 10:
        for k in offspeed_keys:
            adj[k] = adj.get(k, 0.0) * 0.88
        for k in fastball_keys:
            adj[k] = adj.get(k, 0.0) * 1.06

    # normalize
    s = sum(adj.values()) if adj else 1.0
    if s <= 0:
        return probs
    for k in adj:
        adj[k] = adj[k] / s
    return adj


# --- EPRV / baseline utilities (lazy) ---
def _normalize_count_key(balls, strikes):
    return f"{int(balls)}-{int(strikes)}"

def _load_processed_events(path: Path = PROCESSED_DIR / "features.parquet"):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        try:
            return pd.read_parquet(str(p))
        except Exception:
            return None

def compute_outcome_rv_table(sample_rows: int = None):
    """Compute average ΔRE (run_value) per outcome and count (balls-strikes).

    Returns dict outcome -> {count_key: avg_run_value}
    """
    global _OUTCOME_RV
    if _OUTCOME_RV is not None:
        return _OUTCOME_RV
    df = _load_processed_events()
    if df is None:
        _OUTCOME_RV = {}
        return _OUTCOME_RV

    if sample_rows is not None:
        df = df.head(sample_rows)

    # ensure run_value exists (if not, try computing via run_value utility)
    if "run_value" not in df.columns:
        try:
            from src.run_value import run_value
            df = run_value(df)
        except Exception:
            df = df.copy()
            df["run_value"] = 0.0

    # normalize outcome categories from events/description
    def _normalize_outcome(r):
        ev = r.get("events") if "events" in r else None
        desc = r.get("description") if "description" in r else None
        if isinstance(ev, str):
            ev = ev.lower()
        if ev in ("home_run", "home_run") or (isinstance(desc, str) and "home_run" in desc.lower()):
            return "HR"
        if ev == "triple":
            return "3B"
        if ev == "double":
            return "2B"
        if ev == "single":
            return "1B"
        if ev == "field_out" or ev == "grounded_into_double_play" or ev == "force_out":
            return "inplay_out"
        if ev == "strikeout":
            return "swinging_strike"
        if ev == "walk" or ev == "intent_walk":
            return "ball"
        if ev == "hit_by_pitch":
            return "HBP"
        # fouls and called strikes
        if isinstance(desc, str) and "foul" in desc.lower():
            return "foul"
        if isinstance(desc, str) and "called strike" in desc.lower():
            return "called_strike"
        # fallback: treat as inplay_out
        return "inplay_out"

    df = df.assign(_outcome=df.apply(_normalize_outcome, axis=1))
    df = df.assign(_count=df.apply(lambda r: _normalize_count_key(r.get("balls", 0), r.get("strikes", 0)), axis=1))

    groups = df.groupby(["_outcome", "_count"])['run_value'].mean()
    table = {}
    for (outcome, count_key), val in groups.items():
        table.setdefault(outcome, {})[count_key] = float(val)

    _OUTCOME_RV = table
    return _OUTCOME_RV

def build_baseline_table(min_counts: int = 5, sample_rows: int = None):
    """Build baseline[count][pitch_type][zone][p_throws][stand] = avg ΔRE with backoffs.

    This computes aggregated averages and stores them in a nested dict for lookup.
    """
    global _BASELINE
    if _BASELINE is not None:
        return _BASELINE
    df = _load_processed_events()
    if df is None:
        _BASELINE = {}
        return _BASELINE
    if sample_rows is not None:
        df = df.head(sample_rows)

    if "run_value" not in df.columns:
        try:
            from src.run_value import run_value
            df = run_value(df)
        except Exception:
            df = df.copy()
            df["run_value"] = 0.0

    df = df.assign(count_key=df.apply(lambda r: _normalize_count_key(r.get('balls',0), r.get('strikes',0)), axis=1))
    # We'll compute multiple aggregation levels for backoff
    # full key
    agg = df.groupby(['count_key','pitch_type','zone','p_throws','stand'])['run_value'].agg(['mean','count']).reset_index()
    baseline = {}
    for r in agg.itertuples():
        if r.count < min_counts:
            continue
        baseline.setdefault(r.count_key, {}).setdefault(r.pitch_type, {}).setdefault(str(r.zone), {}).setdefault(r.p_throws, {})[r.stand] = float(r.mean)

    # Also compute coarser levels for backoff (drop stand)
    agg2 = df.groupby(['count_key','pitch_type','zone','p_throws'])['run_value'].agg(['mean','count']).reset_index()
    for r in agg2.itertuples():
        if r.count < min_counts:
            continue
        baseline.setdefault(r.count_key, {}).setdefault(r.pitch_type, {}).setdefault(str(r.zone), {}).setdefault(r.p_throws, {})['*'] = float(r.mean)

    # even coarser (drop p_throws)
    agg3 = df.groupby(['count_key','pitch_type','zone'])['run_value'].agg(['mean','count']).reset_index()
    for r in agg3.itertuples():
        if r.count < min_counts:
            continue
        baseline.setdefault(r.count_key, {}).setdefault(r.pitch_type, {}).setdefault(str(r.zone), {})['*'] = float(r.mean)

    # pitch_type-zone global
    agg4 = df.groupby(['count_key','pitch_type'])['run_value'].agg(['mean','count']).reset_index()
    for r in agg4.itertuples():
        if r.count < min_counts:
            continue
        baseline.setdefault(r.count_key, {}).setdefault(r.pitch_type, {})['*'] = float(r.mean)

    # count-level global
    agg5 = df.groupby(['count_key'])['run_value'].agg(['mean','count']).reset_index()
    for r in agg5.itertuples():
        baseline.setdefault(r.count_key, {})['*'] = float(r.mean)

    _BASELINE = baseline
    return _BASELINE

def _baseline_lookup(count_key, pitch_type, zone, p_throws, stand):
    """Lookup baseline with hierarchical backoffs. Returns float or None.
    Backoff order: full -> drop stand -> drop p_throws -> drop zone -> drop pitch_type -> count global
    """
    b = build_baseline_table()
    if not b:
        return None
    ck = count_key
    pt = pitch_type
    zn = str(zone)
    try:
        # full
        val = b[ck][pt][zn][p_throws][stand]
        return val
    except Exception:
        pass
    try:
        val = b[ck][pt][zn][p_throws]['*']
        return val
    except Exception:
        pass
    try:
        val = b[ck][pt][zn]['*']
        return val
    except Exception:
        pass
    try:
        val = b[ck][pt]['*']
        return val
    except Exception:
        pass
    try:
        val = b[ck]['*']
        return val
    except Exception:
        return None

def _load_outcome_classifier():
    global _OUTCOME_CLASSIFIER
    if _OUTCOME_CLASSIFIER is not None:
        return _OUTCOME_CLASSIFIER
    if OUTCOME_CLASSIFIER_PATH.exists():
        try:
            _OUTCOME_CLASSIFIER = joblib.load(OUTCOME_CLASSIFIER_PATH)
            return _OUTCOME_CLASSIFIER
        except Exception:
            _OUTCOME_CLASSIFIER = None
            return None
    return None

def empirical_outcome_dist_for_candidate(count_key, pitch_type, zone, p_throws, stand, laplace=1.0):
    """Compute empirical multinomial P(o | context, candidate) with Laplace smoothing and backoffs."""
    df = _load_processed_events()
    if df is None:
        # uniform fallback over outcomes
        outcomes = ["ball","called_strike","swinging_strike","foul","inplay_out","1B","2B","3B","HR","HBP"]
        unif = {o: 1.0/len(outcomes) for o in outcomes}
        return unif

    # normalize outcome same as compute_outcome_rv_table
    def _norm_outcome_row(r):
        ev = r.get('events') if 'events' in r else None
        desc = r.get('description') if 'description' in r else None
        if isinstance(ev, str): ev = ev.lower()
        if ev == 'home_run': return 'HR'
        if ev == 'triple': return '3B'
        if ev == 'double': return '2B'
        if ev == 'single': return '1B'
        if ev in ('field_out','grounded_into_double_play','force_out'): return 'inplay_out'
        if ev == 'strikeout': return 'swinging_strike'
        if ev in ('walk','intent_walk'): return 'ball'
        if ev == 'hit_by_pitch': return 'HBP'
        if isinstance(desc, str) and 'foul' in desc.lower(): return 'foul'
        if isinstance(desc, str) and 'called strike' in desc.lower(): return 'called_strike'
        return 'inplay_out'

    df = df.assign(_outcome=df.apply(_norm_outcome_row, axis=1))
    df = df.assign(_count=df.apply(lambda r: _normalize_count_key(r.get('balls',0), r.get('strikes',0)), axis=1))

    # try exact match
    filt = (df['_count'] == count_key) & (df['pitch_type'].astype(str) == str(pitch_type)) & (df['zone'].astype(str) == str(zone)) & (df.get('p_throws', '') == p_throws) & (df.get('stand', '') == stand)
    sub = df[filt]
    # backoff sequence
    if sub.empty:
        # drop stand
        filt = (df['_count'] == count_key) & (df['pitch_type'].astype(str) == str(pitch_type)) & (df['zone'].astype(str) == str(zone)) & (df.get('p_throws', '') == p_throws)
        sub = df[filt]
    if sub.empty:
        # drop p_throws
        filt = (df['_count'] == count_key) & (df['pitch_type'].astype(str) == str(pitch_type)) & (df['zone'].astype(str) == str(zone))
        sub = df[filt]
    if sub.empty:
        # drop zone
        filt = (df['_count'] == count_key) & (df['pitch_type'].astype(str) == str(pitch_type))
        sub = df[filt]
    if sub.empty:
        # fall back to count-level distribution
        sub = df[df['_count'] == count_key]
    if sub.empty:
        # global fallback
        sub = df

    counts = sub['_outcome'].value_counts().to_dict()
    outcomes = ["ball","called_strike","swinging_strike","foul","inplay_out","1B","2B","3B","HR","HBP"]
    # apply Laplace smoothing
    smoothed = {o: (counts.get(o,0) + laplace) for o in outcomes}
    total = sum(smoothed.values())
    probs = {o: smoothed[o]/total for o in outcomes}
    return probs

def predict_outcome_distribution(context_row: dict):
    """Predict P(o | context_row) using classifier if present, otherwise empirical estimator.

    context_row should include keys: balls, strikes, pitch_type, zone, p_throws, stand, batter_cluster (optional)
    Returns dict outcome->prob
    """
    clf = _load_outcome_classifier()
    if clf is not None:
        # construct single-row DataFrame with expected feature columns
        try:
            X = pd.DataFrame([context_row])
            proba = clf.predict_proba(X)
            # sklearn multiclass: predict_proba returns list of arrays per class if multilabel; assume single estimator with classes_
            classes = getattr(clf, 'classes_', None)
            if classes is None:
                # fallback to uniform
                return empirical_outcome_dist_for_candidate(_normalize_count_key(context_row.get('balls',0), context_row.get('strikes',0)), context_row.get('pitch_type'), context_row.get('zone'), context_row.get('p_throws'), context_row.get('stand'))
            probs = {str(c): 0.0 for c in classes}
            # sklearn returns array shape (n_samples, n_classes)
            arr = proba if not isinstance(proba, list) else proba[0]
            for cls, p in zip(classes, arr[0] if arr.ndim==2 else arr):
                probs[str(cls)] = float(p)
            # map class labels to our outcome names if needed
            # assume classifier uses same naming as our outcome list
            return probs
        except Exception:
            pass

    # classifier not available or failed → empirical
    count_key = _normalize_count_key(context_row.get('balls',0), context_row.get('strikes',0))
    return empirical_outcome_dist_for_candidate(count_key, context_row.get('pitch_type'), context_row.get('zone'), context_row.get('p_throws'), context_row.get('stand'))

def compute_eprv_and_rvaa_for_candidates(state: dict, candidates: list):
    """Given a state dict and candidate list of (pitch_type, zone), compute raw_EPRV and RVAA per candidate.

    Returns list of dicts with keys: pitch, zone, probs, raw_eprv, baseline, rvaa
    """
    outcome_table = compute_outcome_rv_table()
    results = []
    for cand in candidates:
        pitch = cand.get('pitch')
        zone = cand.get('zone')
        # build context row
        ctx = dict(state)
        ctx['pitch_type'] = pitch
        ctx['zone'] = zone
        count_key = _normalize_count_key(state.get('balls',0), state.get('strikes',0))
        ctx['count_key'] = count_key
        probs = predict_outcome_distribution(ctx)
        # compute raw EPRV
        raw = 0.0
        for o, p in probs.items():
            rv_map = outcome_table.get(o, {})
            rv = rv_map.get(count_key, 0.0)
            raw += p * rv
        # baseline lookup
        baseline_val = _baseline_lookup(count_key, pitch, zone, state.get('p_throws'), state.get('stand'))
        if baseline_val is None:
            baseline_val = 0.0
        rvaa = raw - baseline_val
        results.append({
            'pitch': pitch,
            'zone': zone,
            'probs': probs,
            'raw_eprv': float(raw),
            'baseline': float(baseline_val),
            'rvaa': float(rvaa),
            'count_key': count_key
        })
    return results

def render_eprv_report(chosen: dict, top_probs: dict, batter_cluster_label: str = None):
    """Return a short report string per the requested format."""
    pitch = chosen.get('pitch')
    zone = chosen.get('zone')
    rvaa = chosen.get('rvaa')
    raw = chosen.get('raw_eprv')
    baseline = chosen.get('baseline')
    # describe top probs
    if isinstance(top_probs, dict):
        top = sorted(top_probs.items(), key=lambda x: -x[1])[:3]
        desc = ', '.join([f"{k}:{v:.2f}" for k, v in top])
    else:
        desc = ''
    arche = f" Archetype shifts: {batter_cluster_label}" if batter_cluster_label else ''
    return f"Chosen: {pitch} {zone} (RVAA {rvaa:+.3f}; raw ΔRE {raw:+.3f}). Baseline for {pitch} {zone} at {chosen.get('count_key')} is {baseline:+.3f}. Top outcomes: {desc}.{arche}"

def state_preview(count_key=None):
    return count_key or "?"

# test helpers
def _set_baseline_table(t):
    global _BASELINE
    _BASELINE = t

def _set_outcome_rv_table(t):
    global _OUTCOME_RV
    _OUTCOME_RV = t

def load_model():
    return joblib.load(MODEL_PATH)

def load_arsenals():
    return pd.read_csv(ARSENALS_PATH)

def recommend_next_pitch(state: dict, pitcher_id: int, batter_stats_map: dict = None):
    # Prefer clustered model if present
    clustered_path = ARTIFACTS_DIR / "pitchsequence_clustered_model.pkl"
    model = None
    if clustered_path.exists():
        try:
            model = joblib.load(clustered_path)
        except Exception:
            model = load_model()
    else:
        model = load_model()
    ars = load_arsenals()
    pitches = ars[ars["pitcher"] == pitcher_id]["pitch_type"].unique().tolist()
    if not pitches:
        # fallback default arsenal
        pitches = ["FF", "SL", "CU"]

    # All numeric and categorical columns used in training
    all_cols = [
        "balls","strikes","outs_when_up",
        "on_1b","on_2b","on_3b",
        "release_speed","release_spin_rate",
        "zone","plate_x","plate_z","last_pitch_speed_delta",
        "batter_pitchtype_woba","batter_pitchtype_whiff_rate","batter_pitchtype_run_value",
        "stand","p_throws","pitch_type","last_pitch_type","last_pitch_result"
    ]

    rows = []
    for pitch in pitches:
        row = state.copy()
        row["pitch_type"] = pitch
        # provide realistic default release speed and spin by pitch type to let model differentiate
        pt = str(pitch).upper()
        default_speed = 92.0
        default_spin = 2100.0
        if pt in ("FF", "FT", "FA", "FC"):
            default_speed = 94.5
            default_spin = 2300.0
        elif pt in ("SL", "CU", "KC", "CH"):
            default_speed = 84.0
            default_spin = 2600.0
        elif pt in ("SI", "FS"):
            default_speed = 92.0
            default_spin = 2200.0
        elif pt in ("ST", "SV", "EP"):
            default_speed = 88.0
            default_spin = 2000.0

        row.setdefault("release_speed", default_speed)
        row.setdefault("release_spin_rate", default_spin)
        # if batter_stats_map provided, set the batter-specific features for this pitch
        if batter_stats_map and pitch in batter_stats_map:
            woba, whiff, brv = batter_stats_map.get(pitch, (0.0, 0.0, 0.0))
            row["batter_pitchtype_woba"] = woba
            row["batter_pitchtype_whiff_rate"] = whiff
            row["batter_pitchtype_run_value"] = brv
        else:
            for k in ["batter_pitchtype_woba","batter_pitchtype_whiff_rate","batter_pitchtype_run_value"]:
                row.setdefault(k, 0.0)
        rows.append(row)
    if not rows:
        raise ValueError("No candidate pitches to score for pitcher_id={}".format(pitcher_id))

    X = pd.DataFrame(rows, columns=all_cols)
    if X.shape[0] == 0:
        raise ValueError("Model input X is empty — cannot predict")
    preds = model.predict(X)
    best_idx = preds.argmin()
    preds_dict = dict(zip(pitches, preds))

    # Attempt to adjust probabilities using cluster heuristics if batter cluster info is available
    # Expect state may contain 'batter_cluster' or 'batter_id' mapping elsewhere
    cluster_id = state.get('batter_cluster') or state.get('batter_cluster_id')
    cluster_feats = None
    if cluster_id is not None:
        cluster_feats = get_cluster_features(cluster_id)

    # convert preds (lower is better run_value) into probabilities via softmax on negative values
    import math
    vals = list(preds_dict.values())
    neg = [-v for v in vals]
    max_neg = max(neg) if neg else 0.0
    exps = [math.exp(x - max_neg) for x in neg]
    s = sum(exps) if exps else 1.0
    probs = {p: e / s for p, e in zip(pitches, exps)}

    adjusted_probs = adjust_pitch_recommendation(probs, cluster_feats)

    # Build simple candidate list including intended zone from state (if present)
    intended_zone = state.get('zone') if isinstance(state.get('zone'), (int, str)) else state.get('zone')
    candidates = [{'pitch': p, 'zone': intended_zone} for p in pitches]

    # Compute EPRV and RVAA for each candidate
    try:
        eprv_results = compute_eprv_and_rvaa_for_candidates(state, candidates)
    except Exception:
        eprv_results = []

    # choose candidate with minimal RVAA (pitcher POV: lower is better); tie-breaker min raw_EPRV
    chosen = None
    if eprv_results:
        eprv_results_sorted = sorted(eprv_results, key=lambda r: (r['rvaa'], r['raw_eprv']))
        chosen = eprv_results_sorted[0]
    else:
        chosen = {'pitch': pitches[best_idx], 'zone': intended_zone, 'probs': probs, 'raw_eprv': None, 'baseline': None, 'rvaa': None, 'count_key': _normalize_count_key(state.get('balls',0), state.get('strikes',0))}

    # build a short report
    batter_cluster_label = None
    if cluster_feats is not None:
        # try to extract a friendly label
        batter_cluster_label = cluster_feats.get('label') or cluster_feats.get('cluster')

    report = render_eprv_report(chosen, chosen.get('probs', {}), batter_cluster_label)

    # return extended info: best pitch (by raw preds), its value, raw preds dict, adjusted probs, eprv_results, chosen, report
    return pitches[best_idx], float(preds[best_idx]), preds_dict, adjusted_probs, eprv_results, chosen, report

