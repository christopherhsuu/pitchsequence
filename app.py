import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Next-Pitch Outcome Explorer (2025)", layout="wide")

# ── Load artifacts ──────────────────────────────────────────────────────────────
pipe = joblib.load("artifacts/next_pitch_outcome_model.pkl")
with open("artifacts/feature_columns.json", "r") as f:
    feature_info = json.load(f)

re24 = pd.read_csv("re24_2025.csv")
count_re = pd.read_csv("count_re_2025.csv")
arsenal = pd.read_parquet("artifacts/pitcher_arsenal_2025.parquet")
pitcher_index = pd.read_parquet("artifacts/pitcher_index_2025.parquet")  # names here (1 row/pitcher)
batters = pd.read_parquet("artifacts/batters_2025.parquet")              # IDs only (1 row/batter)

outcome_erv_path = "artifacts/outcome_erv_lookup_2025.csv"
if os.path.exists(outcome_erv_path):
    outcome_erv = pd.read_csv(outcome_erv_path)
    have_outcome_erv = True
else:
    outcome_erv = pd.DataFrame(columns=["base_state","outs","y_class","ERV"])
    have_outcome_erv = False

# ── Zone helpers ────────────────────────────────────────────────────────────────
# Strike zone bounds (smaller box)
ZX = [-0.95, -0.32, 0.32, 0.95]
ZZ = [1.5, 2.3, 3.1, 3.9]




def zone_bucket(x, z):
    if ZX[0] <= x < ZX[-1] and ZZ[0] <= z < ZZ[-1]:
        ix = np.searchsorted(ZX, x, side="right") - 1
        iz = np.searchsorted(ZZ, z, side="right") - 1
        return f"Z{iz}{ix}"
    if z >= ZZ[-1]: return "OOZ_UP"
    if z <  ZZ[0]:  return "OOZ_DOWN"
    if x <= ZX[0]:  return "OOZ_IN"
    if x >= ZX[-1]: return "OOZ_AWAY"
    return "OOZ_MISC"

def strike_zone_fig(x, z, ZX, ZZ):
    fig, ax = plt.subplots(figsize=(2.2, 3.0), constrained_layout=True)  # small figure
    ax.add_patch(plt.Rectangle((ZX[0], ZZ[0]), ZX[-1]-ZX[0], ZZ[-1]-ZZ[0],
                               fill=False, linewidth=2))
    for xv in ZX[1:-1]: ax.axvline(xv, linewidth=0.5, alpha=0.25)
    for zv in ZZ[1:-1]: ax.axhline(zv, linewidth=0.5, alpha=0.25)
    ax.scatter([x], [z], s=50)

    ax.set_xlim(-2.0, 2.0); ax.set_ylim(0.5, 5.0)
    ax.set_xlabel("plate_x (ft, − = in to RHH)", fontsize=9)
    ax.set_ylabel("plate_z (ft)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title("Strike Zone", pad=4, fontsize=10)
    ax.grid(False)
    return fig



# ── ERV helpers ────────────────────────────────────────────────────────────────
global_mean_ball   = float(count_re.loc[count_re["event_type"]=="Ball", "ERV"].mean()) if not count_re.empty else 0.0
global_mean_called = float(count_re.loc[count_re["event_type"]=="CalledStrike", "ERV"].mean()) if not count_re.empty else 0.0

def count_event_erv(base_state, outs, balls, strikes, event_type):
    row = count_re[(count_re.base_state==base_state)&(count_re.outs==outs)&
                   (count_re.balls==balls)&(count_re.strikes==strikes)&
                   (count_re.event_type==event_type)]
    if len(row): return float(row["ERV"].iloc[0])
    row = count_re[(count_re.base_state==base_state)&(count_re.outs==outs)&
                   (count_re.event_type==event_type)]
    if len(row): return float(row["ERV"].mean())
    return global_mean_ball if event_type=="Ball" else global_mean_called

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("Next-Pitch Outcome Explorer (2025)")

# Pitcher dropdown (names; 1 row/pitcher)
pitcher_index = pitcher_index.copy()
pitcher_index["display"] = pitcher_index["pitcher_name"].astype(str).str.strip()
pid_row = st.selectbox(
    "Pitcher",
    options=list(pitcher_index.itertuples(index=False)),
    format_func=lambda r: r.display if hasattr(r, "display") else str(r)
)
pid = int(pid_row.pitcher)
p_throws = pid_row.p_throws

# Pitch type (restricted to chosen pitcher)
p_ars = arsenal[arsenal["pitcher"] == pid]
pitch_type = st.selectbox("Pitch Type", options=sorted(p_ars["pitch_type"].unique().tolist()))

# Batter dropdown (IDs only, as requested)
bid_row = st.selectbox(
    "Batter (ID)",
    options=list(batters[["batter","stand"]].itertuples(index=False)),
    format_func=lambda r: str(r.batter)
)
bid = int(bid_row.batter)
stand = bid_row.stand

# Base–out & count
col1, col2, col3 = st.columns(3)
with col1:
    outs = st.selectbox("Outs", [0,1,2], index=0)
with col2:
    on1 = st.checkbox("Runner on 1st", value=False)
    on2 = st.checkbox("Runner on 2nd", value=False)
    on3 = st.checkbox("Runner on 3rd", value=False)
with col3:
    balls = st.selectbox("Balls", [0,1,2,3], index=0)
    strikes = st.selectbox("Strikes", [0,1,2], index=0)

base_state = f"{int(on1)}{int(on2)}{int(on3)}"

# Location + zone
left, right = st.columns([1, 1])
with left:
    x = st.slider("plate_x (ft, negative = in to RHH)", -2.0, 2.0, 0.0, 0.01)
    z = st.slider("plate_z (ft)", 0.5, 5.0, 2.8, 0.01)
    grid_cell = zone_bucket(x, z)
with right:
    fig = strike_zone_fig(x, z, ZX, ZZ)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")  # tight + controlled DPI
    plt.close(fig)
    st.image(buf.getvalue(), width=260)  # << control how big it appears on screen


# Autofill medians for the chosen pitcher's pitch_type
med = p_ars[p_ars["pitch_type"] == pitch_type].iloc[0]
release_speed = st.number_input("release_speed (mph)", value=float(med["release_speed"]))
pfx_x         = st.number_input("pfx_x (ft)",           value=float(med["pfx_x"]))
pfx_z         = st.number_input("pfx_z (ft)",           value=float(med["pfx_z"]))
release_pos_z = st.number_input("release_pos_z (ft)",   value=float(med["release_pos_z"]))
extension     = st.number_input("extension (ft)",       value=float(med["extension"]))

if not have_outcome_erv:
    st.warning("Outcome ERV lookup not found. Ball/CalledStrike ERV works; contact outcomes will show ERV=0. "
               "Save artifacts/outcome_erv_lookup_2025.csv for full ERV.")

# Predict
if st.button("Predict"):
    feat = pd.DataFrame([{
        "pitch_type": pitch_type,
        "grid_cell": grid_cell,
        "p_throws": p_throws,
        "stand": stand,
        "release_speed": release_speed,
        "pfx_x": pfx_x,
        "pfx_z": pfx_z,
        "release_pos_z": release_pos_z,
        "extension": extension,
        "balls": balls,
        "strikes": strikes,
        "outs_when_up": outs
    }])

    probs   = pipe.predict_proba(feat)[0]
    classes = pipe.classes_

    rows, ev = [], 0.0
    for cls, p in zip(classes, probs):
        if cls in ("Ball","CalledStrike"):
            ev_i = count_event_erv(base_state, outs, balls, strikes, "Ball" if cls=="Ball" else "CalledStrike")
        else:
            sel = outcome_erv[(outcome_erv.base_state==base_state)&
                              (outcome_erv.outs==outs)&
                              (outcome_erv.y_class==cls)]
            ev_i = float(sel["ERV"].iloc[0]) if len(sel) else 0.0
        contrib = float(p) * float(ev_i)
        rows.append({"Outcome": cls, "Prob": float(p), "ERV_if_occurs": float(ev_i), "Contribution": contrib})
        ev += contrib

    st.subheader(f"Expected Run Value of this pitch: **{ev:+.3f}**")
    st.dataframe(pd.DataFrame(rows).sort_values("Contribution", ascending=False).reset_index(drop=True))
    st.caption(f"Grid cell: {grid_cell} | Base state: {base_state}")
