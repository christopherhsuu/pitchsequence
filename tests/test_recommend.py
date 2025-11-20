from src.predict import recommend_next_pitch, MODEL_PATH

pid = 434378

for balls,strikes in [(0,0),(2,1),(3,2)]:
    state = {
        "balls": balls,
        "strikes": strikes,
        "outs_when_up": 0,
        "on_1b": 0,
        "on_2b": 0,
        "on_3b": 0,
        "release_speed": 94.0,
        "release_spin_rate": 2200.0,
        "zone": 5,
        "plate_x": 0.0,
        "plate_z": 2.5,
        "last_pitch_speed_delta": 0.0,
        "batter_pitchtype_woba": 0.0,
        "batter_pitchtype_whiff_rate": 0.0,
        "batter_pitchtype_run_value": 0.0,
        "stand": "R",
        "p_throws": "R",
        "pitch_type": None,
        "last_pitch_type": None,
        "last_pitch_result": None
    }
    print('===', balls, strikes)
    try:
        best, val, preds = recommend_next_pitch(state, pid)
        print('best', best, val)
        for p,v in preds.items():
            print(p, v)
    except Exception as e:
        print('error', e)

print('MODEL_PATH exists:', MODEL_PATH.exists())
