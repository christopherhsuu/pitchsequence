"""Test the location optimizer."""
from src.predict import PitchRecommender
import json

print('Testing PitchRecommender with location optimization...\n')

recommender = PitchRecommender()

# Test recommendation
results = recommender.recommend(
    pitcher_id=543037,  # Example pitcher
    batter_id=545361,   # Example batter
    balls=1,
    strikes=2,
    outs=1,
    on_1b=True,
    on_2b=False,
    on_3b=False,
    stand='R',
    top_n=3
)

print('Top 3 Recommendations:\n')
for i, r in enumerate(results):
    print(f'{i+1}. {r["pitch_type"]} ({r["category"]})')
    print(f'   Probability: {r["probability_pct"]:.1f}%')
    print(f'   Predicted RV: {r["predicted_rv"]:.4f}')
    print(f'   Location: plate_x={r["location"]["plate_x"]:.2f}, plate_z={r["location"]["plate_z"]:.2f}')
    print(f'   Zone: {r["location"]["zone"]} (in_zone={r["location"]["in_zone"]})')
    print(f'   Location-optimized RV: {r["location"]["expected_rv"]:.4f}')
    print(f'   Reasoning: {r["reasoning"]}\n')

print('Location optimization test completed successfully!')
