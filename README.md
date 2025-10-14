## Deploying to Vercel (serverless API)

This repository includes a FastAPI serverless endpoint at `api/recommend.py` that wraps the recommender logic.

Quick steps to deploy:

1. Install the Vercel CLI and log in.
2. Push this repository to GitHub (or connect your repo in the Vercel dashboard).
3. Vercel will detect the Python serverless functions under `api/` and install packages from `requirements.txt`.

Endpoint:

POST /.vercel.app/api/recommend (or `https://<your-deployment>.vercel.app/api/recommend`)

Example JSON body:

{
	"batter": "455117",
	"pitcher": "123456",
	"count": "1-1",
	"situation": {"risp": true, "outs": 2, "late_inning": true}
}

The endpoint returns a JSON recommendation (recommended_sequence, confidence, strategy_notes, ...).

Notes:
- Vercel serverless functions are short-lived. This API is stateless and loads data on startup.
- The Streamlit app in `app/app.py` is not suitable for Vercel serverless; consider using the `/api/recommend` endpoint from a lightweight static frontend or another deployment for Streamlit.
# pitchsequence!
