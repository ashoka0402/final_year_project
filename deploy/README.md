# Deploying VAYU to Cloud Run

One container, one URL: FastAPI on an internal port, Next.js on Cloud Run's
`$PORT` proxying `/api/v1` to it. That means no CORS, no cross-service hop on
every map interaction, and one link to hand a judge.

## Prerequisites

```bash
gcloud auth login                 # interactive — must be run by a human
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

Cloud Run requires **billing enabled** on the project. The free tier is generous
(2M requests/month) but the project still needs a billing account attached.

## 1. Build the slim archive

The working DB is ~566MB, most of it hourly history that exists only to train
the forecast model. The model is already trained and committed, so the container
needs none of it.

```bash
python -m scripts.build_deploy_db     # -> data/vayu_deploy.duckdb (~126MB)
```

This keeps the satellite, fire, hotspot and citizen layers **whole** — they are
what the national views read — and trims the deep training history.

## 2. Deploy

```bash
gcloud run deploy vayu \
  --source . \
  --dockerfile deploy/Dockerfile \
  --region asia-south1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --set-env-vars "GOOGLE_API_KEY=YOUR_KEY,GEMINI_MODEL=gemini-3.6-flash,DEMO_MODE=true,VAYU_DB_PATH=/app/data/vayu.duckdb"
```

**Memory**: 2Gi is not optional. pandas + LightGBM + DuckDB + a Node process in
one container will OOM at 512Mi — that failure mode has already been hit once on
another host, and it presents as a silent restart loop rather than a clear error.

**Secrets**: `--set-env-vars` puts the Gemini key in the service config, which is
fine for a hackathon demo. For anything longer-lived use Secret Manager:

```bash
echo -n "YOUR_KEY" | gcloud secrets create vayu-gemini --data-file=-
gcloud run deploy vayu ... --set-secrets "GOOGLE_API_KEY=vayu-gemini:latest"
```

## 3. Verify

```bash
URL=$(gcloud run services describe vayu --region asia-south1 --format='value(status.url)')
curl -s "$URL/api/v1/health"
curl -s "$URL/api/v1/corridors" | head -c 300
open "$URL/report"
```

## Notes

- **Cold starts.** Loading a 126MB DuckDB plus the LightGBM artifacts takes a few
  seconds. Set `--min-instances 1` before a judged demo so the first click is not
  the slow one; it costs a little but removes the worst first impression.
- **Region.** `asia-south1` (Mumbai) keeps latency low for Indian users and puts
  the data in-country, which matters for a government-facing pilot.
- **The archive is read-only** in the deployed container. Citizen reports are
  written to the container's ephemeral disk and are lost on restart — acceptable
  for a demo, but a real pilot needs Cloud SQL or Firestore for that table.
