# VAYU — single-container deploy for Cloud Run.
#
# One container runs both the FastAPI backend and the Next.js frontend, with
# Next proxying /api/v1 to the API on localhost. Cloud Run gives a service one
# port and one URL, so a single container means one URL for judges, no CORS, and
# no cross-service latency on every map interaction.
#
# The data lives in the image rather than in object storage. That is the right
# trade for a demo: the archive is read-only and immutable, so baking it in
# removes a network dependency and a credential from the critical path. Use
# scripts/build_deploy_db.py first — the full 540MB archive would make cold
# starts miserable.

# ---- stage 1: build the Next.js frontend ------------------------------------
FROM node:20-slim AS web
WORKDIR /web
COPY apps/web/package*.json ./
RUN npm ci
COPY apps/web/ ./
# Same-origin in production: the API is on localhost inside this container.
ENV NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
RUN npm run build

# ---- stage 2: python runtime + both processes -------------------------------
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 NODE_ENV=production

# libgomp: LightGBM. nodejs: to run the built Next server.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 curl ca-certificates gnupg \
 && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY vayu_core/ vayu_core/
COPY services/ services/
COPY config/ config/
COPY models/artifacts/ models/artifacts/
# Slim archive built by scripts/build_deploy_db.py
COPY data/vayu_deploy.duckdb data/vayu.duckdb

COPY --from=web /web/.next /app/web/.next
COPY --from=web /web/public /app/web/public
COPY --from=web /web/node_modules /app/web/node_modules
COPY --from=web /web/package.json /app/web/package.json
COPY --from=web /web/next.config.mjs /app/web/next.config.mjs

COPY deploy/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Cloud Run injects PORT; Next listens on it and proxies the API internally.
ENV PORT=8080 VAYU_DB_PATH=/app/data/vayu.duckdb
EXPOSE 8080
CMD ["/app/start.sh"]
