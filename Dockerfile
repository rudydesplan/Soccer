# =============================================================================
# Stage 1: Build the React frontend
# =============================================================================
FROM node:20-slim AS frontend-builder

WORKDIR /build
COPY soccer-benchmark/frontend/package.json soccer-benchmark/frontend/package-lock.json ./
RUN npm ci

COPY soccer-benchmark/frontend/ ./
RUN npm run build
# Output: /build/dist/


# =============================================================================
# Stage 2: Runtime — Python backend + nginx
# =============================================================================
FROM python:3.13-slim AS runtime

# System dependencies:
#   nginx          — serves static files + proxies /api to uvicorn
#   libgomp1       — required by LightGBM (OpenMP)
#   curl           — used by the health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    libgomp1 \
    curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer-cached unless requirements change)
COPY soccer-benchmark/backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the inference engine and schemas (referenced via sys.path in the backend)
COPY salary_benchmark/ ./salary_benchmark/
COPY schemas.py ./schemas.py

# Copy the runtime data file
COPY player_pool.csv ./player_pool.csv

# Copy the trained model artifacts (4 AutoGluon variants + calibration JSONs)
COPY models/autogluon/           ./models/autogluon/
COPY models/autogluon_no_mv/     ./models/autogluon_no_mv/
COPY models/autogluon_no_mv_no_pos/ ./models/autogluon_no_mv_no_pos/
COPY models/autogluon_no_mv_no_age/ ./models/autogluon_no_mv_no_age/
COPY models/calibration.json             ./models/calibration.json
COPY models/calibration_no_mv.json       ./models/calibration_no_mv.json
COPY models/calibration_no_mv_no_pos.json ./models/calibration_no_mv_no_pos.json
COPY models/calibration_no_mv_no_age.json ./models/calibration_no_mv_no_age.json

# Copy the FastAPI backend
COPY soccer-benchmark/backend/ ./soccer-benchmark/backend/

# Copy the built React app from stage 1
COPY --from=frontend-builder /build/dist/ ./static/

# nginx config and entrypoint
COPY nginx.conf /etc/nginx/conf.d/default.conf
# Remove default nginx site so ours takes effect
RUN rm -f /etc/nginx/sites-enabled/default

COPY start.sh ./start.sh
RUN chmod +x ./start.sh

# Cloud Run injects PORT=8080; nginx listens on 8080
EXPOSE 8080

# PYTHONPATH so salary_benchmark and schemas are importable from any working dir
ENV PYTHONPATH=/app

CMD ["./start.sh"]
