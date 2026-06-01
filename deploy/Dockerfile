# Container image for the goalie-analytics pipeline. Same image serves
# both the FastAPI Cloud Run Service (entrypoint: deploy/api/main.py)
# and the Cloud Run Job worker (entrypoint: deploy/worker/run.py). The
# container is selected at deploy time via --command and --args.
#
# Build:
#   gcloud builds submit --config deploy/cloudbuild.yaml --project goalie-analytics-pro-dev
# Or locally:
#   docker build -f deploy/Dockerfile -t goalie-pipeline:dev .

FROM python:3.11-slim AS base

# Install system deps: ffmpeg (video/audio I/O) + git (some pip pkgs
# pull from GitHub) + libgl1 (opencv-python runtime needs libGL)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps first — leverages layer cache when only code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# FastAPI + uvicorn for the Cloud Run Service path; google-cloud-run
# for dispatching Jobs from the Service
RUN pip install --no-cache-dir \
        fastapi==0.115.* \
        uvicorn[standard]==0.32.* \
        google-cloud-run==0.10.*

# Copy the code + the baked-in model artifacts. The .dockerignore at
# the repo root excludes large items we don't want (data/videos/,
# data/labels/images/, etc); make sure deploy/.dockerignore (or the
# repo's main .dockerignore) is current before building.
COPY . .

# Sanity: confirm baked artifacts are present at build time so we
# fail fast on cache-miss issues rather than at first request.
RUN test -f runs/yolo_curve_n16/work/runs/hockeyai_shot_n16/weights/best.pt \
    || (echo "MISSING: YOLO weights"; exit 1)
RUN test -f runs/audio_curve_n16/work/best.pt \
    || (echo "MISSING: audio weights"; exit 1)
RUN test -d runs/yolo_curve_n16/probs \
    || (echo "MISSING: yolo probs/"; exit 1)
RUN test -d runs/audio_curve_n16/probs \
    || (echo "MISSING: audio probs/"; exit 1)

# Service-mode default. Cloud Run Jobs override CMD via --command/args
# in deploy/deploy.sh.
# Shell form so $PORT is expanded at runtime — Cloud Run injects PORT and
# the container contract requires listening on it. (Exec form cannot
# expand env vars, so a hardcoded 8080 breaks if Cloud Run changes PORT.)
ENV PORT=8080
EXPOSE 8080
CMD exec uvicorn deploy.api.main:app --host 0.0.0.0 --port ${PORT:-8080}
