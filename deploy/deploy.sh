#!/usr/bin/env bash
# Deploy the goalie-analytics pipeline to Cloud Run.
#
# Idempotent: re-running updates the existing Service + Job to the
# latest image. First run does first-time setup (Artifact Registry repo
# + service-account + IAM bindings).
#
# Requires:
#   - gcloud SDK authenticated as a principal with sufficient perms
#   - Project + region per the env vars below (or override on call)
#
# Usage:
#   bash deploy/deploy.sh                 # full deploy at :latest
#   IMAGE_TAG=abc123 bash deploy/deploy.sh   # pin to a specific SHA

set -euo pipefail

PROJECT_ID=${PROJECT_ID:-goalie-analytics-pro-dev}
REGION=${REGION:-us-central1}
REPO=${REPO:-goalie-pipeline}
IMAGE=${IMAGE:-goalie-pipeline}
IMAGE_TAG=${IMAGE_TAG:-latest}

SERVICE_NAME=${SERVICE_NAME:-goalie-pipeline-api}
JOB_NAME=${JOB_NAME:-goalie-pipeline-worker}
# Two service accounts with separate concerns:
#
#   - SA_NAME (runtime):  used by both the API Service (to dispatch
#                          Jobs) and the Job (to call Gemini / read+
#                          write GCS). Project-level perms.
#   - CALLER_SA_NAME:      identity for clients that need to INVOKE
#                          the API. Has only roles/run.invoker on the
#                          Service. Hand this SA's email to anyone
#                          (frontend, dashboard, external system,
#                          another GCP service) that needs to POST /run
#                          or GET /status. They can either impersonate
#                          it via short-lived tokens (recommended) or
#                          mint a long-lived key for legacy systems.
SA_NAME=${SA_NAME:-goalie-pipeline-sa}
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
CALLER_SA_NAME=${CALLER_SA_NAME:-goalie-api-caller-sa}
CALLER_SA_EMAIL="${CALLER_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

GCS_BUCKET=${GCS_BUCKET:-goalie_video_bucket}

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:${IMAGE_TAG}"

echo "──────────────────────────────────────────────────────────────────"
echo "Deploying:"
echo "  project:  ${PROJECT_ID}"
echo "  region:   ${REGION}"
echo "  image:    ${IMAGE_URI}"
echo "  service:  ${SERVICE_NAME}"
echo "  job:      ${JOB_NAME}"
echo "  runtime sa: ${SA_EMAIL}"
echo "  caller sa:  ${CALLER_SA_EMAIL}"
echo "──────────────────────────────────────────────────────────────────"

# ── 1. Enable required APIs ──────────────────────────────────────────
echo "[1/6] Enabling APIs (run.googleapis.com, artifactregistry, storage, aiplatform)..."
gcloud services enable \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    storage.googleapis.com \
    aiplatform.googleapis.com \
    --project "${PROJECT_ID}" \
    --quiet

# ── 2. Artifact Registry repo (idempotent) ───────────────────────────
echo "[2/6] Ensuring Artifact Registry repo exists..."
if ! gcloud artifacts repositories describe "${REPO}" \
        --location "${REGION}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
    gcloud artifacts repositories create "${REPO}" \
        --repository-format=docker \
        --location="${REGION}" \
        --description="goalie-analytics pipeline images" \
        --project "${PROJECT_ID}"
fi

# ── 3. Service accounts + IAM ─────────────────────────────────────────
echo "[3/7] Ensuring service accounts + IAM bindings..."

# Runtime SA — used by both Service and Job
if ! gcloud iam service-accounts describe "${SA_EMAIL}" \
        --project "${PROJECT_ID}" >/dev/null 2>&1; then
    gcloud iam service-accounts create "${SA_NAME}" \
        --display-name="goalie pipeline runtime SA" \
        --project "${PROJECT_ID}"
fi

# Caller SA — identity that clients use to invoke the API. The
# run.invoker grant on the Service is added in step 7 (after the
# Service exists). No project-level perms — minimum-privilege.
if ! gcloud iam service-accounts describe "${CALLER_SA_EMAIL}" \
        --project "${PROJECT_ID}" >/dev/null 2>&1; then
    gcloud iam service-accounts create "${CALLER_SA_NAME}" \
        --display-name="goalie pipeline API caller SA" \
        --description="Identity for clients invoking the pipeline API. \
Grants only roles/run.invoker on the goalie-pipeline-api Service." \
        --project "${PROJECT_ID}"
fi

# Project-level grants for the runtime SA (smallest viable roles):
#   - run.developer:    dispatch Cloud Run Jobs from the Service
#   - aiplatform.user:  call Gemini via Vertex AI
#   - storage.objectAdmin (bucket-scoped): read videos + write
#                        customer JSON + write per-stage output blobs
for role in roles/run.developer roles/aiplatform.user; do
    gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="${role}" \
        --condition=None \
        --quiet >/dev/null
done
gcloud storage buckets add-iam-policy-binding "gs://${GCS_BUCKET}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/storage.objectAdmin" \
    --quiet >/dev/null

# ── 4. Build + push image (skips if IMAGE_TAG already exists) ────────
echo "[4/7] Building image via Cloud Build..."
gcloud builds submit \
    --config deploy/cloudbuild.yaml \
    --project "${PROJECT_ID}" \
    --substitutions=_PROJECT_ID="${PROJECT_ID}",_REGION="${REGION}",_REPO="${REPO}",_IMAGE="${IMAGE}",_TAG="${IMAGE_TAG}"

# ── 5. Deploy the Job (worker) ───────────────────────────────────────
echo "[5/7] Deploying Cloud Run Job..."
# Job timeout: 60min/task by default. Pipeline can take 25+ min per vid;
# bump to 3h to absorb worst-case GCS latency + Gemini retries.
gcloud run jobs deploy "${JOB_NAME}" \
    --image="${IMAGE_URI}" \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --service-account="${SA_EMAIL}" \
    --command="python3" \
    --args="deploy/worker/run.py" \
    --memory=4Gi \
    --cpu=2 \
    --task-timeout=10800s \
    --max-retries=0 \
    --set-env-vars="PROJECT_ID=${PROJECT_ID},REGION=${REGION},GCS_BUCKET=${GCS_BUCKET}" \
    --quiet

# ── 6. Deploy the Service (API) ──────────────────────────────────────
echo "[6/7] Deploying Cloud Run Service..."
gcloud run deploy "${SERVICE_NAME}" \
    --image="${IMAGE_URI}" \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --service-account="${SA_EMAIL}" \
    --memory=512Mi \
    --cpu=1 \
    --no-allow-unauthenticated \
    --set-env-vars="PROJECT_ID=${PROJECT_ID},REGION=${REGION},JOB_NAME=${JOB_NAME},GCS_BUCKET=${GCS_BUCKET}" \
    --quiet

# ── 7. Bind caller SA as the Service's invoker ───────────────────────
# Must happen AFTER step 6 (Service has to exist before we can grant
# roles/run.invoker on it). Idempotent — re-running is a no-op.
echo "[7/7] Granting caller SA invoker access to the Service..."
gcloud run services add-iam-policy-binding "${SERVICE_NAME}" \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --member="serviceAccount:${CALLER_SA_EMAIL}" \
    --role="roles/run.invoker" \
    --quiet >/dev/null

SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --region="${REGION}" --project="${PROJECT_ID}" --format='value(status.url)')

echo ""
echo "──────────────────────────────────────────────────────────────────"
echo "✅ Deployed."
echo "  Service URL: ${SERVICE_URL}"
echo "  Job name:    ${JOB_NAME}"
echo "  Caller SA:   ${CALLER_SA_EMAIL}"
echo ""
echo "Test the API as YOURSELF (your gcloud principal has roles/owner"
echo "so it can invoke directly):"
echo "  curl -H \"Authorization: Bearer \$(gcloud auth print-identity-token)\" \\"
echo "       \"${SERVICE_URL}/health\""
echo ""
echo "Test the API as the CALLER SA (recommended for production clients):"
echo "  TOKEN=\$(gcloud auth print-identity-token \\"
echo "      --impersonate-service-account=${CALLER_SA_EMAIL} \\"
echo "      --audiences=${SERVICE_URL})"
echo "  curl -H \"Authorization: Bearer \${TOKEN}\" \"${SERVICE_URL}/health\""
echo ""
echo "To impersonate the caller SA, your principal needs"
echo "roles/iam.serviceAccountTokenCreator on it:"
echo "  gcloud iam service-accounts add-iam-policy-binding ${CALLER_SA_EMAIL} \\"
echo "      --member=\"user:YOUR_EMAIL@example.com\" \\"
echo "      --role=\"roles/iam.serviceAccountTokenCreator\""
echo "──────────────────────────────────────────────────────────────────"
