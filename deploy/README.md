# Cloud Run deployment

Deploys the goalie-analytics pipeline as a **Cloud Run Service** (thin
API dispatcher) plus a **Cloud Run Job** (per-vID pipeline worker).

```
   caller ─POST /run─► Service ─dispatch─► Job (per vID)
                          │                  │
                          │                  ├─ stage 1 (cv_seg or fusion)
                          │                  ├─ stage 2 (metrics_seg)
                          │                  └─ stage 3 (feedback_seg)
                          │                            │
                          │                            └─ writes "Processing (X%)"
                          │                               to customer JSON in GCS
                          │
   caller ─GET /status/{cust}/{vID}─► Service ─reads─► customer JSON in GCS
                                                       returns analyticsStatus
```

## What's in this directory

| File | Purpose |
|---|---|
| `Dockerfile` | Single container image used by both Service and Job |
| `.dockerignore` | Keeps videos / training scratch out of the build context |
| `cloudbuild.yaml` | Cloud Build config — builds + pushes to Artifact Registry |
| `api/main.py` | FastAPI Service: `POST /run`, `GET /status/...`, `GET /health` |
| `worker/run.py` | Cloud Run Job entrypoint — env vars → `run_pipeline.py` |
| `deploy.sh` | One-shot deploy script (idempotent) |

## One-time setup + deploy

```bash
# Auth + select project
gcloud auth login
gcloud config set project goalie-analytics-pro-dev

# Build, push, deploy Service + Job (idempotent — re-run to update)
bash deploy/deploy.sh
```

The script:
1. Enables required APIs (Run, Artifact Registry, Storage, AI Platform).
2. Creates Artifact Registry repo `goalie-pipeline` if missing.
3. Creates two service accounts:
   - **Runtime SA** `goalie-pipeline-sa` — used by both Service and Job
     to do their work. Gets project-level `run.developer` +
     `aiplatform.user` + bucket-scoped `storage.objectAdmin` on
     `gs://goalie_video_bucket`.
   - **Caller SA** `goalie-api-caller-sa` — identity for clients
     invoking the API. Gets ONLY `roles/run.invoker` on the Service
     (step 7, after the Service exists). Hand this SA's email to any
     frontend, dashboard, or external service that needs to call
     `POST /run` / `GET /status`.
4. Builds the image via Cloud Build.
5. Deploys the Job (worker) with 4Gi RAM, 2 CPU, 3h task-timeout.
6. Deploys the Service (API) with 512Mi RAM, 1 CPU, IAM auth (no
   unauthenticated access).
7. Binds the caller SA as an invoker on the Service.

Override defaults via env vars:

```bash
PROJECT_ID=my-other-project REGION=us-east1 bash deploy/deploy.sh
IMAGE_TAG=abc123 bash deploy/deploy.sh   # pin to a specific Git SHA
```

## API usage

### Getting an auth token

Two flows:

**(a) As yourself** — uses your own `gcloud` identity. Works if your
principal has `roles/run.invoker` on the Service (project owners do
by default).

```bash
TOKEN=$(gcloud auth print-identity-token)
```

**(b) As the caller SA** — recommended for production clients
(frontends, external services, CI jobs). The caller SA was created
by `deploy.sh` and is the principal you should hand to anyone who
needs API access. To impersonate it, your gcloud principal needs
`roles/iam.serviceAccountTokenCreator` on the caller SA:

```bash
CALLER_SA=goalie-api-caller-sa@goalie-analytics-pro-dev.iam.gserviceaccount.com
SERVICE_URL=https://goalie-pipeline-api-301726916294.us-central1.run.app

# One-time: grant yourself permission to impersonate the caller SA
gcloud iam service-accounts add-iam-policy-binding "${CALLER_SA}" \
    --member="user:YOU@example.com" \
    --role="roles/iam.serviceAccountTokenCreator"

# Per call: mint a fresh short-lived ID token audience-scoped to the Service.
# The `| tr -d '[:space:]'` strips a trailing newline that gcloud appends —
# without it, the newline lands inside the Authorization header and libcurl
# returns "error 43: A libcurl function was given a bad argument" with HTTP 000.
TOKEN=$(gcloud auth print-identity-token \
    --impersonate-service-account="${CALLER_SA}" \
    --audiences="${SERVICE_URL}" | tr -d '[:space:]')
```

For workloads running INSIDE GCP (other Cloud Run services, Cloud
Functions, GKE), use the caller SA directly as the runtime SA and skip
impersonation — the metadata server mints tokens automatically.

For workloads OUTSIDE GCP that need long-lived credentials, you can
create a JSON key for the caller SA (`gcloud iam service-accounts keys
create`) and use it with the standard Google auth libraries, but
prefer Workload Identity Federation where possible.

### Dispatch a pipeline run

```bash
SERVICE_URL=$(gcloud run services describe goalie-pipeline-api \
    --region us-central1 --format='value(status.url)')

curl -X POST "${SERVICE_URL}/run" \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "Content-Type: application/json" \
    -d '{
        "customer_id": "CUST000031",
        "vID":         ["dwGsP6QKDs8"],
        "stage1_mode": "hybrid",
        "steps":       [1, 2, 3]
    }'
```

Response (immediate, doesn't wait for the run):

```json
{
  "customer_id": "CUST000031",
  "executions": [
    {
      "vID":             "dwGsP6QKDs8",
      "execution_name":  "projects/.../jobs/goalie-pipeline-worker/executions/...",
      "status_url":      "/status/CUST000031/dwGsP6QKDs8"
    }
  ]
}
```

### Poll for status

```bash
curl -H "Authorization: Bearer ${TOKEN}" \
     "${SERVICE_URL}/status/CUST000031/dwGsP6QKDs8"
```

Response:

```json
{
  "customer_id":     "CUST000031",
  "vID":             "dwGsP6QKDs8",
  "analyticsStatus": "Processing (47%)",
  "updatedDate":     null
}
```

The `analyticsStatus` field is updated in real-time as the pipeline
runs — `Ready for Analysis` → `Processing (33%)` (after stage 1) →
ticks through stages 2 and 3 → `Complete` (after stage 3 succeeds).
See `util/progress.py` for the writer + commit `d8a0584` for design notes.

### Calling from code (on-GCP clients)

For callers running inside GCP (another Cloud Run service, a Cloud
Function, GKE) that use `goalie-api-caller-sa` as their runtime SA,
mint an audience-scoped ID token from the metadata server and hit the
Service directly. No impersonation, no key files.

First, bind the caller SA as the calling service's runtime SA:

```bash
gcloud run services update YOUR_OTHER_SERVICE \
    --region us-central1 \
    --service-account=goalie-api-caller-sa@goalie-analytics-pro-dev.iam.gserviceaccount.com
```

Then in your code:

**Python** (`google-auth` + `requests`):

```python
import google.auth.transport.requests
import google.oauth2.id_token
import requests

SERVICE_URL = "https://goalie-pipeline-api-301726916294.us-central1.run.app"

auth_req = google.auth.transport.requests.Request()
token = google.oauth2.id_token.fetch_id_token(auth_req, SERVICE_URL)

resp = requests.post(
    f"{SERVICE_URL}/run",
    headers={"Authorization": f"Bearer {token}"},
    json={
        "customer_id": "CUST000031",
        "vID": ["dwGsP6QKDs8"],
        "stage1_mode": "hybrid",
        "steps": [1, 2, 3],
    },
    timeout=30,
)
resp.raise_for_status()
print(resp.json())
```

**Node.js** (`google-auth-library`):

```js
import {GoogleAuth} from 'google-auth-library';

const SERVICE_URL = 'https://goalie-pipeline-api-301726916294.us-central1.run.app';

const auth = new GoogleAuth();
const client = await auth.getIdTokenClient(SERVICE_URL);

const resp = await client.request({
  url: `${SERVICE_URL}/run`,
  method: 'POST',
  data: {
    customer_id: 'CUST000031',
    vID: ['dwGsP6QKDs8'],
    stage1_mode: 'hybrid',
    steps: [1, 2, 3],
  },
});
console.log(resp.data);
```

**Go** (`google.golang.org/api/idtoken`):

```go
package main

import (
    "bytes"
    "context"
    "fmt"
    "io"

    "google.golang.org/api/idtoken"
)

func main() {
    const serviceURL = "https://goalie-pipeline-api-301726916294.us-central1.run.app"

    ctx := context.Background()
    client, err := idtoken.NewClient(ctx, serviceURL)
    if err != nil {
        panic(err)
    }

    body := []byte(`{
        "customer_id": "CUST000031",
        "vID": ["dwGsP6QKDs8"],
        "stage1_mode": "hybrid",
        "steps": [1, 2, 3]
    }`)

    resp, err := client.Post(serviceURL+"/run", "application/json", bytes.NewReader(body))
    if err != nil {
        panic(err)
    }
    defer resp.Body.Close()
    out, _ := io.ReadAll(resp.Body)
    fmt.Println(string(out))
}
```

The audience passed to `fetch_id_token` / `getIdTokenClient` /
`idtoken.NewClient` MUST be the Service base URL (no path, no trailing
slash). If you pass `${SERVICE_URL}/run`, the token's `aud` claim won't
match what the Service validates and you'll get HTTP 401.

### Granting another principal access to call the API

Easiest path: let them impersonate the existing caller SA (no new IAM
binding on the Service needed).

```bash
gcloud iam service-accounts add-iam-policy-binding \
    goalie-api-caller-sa@goalie-analytics-pro-dev.iam.gserviceaccount.com \
    --member="user:alice@example.com" \
    --role="roles/iam.serviceAccountTokenCreator"
```

For direct invoker grants (e.g. a user who shouldn't go through
impersonation, or another service account):

```bash
gcloud run services add-iam-policy-binding goalie-pipeline-api \
    --region us-central1 \
    --member="user:alice@example.com" \
    --role="roles/run.invoker"
```

For service-to-service calls use `serviceAccount:caller@…iam.gserviceaccount.com`.

## Granting the worker access to its prerequisites

The Job's SA (`goalie-pipeline-sa`) already gets the bucket-level perms
from `deploy.sh`. If you store videos / customer configs in a DIFFERENT
bucket, also bind `storage.objectAdmin` on that bucket.

## Operational notes

- **Cold starts**: image is ~1–2 GB (Python + PyTorch + OpenCV + baked
  weights). Service cold start ~10–20s; Job tasks always cold-start.
  Acceptable for an async dispatch pattern.
- **Cost**: Service runs as long as there are requests in flight
  (typically <1s per request — cheap). Each Job execution = full
  pipeline cost (~$3 per vID, dominated by Gemini Pro calls in stage 2).
- **Timeouts**: Job task-timeout is 3h. Single pipeline runs take 15–30
  min, so the 3h buffer absorbs Gemini retries and GCS slow paths
  without forcing a restart.
- **Retries**: Job is deployed with `--max-retries=0`. If the worker
  exits non-zero, the execution is reported as failed and the
  customer JSON's analyticsStatus is set to `Failed: worker exit N`
  by `deploy/worker/run.py`.
- **Concurrency**: One Job execution per vID. Multiple vIDs in one
  `POST /run` dispatch in parallel — each gets its own execution.
- **Updating the image**: re-run `bash deploy/deploy.sh`. New Service
  revisions roll out automatically; the Job uses the new image on the
  next execution (in-flight executions keep the old image).

## Local dev / quick test

```bash
# Run the API locally (no GCS dispatch — just lets you hit /health
# and validate request shapes)
pip install fastapi uvicorn[standard] google-cloud-run
PORT=8080 uvicorn deploy.api.main:app --host 0.0.0.0 --port 8080
```

To exercise the dispatch path locally you need GCP creds:

```bash
gcloud auth application-default login
PORT=8080 uvicorn deploy.api.main:app --port 8080
# In another shell:
curl -X POST http://localhost:8080/run \
    -H "Content-Type: application/json" \
    -d '{"customer_id": "CUST000031", "vID": ["dwGsP6QKDs8"]}'
```
