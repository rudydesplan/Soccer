# GCP Cloud Run deployment

Deploy the Soccer Salary Benchmark (FastAPI + React + AutoGluon models) to **Google Cloud Run** as a single Docker container behind nginx.

## Architecture

```
Internet → Cloud Run (port 8080)
              └── nginx
                    ├── /        → React static build
                    └── /api/*   → uvicorn (FastAPI, port 8001)
```

On startup the backend loads the player pool and all four AutoGluon model variants before `/api/health` returns OK (~15–20 s). Cloud Run waits for that startup probe before routing traffic.

**Resource profile:** 2 CPU, 4 Gi RAM, min 1 instance (always-on for demos).

## Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Multi-stage build (Node frontend + Python runtime + nginx) |
| `nginx.conf` | Reverse proxy and SPA fallback |
| `start.sh` | Starts uvicorn and nginx |
| `.dockerignore` | Keeps build context small |
| `cloudrun-service.yaml.template` | Parameterized Cloud Run service spec |
| `deploy-cloudrun.sh` | Full deploy pipeline (build → deploy → public IAM) |
| `deploy.cloudrun.env.example` | Example configuration — copy to `deploy.cloudrun.env` |

Generated at deploy time (gitignored):

- `cloudrun-service.yaml` — rendered from the template
- `deploy.cloudrun.env` — your local project settings

---

## Prerequisites

1. **Google Cloud project** with billing enabled.
2. **gcloud CLI** installed and authenticated:

   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

3. **APIs enabled:**

   ```bash
   gcloud services enable \
     run.googleapis.com \
     cloudbuild.googleapis.com \
     artifactregistry.googleapis.com \
     --project YOUR_PROJECT_ID
   ```

4. **Artifact Registry** Docker repository in your deploy region (skip if it already exists, e.g. `cloud-run-source-deploy` from Cloud Run source deploy):

   ```bash
   export GCP_REGION=europe-west1
   export GCP_PROJECT_ID=YOUR_PROJECT_ID

   gcloud artifacts repositories create cloud-run-source-deploy \
     --repository-format=docker \
     --location="$GCP_REGION" \
     --project="$GCP_PROJECT_ID" \
     --description="Cloud Run container images"
   ```

5. **IAM permissions** (typical minimum for deploy):

   - Cloud Build: submit builds
   - Cloud Run: `run.services.create`, `run.services.update`
   - Artifact Registry: push images

   To make the service **publicly accessible**, you also need `run.services.setIamPolicy` (included in `roles/run.admin` or `roles/owner`). Without it, deploy still works but anonymous users get **403 Forbidden**.

---

## Configuration

Copy the example env file and edit it:

```bash
cp deploy.cloudrun.env.example deploy.cloudrun.env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `GCP_PROJECT_ID` | `gcloud config` project | GCP project ID |
| `GCP_REGION` | `europe-west1` | Region for Cloud Build, Artifact Registry, and Cloud Run |
| `ARTIFACT_REGISTRY_REPO` | `cloud-run-source-deploy` | Artifact Registry repository name |
| `IMAGE_NAME` | `soccer-benchmark` | Docker image name |
| `IMAGE_TAG` | `latest` | Image tag |
| `SERVICE_NAME` | `soccer-salary-benchmark` | Cloud Run service name (URL slug) |
| `MAKE_PUBLIC` | `true` | Run the `allUsers` invoker IAM binding after deploy |

Example for Naboo:

```bash
GCP_PROJECT_ID=naboo-app-365515
GCP_REGION=europe-west1
ARTIFACT_REGISTRY_REPO=cloud-run-source-deploy
IMAGE_NAME=soccer-benchmark
IMAGE_TAG=latest
SERVICE_NAME=soccer-salary-benchmark
MAKE_PUBLIC=true
```

The full image URI is built as:

```
${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${ARTIFACT_REGISTRY_REPO}/${IMAGE_NAME}:${IMAGE_TAG}
```

---

## Quick deploy (recommended)

From the repository root:

```bash
chmod +x deploy-cloudrun.sh
./deploy-cloudrun.sh --env-file deploy.cloudrun.env
```

Or with inline variables:

```bash
GCP_PROJECT_ID=your-project-id GCP_REGION=europe-west1 ./deploy-cloudrun.sh
```

The script runs, in order:

1. Render `cloudrun-service.yaml` from `cloudrun-service.yaml.template`
2. Build and push the Docker image via Cloud Build
3. Deploy with `gcloud run services replace`
4. Grant public access (`allUsers` + `roles/run.invoker`) if `MAKE_PUBLIC=true`
5. Print the service URL

### Script options

```bash
./deploy-cloudrun.sh --help
./deploy-cloudrun.sh --env-file deploy.cloudrun.env --skip-build    # image already in registry
./deploy-cloudrun.sh --env-file deploy.cloudrun.env --skip-public     # skip IAM binding
./deploy-cloudrun.sh --env-file deploy.cloudrun.env --skip-deploy     # build only
```

---

## Manual deploy (step by step)

Equivalent to what `deploy-cloudrun.sh` does. Replace placeholders with your values.

```bash
export GCP_PROJECT_ID=your-project-id
export GCP_REGION=europe-west1
export ARTIFACT_REGISTRY_REPO=cloud-run-source-deploy
export IMAGE_NAME=soccer-benchmark
export IMAGE_TAG=latest
export SERVICE_NAME=soccer-salary-benchmark

export CONTAINER_IMAGE="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${ARTIFACT_REGISTRY_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"
```

### 1. Render the service YAML

```bash
sed \
  -e "s|\${GCP_PROJECT_ID}|${GCP_PROJECT_ID}|g" \
  -e "s|\${SERVICE_NAME}|${SERVICE_NAME}|g" \
  -e "s|\${CONTAINER_IMAGE}|${CONTAINER_IMAGE}|g" \
  cloudrun-service.yaml.template > cloudrun-service.yaml
```

### 2. Build and push the image

Cloud Build runs in GCP (~5–15 min for the first build; image is ~2 GB with models):

```bash
gcloud builds submit . \
  --tag "$CONTAINER_IMAGE" \
  --project "$GCP_PROJECT_ID" \
  --region "$GCP_REGION"
```

Wait until the build log shows **DONE** and `latest: digest: sha256:...` before deploying. Deploying before the push finishes causes:

```
Image '.../soccer-benchmark:latest' not found.
```

Local alternative (build on your machine, then push):

```bash
gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev"

docker build -t "$CONTAINER_IMAGE" .
docker push "$CONTAINER_IMAGE"
```

### 3. Deploy the Cloud Run service

```bash
gcloud run services replace cloudrun-service.yaml \
  --region "$GCP_REGION" \
  --project "$GCP_PROJECT_ID"
```

First revision may take several minutes (pull image + model warm-up).

### 4. Make the service public (optional)

Required for investors or anyone **without** a GCP account:

```bash
gcloud run services add-iam-policy-binding "$SERVICE_NAME" \
  --region "$GCP_REGION" \
  --project "$GCP_PROJECT_ID" \
  --member="allUsers" \
  --role="roles/run.invoker"
```

This step needs `run.services.setIamPolicy`. If you get `PERMISSION_DENIED`, ask a project `run.admin` or owner to run the command above.

### 5. Get the URL and verify

```bash
gcloud run services describe "$SERVICE_NAME" \
  --region "$GCP_REGION" \
  --project "$GCP_PROJECT_ID" \
  --format='value(status.url)'
```

Health check (public, after IAM binding):

```bash
curl "https://YOUR_SERVICE_URL/api/health"
# {"status":"ok"}
```

Authenticated check (works even without public IAM):

```bash
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  "https://YOUR_SERVICE_URL/api/health"
```

---

## Public access without admin rights

Public access is **IAM**, not YAML. Editing `cloudrun-service.yaml` or setting `ingress: all` does **not** allow unauthenticated users.

If you cannot run the IAM binding:

1. **Ask a run.admin** to run the `add-iam-policy-binding` command (one line).
2. **Temporary demo tunnel** (your laptop must stay on):

   ```bash
   # Terminal 1
   gcloud run services proxy "$SERVICE_NAME" \
     --region "$GCP_REGION" \
     --project "$GCP_PROJECT_ID" \
     --port 8080

   # Terminal 2
   ngrok http 8080
   ```

   Share the ngrok URL with investors.

3. **Redeploy elsewhere** (Render, Fly.io, Railway) using the same `Dockerfile` for a host that is public by default.

Set `MAKE_PUBLIC=false` in `deploy.cloudrun.env` or use `--skip-public` so the deploy script does not fail on IAM.

---

## Redeploy after code changes

Full redeploy:

```bash
./deploy-cloudrun.sh --env-file deploy.cloudrun.env
```

Image already built, only config changed:

```bash
./deploy-cloudrun.sh --env-file deploy.cloudrun.env --skip-build
```

---

## Local Docker smoke test

Before pushing to GCP:

```bash
docker build -t soccer-benchmark:local .
docker run --rm -p 8080:8080 soccer-benchmark:local
```

Then open http://localhost:8080 and http://localhost:8080/api/health .

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|--------|-----|
| `Image ... not found` | Deploy before Cloud Build finished | Wait for build **DONE**, then redeploy |
| `403 Forbidden` on URL | No public IAM binding | Run `add-iam-policy-binding` or use identity token |
| `PERMISSION_DENIED` on IAM | Missing `run.services.setIamPolicy` | Ask run.admin; use `--skip-public` |
| `startup-cpu-boost` on Service metadata | Annotation in wrong place | Must be under `spec.template.metadata.annotations` (template is correct) |
| Predictions fail / all null | Missing `lightgbm` | Already in `soccer-benchmark/backend/requirements.txt` |
| Startup probe timeout | Models slow to load | Startup probe allows ~65 s; increase `failureThreshold` if needed |

Cloud Run logs:

```bash
gcloud run services logs read "$SERVICE_NAME" \
  --region "$GCP_REGION" \
  --project "$GCP_PROJECT_ID" \
  --limit 50
```

---

## Cost notes

- **minScale: 1** keeps one instance always running (no cold starts for demos; continuous cost).
- Image + models ≈ **2 GB**; first Cloud Build upload is slow.
- For a one-off demo, consider `minScale: "0"` in the template to scale to zero when idle.

---

## Example: Naboo production deploy

```bash
cp deploy.cloudrun.env.example deploy.cloudrun.env
# edit: GCP_PROJECT_ID=naboo-app-365515, GCP_REGION=europe-west1

./deploy-cloudrun.sh --env-file deploy.cloudrun.env
```

If IAM binding fails, a run.admin runs:

```bash
gcloud run services add-iam-policy-binding soccer-salary-benchmark \
  --region europe-west1 \
  --project naboo-app-365515 \
  --member="allUsers" \
  --role="roles/run.invoker"
```

Service URL pattern: `https://soccer-salary-benchmark-PROJECT_NUMBER.REGION.run.app`
