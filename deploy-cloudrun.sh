#!/usr/bin/env bash
# Build, deploy, and optionally publish the Soccer Salary Benchmark on Cloud Run.
#
# Usage:
#   ./deploy-cloudrun.sh
#   ./deploy-cloudrun.sh --env-file deploy.cloudrun.env
#   GCP_PROJECT_ID=my-proj GCP_REGION=europe-west1 ./deploy-cloudrun.sh
#
# Options:
#   --env-file FILE   Load variables from a shell-style env file
#   --skip-build      Skip Cloud Build (image must already exist in Artifact Registry)
#   --skip-deploy     Skip gcloud run services replace
#   --skip-public     Skip allUsers invoker IAM binding
#   --help            Show this help
#
# Prerequisites:
#   gcloud CLI, authenticated (gcloud auth login)
#   Cloud Build API, Cloud Run API, Artifact Registry API enabled
#   Permission to run builds and update Cloud Run services
#   For MAKE_PUBLIC=true: run.services.setIamPolicy on the service

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/cloudrun-service.yaml.template"
RENDERED="${SCRIPT_DIR}/cloudrun-service.yaml"

SKIP_BUILD=false
SKIP_DEPLOY=false
SKIP_PUBLIC=false
ENV_FILE=""

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \?//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="${2:?--env-file requires a path}"
      shift 2
      ;;
    --skip-build) SKIP_BUILD=true; shift ;;
    --skip-deploy) SKIP_DEPLOY=true; shift ;;
    --skip-public) SKIP_PUBLIC=true; shift ;;
    --help|-h) usage 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage 1
      ;;
  esac
done

if [[ -n "$ENV_FILE" ]]; then
  if [[ ! -f "$ENV_FILE" ]]; then
    echo "Env file not found: $ENV_FILE" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  set -a
  source "$ENV_FILE"
  set +a
fi

# Defaults (override via env file or environment)
GCP_PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
GCP_REGION="${GCP_REGION:-europe-west1}"
ARTIFACT_REGISTRY_REPO="${ARTIFACT_REGISTRY_REPO:-cloud-run-source-deploy}"
IMAGE_NAME="${IMAGE_NAME:-soccer-benchmark}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
SERVICE_NAME="${SERVICE_NAME:-soccer-salary-benchmark}"
MAKE_PUBLIC="${MAKE_PUBLIC:-true}"

if [[ -z "$GCP_PROJECT_ID" || "$GCP_PROJECT_ID" == "(unset)" ]]; then
  echo "GCP_PROJECT_ID is required. Set it in deploy.cloudrun.env or export it." >&2
  exit 1
fi

CONTAINER_IMAGE="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${ARTIFACT_REGISTRY_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"

render_service_yaml() {
  if [[ ! -f "$TEMPLATE" ]]; then
    echo "Template not found: $TEMPLATE" >&2
    exit 1
  fi
  sed \
    -e "s|\${GCP_PROJECT_ID}|${GCP_PROJECT_ID}|g" \
    -e "s|\${SERVICE_NAME}|${SERVICE_NAME}|g" \
    -e "s|\${CONTAINER_IMAGE}|${CONTAINER_IMAGE}|g" \
    "$TEMPLATE" > "$RENDERED"
  echo "Rendered ${RENDERED}"
}

build_image() {
  echo "==> Building and pushing ${CONTAINER_IMAGE}"
  gcloud builds submit "$SCRIPT_DIR" \
    --tag "$CONTAINER_IMAGE" \
    --project "$GCP_PROJECT_ID" \
    --region "$GCP_REGION"
}

deploy_service() {
  echo "==> Deploying Cloud Run service ${SERVICE_NAME} in ${GCP_REGION}"
  gcloud run services replace "$RENDERED" \
    --region "$GCP_REGION" \
    --project "$GCP_PROJECT_ID"
}

make_public() {
  echo "==> Granting public invoker access (allUsers)"
  if ! gcloud run services add-iam-policy-binding "$SERVICE_NAME" \
    --region "$GCP_REGION" \
    --project "$GCP_PROJECT_ID" \
    --member="allUsers" \
    --role="roles/run.invoker"; then
    echo ""
    echo "WARNING: Could not set public IAM binding (need run.services.setIamPolicy)." >&2
    echo "Ask a project run.admin to run:" >&2
    echo "  gcloud run services add-iam-policy-binding ${SERVICE_NAME} \\" >&2
    echo "    --region ${GCP_REGION} --project ${GCP_PROJECT_ID} \\" >&2
    echo "    --member=allUsers --role=roles/run.invoker" >&2
    echo ""
    echo "Or use: ./deploy-cloudrun.sh --skip-public  and tunnel with gcloud run services proxy" >&2
    return 0
  fi
}

print_url() {
  local url
  url="$(gcloud run services describe "$SERVICE_NAME" \
    --region "$GCP_REGION" \
    --project "$GCP_PROJECT_ID" \
    --format='value(status.url)' 2>/dev/null || true)"
  if [[ -n "$url" ]]; then
    echo ""
    echo "Service URL: ${url}"
    echo "Health:      ${url}/api/health"
  fi
}

echo "Project:  ${GCP_PROJECT_ID}"
echo "Region:   ${GCP_REGION}"
echo "Image:    ${CONTAINER_IMAGE}"
echo "Service:  ${SERVICE_NAME}"
echo ""

render_service_yaml

if [[ "$SKIP_BUILD" == false ]]; then
  build_image
else
  echo "==> Skipping build (--skip-build)"
fi

if [[ "$SKIP_DEPLOY" == false ]]; then
  deploy_service
else
  echo "==> Skipping deploy (--skip-deploy)"
fi

if [[ "$SKIP_PUBLIC" == false && "$MAKE_PUBLIC" == "true" ]]; then
  make_public
else
  echo "==> Skipping public IAM (--skip-public or MAKE_PUBLIC=false)"
fi

print_url
echo "Done."
