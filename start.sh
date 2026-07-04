#!/bin/sh
set -e

# Start uvicorn from the backend directory (hyphen in soccer-benchmark prevents
# dotted module path imports; cd + module name works identically to local dev)
cd /app/soccer-benchmark/backend
uvicorn main:app \
    --host 127.0.0.1 \
    --port 8001 \
    --workers 1 \
    --proxy-headers &
UVICORN_PID=$!

cd /app
nginx -g 'daemon off;' &
NGINX_PID=$!

# Cloud Run only watches whether *this* process (and port 8080) is alive.
# Plain `uvicorn & ; exec nginx` leaves a dead backend undetected: nginx keeps
# answering on 8080 while every /api/* call 502s forever, with no restart.
# /bin/sh here is dash (no `wait -n`), so poll instead: exit as soon as
# EITHER process dies, so the container exits and Cloud Run replaces it.
trap 'kill -TERM "$UVICORN_PID" "$NGINX_PID" 2>/dev/null' TERM INT

while kill -0 "$UVICORN_PID" 2>/dev/null && kill -0 "$NGINX_PID" 2>/dev/null; do
    sleep 1
done

kill -TERM "$UVICORN_PID" "$NGINX_PID" 2>/dev/null
wait
exit 1
