# Soccer Salary Benchmark — API guide

REST API built with FastAPI. Interactive docs are generated from the code:

| URL | What |
|-----|------|
| `/docs` | Swagger UI (interactive "Try it out") |
| `/redoc` | ReDoc (reference layout) |
| `/openapi.json` | Raw OpenAPI 3.1 schema |

These work both locally (`http://localhost:8001/docs`) and on the deployed
Cloud Run service (nginx proxies them to the backend).

## Base URLs

| Environment | Base URL |
|-------------|----------|
| Local backend | `http://localhost:8001` |
| Local via Vite dev server | `http://localhost:5173` (proxies `/api`) |
| Docker / Cloud Run | `https://<service-url>` (nginx on port 8080) |

All endpoints are prefixed with `/api`.

## Authentication

The API itself has **no authentication** — no accounts, cookies, or API keys.
CORS allows any origin (`allow_credentials=False`, safe because nothing is
credentialed).

On **Cloud Run**, access control happens in front of the app via IAM:

- **Public service** (`allUsers` has `roles/run.invoker`): anyone can call it.
- **Private service**: send a Google identity token —

  ```bash
  curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
    https://<service-url>/api/health
  ```

## Endpoints

### `GET /api/health`

Liveness/startup check. Returns `200` only after the player pool and all
model variants are loaded (Cloud Run's startup probe points here).

```bash
curl http://localhost:8001/api/health
# {"status": "ok"}
```

### `GET /api/players/search?q=<query>&limit=<n>`

Case-insensitive substring search on player names. `q` needs at least 2
characters; `limit` is 1–50 (default 20).

```bash
curl "http://localhost:8001/api/players/search?q=haaland"
```

Returns a list of players with `id`, position, team, league, nationality,
age, market value, and salary (when known). Use the `id` for the benchmark.

### `GET /api/players/{player_id}`

Full detail for one player (row ID from search results). `404` if the ID is
out of range.

```bash
curl http://localhost:8001/api/players/1234
```

### `GET /api/players/options`

Distinct positions and competitions in the pool — used to populate the
manual ("Custom Player") benchmark form.

```bash
curl http://localhost:8001/api/players/options
```

### `POST /api/benchmark`

Run the salary benchmark. Three ways to identify the player, in order of
preference:

**1. By `player_id`** (preferred — names are not unique):

```bash
curl -X POST http://localhost:8001/api/benchmark \
  -H "Content-Type: application/json" \
  -d '{"player_id": 1234}'
```

**2. By `player_name`:**

```bash
curl -X POST http://localhost:8001/api/benchmark \
  -H "Content-Type: application/json" \
  -d '{"player_name": "Erling Haaland", "range_width": "wide"}'
```

**3. Manual / hypothetical player** (`main_position` and `age_months`
required; `market_value_current_eur` optional — without it the engine routes
to the no-market-value fallback model):

```bash
curl -X POST http://localhost:8001/api/benchmark \
  -H "Content-Type: application/json" \
  -d '{
    "main_position": "Centre-Forward",
    "competition_id": "GB1",
    "competition_country": "England",
    "age_months": 300,
    "market_value_current_eur": 50000000,
    "annual_fixed_eur": 10000000
  }'
```

Options:

| Field | Values | Default | Effect |
|-------|--------|---------|--------|
| `range_width` | `"normal"`, `"wide"` | `"normal"` | Width of the expected salary range |
| `full_comparables` | `true`/`false` | `false` | Return all comparables instead of top 10 |

Key response fields:

| Field | Meaning |
|-------|---------|
| `expected_salary_low/median/high_eur` | Calibrated expected salary range (EUR gross/year) |
| `actual_salary_eur`, `salary_percentile` | Known salary and its position in the expected range |
| `salary_status` | `OVERPAID` / `UNDERPAID` / `FAIRLY_PAID` / `UNKNOWN` |
| `benchmark_confidence` | `HIGH` / `MEDIUM` / `LOW` — based on comparables quality |
| `model_used` | `full`, `no_mv`, `no_mv_no_pos`, or `no_mv_no_age` (fallback routing) |
| `comparable_players` | Similar players with similarity scores |
| `benchmark_warning` | Present when results should be read with caution |

## Errors

All errors return `{"detail": "<message>"}`:

| Status | When |
|--------|------|
| `400` | Valid schema but unusable request (e.g. manual benchmark without `age_months`) |
| `404` | Player not found (unknown `player_id`, no name match) |
| `422` | Request failed schema validation (FastAPI/Pydantic) |
| `500` | Unexpected server error — details are logged server-side, never leaked |
| `503` | Player pool file missing on the server |

## Interpretation caveats

Predictions describe the **market**, which may embed biases — nationality is
a model feature, and OVERPAID/UNDERPAID labels are market descriptions, not
normative advice for contract decisions. See the "Limitations" section of the
main [README](README.md).
