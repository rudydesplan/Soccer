# Soccer Salary Benchmark

A complete prototype that answers the question: **"Are we paying this player too much or too little compared to the market?"**

Given a player's profile (position, age, league, market value), the system predicts an expected salary range, compares against similar players, and communicates the result with confidence scoring — all through a visual interface a sporting director can understand.

## How this prototype maps to the challenge deliverables

| Deliverable | Where it lives | Where it's documented |
|---|---|---|
| 1. Salary acquisition + cleaning pipeline | `capology_pipeline/` | [Data Sources](#data-sources-considered-and-chosen), [Pipeline](#data-acquisition-pipeline), [Cleaning](#cleaning-and-transformation-decisions) |
| 2. Comparison/benchmarking algorithm | `salary_benchmark/` | [Benchmarking Algorithm](#benchmarking-algorithm) — incl. [why the range is model-predicted rather than a peer-salary percentile](#why-a-model-predicted-range-instead-of-peer-salary-percentiles) |
| 3. Visualisation interface | `soccer-benchmark/` (FastAPI + React) | [Visualisation Interface](#visualisation-interface) — select a pool player **or enter a custom player** at `/manual` |
| 4. README (sources, decisions, logic, limitations, next steps) | this file | All sections below; brutally honest [Limitations](#limitations) |

Robustness asks from the brief: explicit error handling ([here](#error-handling)), input validation ([here](#input-validation)), idempotent re-runs ([here](#key-design-decisions) — disk cache + chunk/resume + atomic writes), and tests (435 across pipeline, engine, API, and data integrity).

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Data Sources Considered and Chosen](#data-sources-considered-and-chosen)
3. [Data Acquisition Pipeline](#data-acquisition-pipeline)
4. [Cleaning and Transformation Decisions](#cleaning-and-transformation-decisions)
5. [Feature Engineering](#feature-engineering)
6. [Benchmarking Algorithm](#benchmarking-algorithm)
7. [Model Training and Selection](#model-training-and-selection)
8. [Visualisation Interface](#visualisation-interface)
9. [How to Run](#how-to-run)
10. [Limitations](#limitations)
11. [What I Would Do Differently With More Time](#what-i-would-do-differently-with-more-time)

---

## Architecture Overview

```
data (1).csv (SoccerSolver input: 19,476 players)
    │
    ▼
┌─────────────────────────────────────────┐
│  Enrichment Pipeline (capology_pipeline)│
│  • Capology salary scraping + matching  │
│  • Transfermarkt market value API       │
│  • Chunk/resume mode for robustness     │
└─────────────────────────────────────────┘
    │
    ▼
data_full.csv (19,476 rows × 24 columns)
    │
    ▼
┌─────────────────────────────────────────┐
│  Feature Engineering (build_model_...)  │
│  • Log transforms, age, contract dates  │
│  • Missing-value indicators             │
└─────────────────────────────────────────┘
    │
    ├──► data_test.csv (model training: 16 features + player ID)
    └──► player_pool.csv (benchmark pool: 35 columns)
              │
              ▼
┌─────────────────────────────────────────┐
│  ML Models (AutoGluon LightGBM ensembles)│
│  • 4 variants: full + 3 fallbacks for   │
│    missing market value / position / age │
│  • All trained on the 3,363 salary rows │
│  • Full: RMSE 0.539, R² 0.746 (grouped  │
│    holdout), MAPE 46.6%                 │
│  • Each variant has its own out-of-fold │
│    calibration (wider for fallbacks)    │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  Salary Benchmark Engine                │
│  • Variant routing on feature availability│
│  • Prediction → calibrated range        │
│  • Out-of-fold calibration (no leakage) │
│  • Comparable player search (3 levels)  │
│  • Honest confidence scoring            │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  Interface (FastAPI + React)            │
│  • Player search with autocomplete      │
│  • Salary range visualisation           │
│  • Normal/wide range toggle             │
│  • Comparable players table             │
│  • Confidence badge                     │
└─────────────────────────────────────────┘
```

---

## Data Sources Considered and Chosen

### Sources evaluated

| Source | Type | Coverage | Reliability | Decision |
|--------|------|----------|-------------|----------|
| **Capology** | Web scraping | Top 5 European leagues + lower divisions | High — updated regularly, individual player pages | ✅ **Chosen** |
| **Transfermarkt** | API (self-hosted) | Global, 100k+ players | High — market values, transfer history | ✅ **Chosen** (market value) |
| FBref / StatsBomb | Statistics | Performance data, no salaries | N/A for salary | ❌ No salary data |
| Kaggle datasets | Static CSV | Outdated (2020-2022) | Low — stale, incomplete | ❌ Too old |
| DNCG (France) | Official reports | French clubs only | High but aggregated per club | ❌ Not per-player |
| LFP / DFL reports | Official | Single country | Medium — aggregated | ❌ Not granular enough |
| SalarySport / Spotrac | Web | Partial, US-focused | Medium | ❌ Limited European coverage |

### Why Capology

- **Individual player salary data** (annual gross, bonus, contract details).
- **Coverage**: Premier League, LaLiga, Serie A, Bundesliga, Ligue 1 — the same leagues in our input file.
- **Structured data**: contract dates, release clauses, status (active/loan/free agent).
- **Searchable player index** (46,000+ players) enabling programmatic matching.

### Why Transfermarkt (self-hosted API)

- The input CSV provides Transfermarkt player/team IDs.
- Market value is the single most predictive feature for salary.
- Self-hosting the [felipeall/transfermarkt-api](https://github.com/felipeall/transfermarkt-api) avoids rate limits and gives full control.

### Legality and terms-of-use considerations

The challenge asks for "open and legal" sources; scraping deserves an honest note rather than silence:

- **Capology** publishes salary estimates on publicly accessible pages with no authentication, paywall, or technical access barrier. This prototype scrapes those pages with rate limiting and exponential backoff, caches every response to disk so each page is fetched **once** (re-runs generate zero traffic), and stores only the extracted fields — raw pages are never redistributed. Salary figures themselves are factual data (not creative work) and are Capology's *estimates*, aggregated from press reporting.
- **Transfermarkt** is accessed through a self-hosted open-source API wrapper rather than direct page scraping, again with disk caching. Only the market values and profile fields for the players in the input file are fetched.
- **Residual risk, stated plainly**: site terms of use for both sources may restrict automated access regardless of technical accessibility. For a one-off recruitment prototype with cached, minimal-volume access this is a limited and, in my judgment, acceptable risk. A **production** deployment must not rely on scraping: it should license data commercially (e.g. Capology's own data services, Sportradar, or league-official disclosures) — this is listed in [What I Would Do Differently](#what-i-would-do-differently-with-more-time).

---

## Data Acquisition Pipeline

### Architecture

```
capology_pipeline/
├── config.py          # All tunables (workers, ports, timeouts)
├── normalize.py       # Text normalization, transliteration, date parsing
├── cache.py           # Disk cache for HTTP responses (idempotency)
├── http_client.py     # HTTP/2 sessions with retry logic
├── capology.py        # Capology scraping, Splink matching, page parsing
├── transfermarkt.py   # Local TM server management + market value client
├── enricher.py        # Orchestrator: parallel enrichment + chunk/resume
└── cli.py             # CLI entry point
```

### Key design decisions

1. **Disk cache for all HTTP responses** — every Capology page and Transfermarkt API call is cached to disk. Reruns are instant and don't hit external servers.

2. **Chunk/resume mode** — the pipeline splits the 19,476-row input into chunks (default 500 rows). Each chunk is written atomically. If the process crashes, rerunning skips completed chunks.

3. **Concurrent scraping** — Capology uses 12 thread workers; Transfermarkt uses 16 concurrent requests against the local API with 8 uvicorn workers.

4. **Retry with backoff** — HTTP client retries 4 times with exponential backoff (1.5s, 3s, 6s) on 429/5xx errors.

5. **Player matching via Splink** — fuzzy probabilistic record linkage handles name variations (transliteration, nicknames, fuller names). Birth date and club validation prevent false positives.

### Error handling

- Network failures: retry with backoff, then skip player (null salary).
- Name mismatches: multi-stage matching (exact → last-token → fuzzy → mononym → nickname).
- Bad matches: birth date + club token validation rejects wrong candidates.
- Format changes: parser extracts from JavaScript variables in page source; if structure changes, fields return null rather than crashing.

### Input validation

- Missing required columns raise `ValueError` immediately.
- Numeric fields are coerced with `errors="coerce"` — invalid values become NaN.
- Dates are parsed with timezone handling and normalization.

---

## Cleaning and Transformation Decisions

### Handling missing data

| Field | Missing count | Decision |
|-------|:---:|---|
| `capology_url` | 9,872 / 19,476 | Expected — many lower-division players are not in Capology |
| `annual_fixed_eur` | 16,113 / 19,476 | Only 3,363 have salary — used for training; all 19,476 used for comparables |
| `market_value_current_eur` | 924 / 19,476 | Missing-value indicator (`has_market_value`) added |
| `status` | 10,489 / 19,476 | Kept as-is; AutoGluon handles missing categoricals natively |
| `contract_months_remaining` | 12,567 / 19,476 | NaN when no expiration date; model handles gracefully |

### Reproducibility

Feature builds use `--as-of-date` to fix the reference date for `contract_months_remaining`. Metadata sidecar files (`*.meta.json`) record the exact date used. Without `--as-of-date`, the default is today — which means rebuilding features tomorrow changes the dataset.

### Handling inconsistent data

- **Name transliteration**: `Ø→O`, `ø→o`, `ı→i`, `ł→l`, `đ→d`, `æ→ae`, `ß→ss` applied before matching.
- **Duplicate players**: 1,948 rows involve a shared `transfermarkt_player_id` (986 rows beyond the first occurrence — same player in multiple competition entries). Kept as-is because the input file contains them; grouped splits prevent them leaking across train/test.
- **Bad Capology matches** (82 rows): identified via `/player/...` raw URL pattern, validated as wrong players, reset to null.
- **Birth-date audit (pipeline-enforced)**: every enriched row is re-checked against the scraped Capology birth date (`PlayerEnricher.audit_capology_dob`). Enrichment whose Capology DOB contradicts the source DOB is nulled out — both right after enrichment and again when merging resumed chunk files, so stale chunks can never reintroduce wrong-player salaries.
- **Salary currency**: all values normalized to EUR (Capology provides EUR gross).

### What was dropped

- Original `market_value` column from input (stale) — replaced by freshly fetched `market_value_current_eur`.
- Raw monetary columns after log-transformation (to prevent leakage in model features).

---

## Feature Engineering

### Final model features (16 columns)

| Feature | Type | Description |
|---------|------|-------------|
| `main_position` | categorical | Player's primary position |
| `nationality` | categorical | Player nationality |
| `competition_id` | categorical | League identifier (GB1, ES1, etc.) |
| `competition_country` | categorical | Country of the league |
| `status` | categorical | Contract status (active, loan, free agent) |
| `season_start_year` | integer | Season year (2025) |
| `age_months` | float | Player age in fractional months at season start |
| `contract_length_months` | float | Total contract length from Capology gross contract |
| `contract_months_remaining` | float | Months from today until contract expiration |
| `contract_recency_months` | float | Months since contract was signed |
| `has_contract_dates` | binary | Whether signed + expiration dates are known |
| `has_market_value` | binary | Whether market value is available |
| `log_market_value_current_eur` | float | log1p(market_value) |
| `has_release_clause` | binary | Whether a release clause exists |
| `log_release_clause_eur` | float | log1p(release_clause) |
| `log_annual_fixed_eur` | float | **Target**: log1p(annual_fixed_salary) |

### Design choices

- **Log-scale target**: salary distributions are heavily right-skewed; log1p normalizes them for regression.
- **Categorical columns kept as text**: AutoGluon handles encoding internally (label encoding for LightGBM).
- **Missing-value indicators**: explicit binary flags let the model learn different behavior when data is absent vs. zero.
- **Calendar-fractional months**: precise month calculations using DateOffset arithmetic, not approximate 30.44-day divisions.

---

## Benchmarking Algorithm

### Overview

```
Player selected
    │
    ▼
Variant routing: pick model based on which features the player has
(full / no_mv / no_mv_no_pos / no_mv_no_age — refuse if position AND age missing)
    │
    ▼
Selected AutoGluon model predicts log(salary)
    │
    ▼
That variant's calibration converts prediction → salary range (low / median / high)
    │
    ▼
Comparable player search finds similar profiles
(filters whose target-side value is unknown are skipped)
    │
    ▼
Confidence scoring evaluates reliability (capped for fallback variants)
    │
    ▼
Output: range, status, comparables, confidence, model_used, warning
```

### Why a model-predicted range instead of peer-salary percentiles

The challenge describes the algorithm as *"calculate an expected salary range by comparing them with players of a similar profile."* The most literal implementation would take the comparable players' salaries and report their percentiles as the range. This prototype deliberately does **not** do that, and the reasoning is the core design decision of the project:

1. **Only 3,363 of 19,476 pool players (17%) have a known salary.** A peer-percentile range needs enough *salary-known* comparables after filtering by position, league, age, and market value. For most queries — especially outside the top five leagues — that intersection contains a handful of players or none. A range built from 3 salaries is noise wearing a suit; a range that often can't be computed at all fails the product question.
2. **Peer percentiles confound "similar profile" with "similar pay".** Comparables are selected on profile, but their salaries embed each club's wage structure, contract timing, and negotiation history. The p25–p75 of 8 arbitrary peers has no calibrated meaning — you cannot state its coverage probability. The model + out-of-fold residual approach yields a range with a **measured** coverage (50.5% observed vs 50% target on the honest grouped holdout).
3. **The model generalizes between profiles; a lookup cannot.** A 24-year-old winger with a €40M market value in Serie A gets a sensible estimate even if no near-identical peer has a known salary, because the model interpolates across position, league, age, and market-value gradients learned from all 3,363 salaries at once.

**The comparison logic the challenge asks for is still central — it is the honesty layer rather than the arithmetic:** comparables are searched with 3-level relaxation, displayed with similarity scores, and *gate the confidence*: the hard rule forces LOW confidence when no same-league comparable has a known salary, the "model-only" warning fires when zero comparables (or zero with salary) support the estimate, and the salary percentile reported to the user **is** computed among the comparables' actual salaries. In short: the model prices the player, the peers decide how much to trust the price.

### Reading the result — a non-technical example

What a sporting director sees for **Erling Haaland** (Manchester City, actual salary €31.5M):

> **Expected salary range (wide): €10.5M – €37.5M, median €20.0M.**
> **Verdict: FAIRLY PAID** — his actual €31.5M sits inside the range, in the upper part.
> **Confidence: LOW** — only 5 comparables; superstars have few true peers, and the tool says so instead of bluffing.
> **Percentile: 100th** among his comparables — he out-earns every similar player.
> **Comparables**: Alexander Isak, João Pedro, Benjamin Sesko… each with a similarity score, so you can judge whether the comparison set makes sense.

How to act on each field:

- **The range** is where the market would price this profile. Negotiating inside it is defensible; far outside it needs a story (leadership, marketing value, resale upside — things the model can't see).
- **The verdict** (OVERPAID / FAIRLY PAID / UNDERPAID) compares actual salary to the range. UNKNOWN means we don't have the player's real salary — enter it on the [Custom Player page](#pages) to get the verdict.
- **Confidence** is the tool grading its own homework: HIGH means many similar players with known salaries back the estimate; LOW means you're looking at an extrapolation — treat it as a starting point, not an answer.
- **Normal vs wide range**: normal (50%) is the tight negotiation band — half of real salaries fall inside it; wide (80%) is the plausibility envelope — a salary outside even the wide range is a genuine outlier.
- **Warnings** (amber banner) are never decorative: "fallback model" means key data was missing and the estimate is weaker; "model-only" means no peer validates the number.

### 1. Salary prediction

The AutoGluon LightGBM ensemble predicts `log_annual_fixed_eur`. The prediction is converted back to EUR via `expm1()`.

Four model variants are served, selected by a routing cascade on what the player is missing:

| Player has… | Variant used | Model directory | Confidence cap | Pool players |
|---|---|---|:---:|:---:|
| Market value + position + age | `full` | `models/autogluon` | none | 18,549 |
| Position + age, **no market value** | `no_mv` | `models/autogluon_no_mv` | MEDIUM | 865 |
| Age, **no position** (MV ignored if present) | `no_mv_no_pos` | `models/autogluon_no_mv_no_pos` | always LOW | 51 |
| Position, **no age** (MV ignored if present) | `no_mv_no_age` | `models/autogluon_no_mv_no_age` | always LOW | 9 |
| **Neither position nor age** | — refused with a clear error | — | — | 2 |

Each fallback is trained with the missing features excluded — nothing is imputed or invented. They are measurably weaker (see [Fallback models](#fallback-models-players-with-missing-key-features)), so each ships with its own wider out-of-fold calibration, an explicit "fallback model" warning in the response, and a capped confidence badge. The response's `model_used` field always states which variant produced the estimate.

Edge case: 3 players have a market value but no position. They route to `no_mv_no_pos`, which discards their market value — a dedicated fourth fallback for 3 players wasn't worth the training and maintenance cost, and the warning makes the trade-off visible.

### 2. Calibrated salary range

Residuals (actual − predicted) are collected **out-of-fold** with 5-fold grouped cross-validation (grouped by `transfermarkt_player_id`) over the 3,363 known-salary players — see [Calibration method](#calibration-method). Each model variant has its **own** calibration file, built with the same protocol on that variant's feature set (`calibration.json`, `calibration_no_mv.json`, `calibration_no_mv_no_pos.json`, `calibration_no_mv_no_age.json`), so a weaker model automatically yields a wider range. Percentiles of these residuals define the range:

- **Normal range**: p25 to p75 of residuals → targets ~50% coverage
- **Wide range**: p10 to p90 → targets ~80% coverage

```
low    = expm1(prediction + residual_p25)
median = expm1(prediction)
high   = expm1(prediction + residual_p75)
```

### 3. Comparable player search (3 levels)

The system finds similar players using progressively relaxed criteria:

| Level | Position | League | Age | Market Value | Min players |
|:---:|---|---|---|---|:---:|
| 1 (strict) | Same | Same competition | ±24 months | 0.5x – 2.0x | 5 |
| 2 (relaxed) | Same | Same country | ±36 months | 0.33x – 3.0x | 5 |
| 3 (broad) | Same group | Any | ±60 months | 0.25x – 4.0x | 5 |

Position groups: attacker, midfielder, defender, goalkeeper.

**Missing target values**: any filter whose target-side value is unknown is skipped at every level — a player without a market value keeps the position/league/age filters, a player without a position keeps the league/age/market-value filters, and so on. Otherwise a NaN target value would match nobody.

### 4. Similarity scoring (0–1)

Each comparable receives a weighted similarity score:

| Component | Weight | Logic |
|-----------|:---:|---|
| Market value | 35% | Exponential decay on log-ratio |
| League | 25% | 1.0 same competition, 0.6 same country, 0.2 other |
| Position | 20% | 1.0 exact, 0.5 same group, 0.0 different |
| Age | 12% | Exponential decay (half-life 24 months) |
| Contract | 8% | Exponential decay on month difference |

When the target's value for a component is unknown (market value, position, age, or contract), that component scores a neutral **0.5** for every comparable — missing data is neither rewarded nor punished.

### 5. Confidence scoring

| Confidence | Criteria |
|:---:|---|
| **HIGH** | ≥50 comparables, avg similarity ≥0.75, range width <60% of median |
| **MEDIUM** | ≥20 comparables, avg similarity ≥0.60, range width <100% of median |
| **LOW** | Otherwise |

**Hard rule**: if no comparable player from the same league has a known salary, confidence is always **LOW** regardless of other criteria — we acknowledge we're extrapolating from other leagues.

**Fallback caps**: the `no_mv` variant is capped at **MEDIUM** (a weaker model shouldn't wear a HIGH badge), and the `no_mv_no_pos` / `no_mv_no_age` variants are always **LOW** — their holdout accuracy (see [Fallback models](#fallback-models-players-with-missing-key-features)) does not support anything stronger.

### 6. Salary status

| Status | Condition |
|---|---|
| UNDERPAID | Actual salary < expected low |
| FAIRLY_PAID | Actual salary within range |
| OVERPAID | Actual salary > expected high |
| UNKNOWN | No actual salary available |

---

## Model Training and Selection

### Models compared

Cross-validated comparison on identical grouped 5-fold splits (`model_comparison.py`, equal 180s budget per fold, results in `models/model_comparison_results.json`):

| Model | RMSE (mean ± std) | R² | Notes |
|-------|:---:|:---:|---|
| **AutoGluon** (LightGBM ensemble) | **0.536 ± 0.021** | **0.769** | Production choice |
| FLAML (grouped inner CV) | 0.573 ± 0.039 | 0.735 | Good, higher variance |
| Ridge (imputed + scaled) | 0.594 ± 0.023 | 0.715 | Interpretable linear baseline |

AutoGluon has the lowest mean RMSE on every fold, but with only 5 paired folds none of the pairwise Wilcoxon tests reach significance (p = 0.06–0.13) — the ranking is consistent, not statistically proven.

### Training protocol (evaluation and production kept separate)

1. **Evaluation**: grouped 80/20 holdout split by `transfermarkt_player_id` (2,696 train / 667 test — the same player never appears on both sides). A model is trained on the train split only and scored on the held-out 20%. These are the honest numbers below.
2. **Production**: a final model is trained on **all 3,363 salary rows** and saved to `models/autogluon`. Its quality estimate comes from step 1.

Both models, the split definition, data hash, feature list, and AutoGluon version are recorded in `models/autogluon_results.json`.

### Business accuracy (grouped holdout, n=667 — what matters to a sporting director)

| Metric | Value | Interpretation |
|--------|:---:|---|
| Predictions within ±20% of actual | **32.2%** | ~3 in 10 predictions are very close |
| Predictions within ±50% of actual | **71.1%** | ~7 in 10 are in the right ballpark |
| MAPE | **46.6%** | Average error is 47% of actual salary |
| Median APE | **31.4%** | Half of predictions err by less than 32% |

**Holdout metrics (grouped by player, 20% held out)**:
| Metric | Value |
|--------|:---:|
| RMSE | 0.539 |
| R² | 0.747 |
| MAPE | 46.6% |
| Within ±20% | 32.2% |
| Within ±50% | 71.1% |

**Honest interpretation**: the model is useful for orientation (is this player in a €2M or €10M range?) but not precise enough for contract negotiation. This is expected given that salary depends heavily on agent leverage, commercial value, and club finances — factors we don't have. Holdout MAPE is ~47% with grouped splits (same player never in both train and test). In-sample numbers computed with the production model on its own training data look better but are optimistic by construction; `models/evaluation_report.json` labels them `in_sample_production_model`.

### Calibration method

Salary ranges use **grouped out-of-fold residuals** (5-fold `GroupKFold` by `transfermarkt_player_id`) to avoid two leakage paths: the production model saw all training data (so in-sample residuals would be too tight), and duplicated players could otherwise appear on both sides of a fold. Five fold models are trained (same preset as production, 15 minutes per fold) on ~80% each and residuals are collected on the held-out ~20%, then merged.

- Normal range: p25–p75 of out-of-fold residuals (targets ~50% coverage)
- Wide range: p10–p90 of out-of-fold residuals (targets ~80% coverage)

Measured coverage for the current model (`models/evaluation_report.json`):

| Scope | Normal (p25–p75) | Wide (p10–p90) |
|-------|:---:|:---:|
| **Grouped holdout** (honest, n=667) | 50.5% | 77.2% |
| In-sample production model (optimistic) | 58.1% | 86.3% |

Holdout coverage is close to the 50% / 80% targets. In-sample coverage is higher because the production model trained on all salary rows.

### Fallback models (players with missing key features)

Market value is the strongest feature (permutation importance 0.596), but 924 pool players have none — and 51 players (48 of them within those 924) also miss their position, 9 their age. Rather than refusing them or silently imputing, three fallback AutoGluon models are trained with the missing features excluded (`train_fallback_no_mv.py --variant <name>`), each evaluated on the **identical grouped holdout** as the full model (same seed, same 667 test players):

| Metric | Full (with MV) | `no_mv` | `no_mv_no_pos` | `no_mv_no_age` |
|--------|:---:|:---:|:---:|:---:|
| RMSE | **0.539** | 0.683 | 0.703 | 0.887 |
| R² | **0.747** | 0.592 | 0.568 | 0.314 |
| MAPE | **46.6%** | 62.5% | 66.7% | 93.4% |
| Median APE | **31.4%** | 43.5% | 43.8% | 54.0% |
| Within ±20% | **32.2%** | 24.7% | 21.9% | 19.2% |
| Within ±50% | **71.1%** | 57.1% | 58.2% | 46.5% |

Age turns out to matter much more than position once market value is gone: dropping position barely hurts (R² 0.592 → 0.568), while dropping age collapses R² to 0.314 — the `no_mv_no_age` estimate is little more than a league/nationality/contract prior.

Each variant has its own grouped out-of-fold calibration (normal-range width in log-space: full 0.65, `no_mv` 0.92, `no_mv_no_pos` 0.92, `no_mv_no_age` 1.16) — the intervals honestly reflect each model's real error. At benchmark time the engine routes on feature availability, comparable search skips the filters whose target-side value is unknown, the response carries `model_used` plus an explicit warning, and confidence is capped at MEDIUM for `no_mv` and forced to LOW for the two weaker variants.

### Why AutoGluon for production

1. **Robust categorical/missing-value handling** — no manual preprocessing needed.
2. **Bagged + stacked ensemble** — reduces variance on small training set (3,363 rows).
3. **Simple inference API** — `predictor.predict(df)` works directly on new data.
4. **Refit on full data** — `refit_full()` retrains the best model on all training data after validation.

**Intentionally excluded model families**: neural-net and foundation-model families (`torch`/`fastai`, `catboost`, `tabpfn`/`tabm`) are excluded via `excluded_model_types` in **every** training path — production (`train_autogluon_cpu.py`), calibration folds, evaluation CV, and model comparison — even though their dependencies are installed in the venv (see `requirements-ml.txt`). Tree ensembles dominate on this small tabular dataset, Apple Silicon GPU is unsupported by AutoGluon, and keeping one model family everywhere means the calibration fold models genuinely match the production model.

### Feature importance (top 5, grouped holdout)

| Feature | Permutation Importance |
|---------|:---:|
| `log_market_value_current_eur` | 0.596 |
| `age_months` | 0.578 |
| `competition_id` | 0.067 |
| `contract_length_months` | 0.024 |
| `contract_recency_months` | 0.020 |

Market value and age together explain ~85% of salary variance. This aligns with football economics: clubs pay based on player value and career stage.

---

## Visualisation Interface

### Technology stack

- **Backend**: FastAPI (Python) — serves the benchmark API
- **Frontend**: React + TypeScript + Vite + Tailwind CSS + Recharts
- **Communication**: REST API with JSON, Vite dev proxy

### Pages

| Page | URL | Description |
|------|-----|-------------|
| Home | `/` | Hero + player search with autocomplete |
| Benchmark | `/benchmark/:playerId` | Full salary analysis for a pool player |
| **Custom Player** | `/manual` | **Enter any player manually** — position + age required; league, market value, and current salary optional. Supplying the current salary yields the OVERPAID / FAIRLY PAID / UNDERPAID verdict; omitting market value routes to the fallback model with an explicit warning |
| Comparables | `/comparables/:playerId` | Extended comparable players table |

The Custom Player form satisfies the "selected **or entered**" requirement end-to-end: dropdowns are populated from the live pool (`GET /api/players/options` — all positions and competitions), age is entered in years and validated (12.5–50), and the result view is identical to the pool-player benchmark (range chart, verdict, confidence, warnings, comparables). The same manual path is also available from the CLI (`python run_benchmark.py --main-position ... --age-months ...`) and the raw API (`POST /api/benchmark`).

### Benchmark page displays

- **Player card**: name, position, competition, age, market value
- **Salary range chart**: horizontal stacked bar with actual salary reference line
- **Status badge**: OVERPAID / FAIRLY_PAID / UNDERPAID / UNKNOWN
- **Confidence badge**: HIGH / MEDIUM / LOW
- **Fallback badge** (amber): shown whenever a fallback variant produced the estimate, stating exactly which features were missing (e.g. "Fallback model (no market value or position)")
- **Warning banner**: the `benchmark_warning` text (fallback notes, model-only estimates without peer validation)
- **Percentile**: player's salary rank among comparables
- **Top 5 comparable players**: with similarity scores

Search results flag incomplete players inline: "no market value — wider fallback estimate", "no market value, position — wider fallback estimate", or "no market value, position, age — cannot benchmark" for the 2 players no variant can serve.

---

## How to Run

### Prerequisites

- Python 3.13+
- Node.js 18+

### Setup

```bash
cd /Users/rudydesplan/Soccer

# ML environment (model training + inference + backend)
python3.13 -m venv .venv_ml
.venv_ml/bin/pip install -r requirements-ml.txt
# (torch/catboost/tabpfn/tabm are installed but excluded from AutoGluon via
#  excluded_model_types — see requirements-ml.txt)

# Frontend
cd soccer-benchmark/frontend
npm install
```

### Rebuild features (if data changes)

```bash
# Model training features
.venv_ml/bin/python build_model_features.py --input data_full.csv --output data_test.csv

# Benchmark player pool
.venv_ml/bin/python build_model_features.py --input data_full.csv --output player_pool.csv --mode pool

# Recalibrate
.venv_ml/bin/python -m salary_benchmark.calibration
```

### Run the interface

```bash
# Terminal 1: Backend (port 8001)
cd soccer-benchmark/backend
../../.venv_ml/bin/python -c "import uvicorn; uvicorn.run('main:app', host='127.0.0.1', port=8001)"

# Terminal 2: Frontend (port 5173)
cd soccer-benchmark/frontend
npm run dev
```

Open: http://localhost:5173

### CLI benchmark (no server needed)

```bash
.venv_ml/bin/python run_benchmark.py --player "Lamine Yamal"
.venv_ml/bin/python run_benchmark.py --player "Erling Haaland" --range wide
.venv_ml/bin/python run_benchmark.py --player "Gabriel" --json
```

### Retrain model

```bash
.venv_ml/bin/python train_autogluon_cpu.py --input data_test.csv --time 10800

# Fallback models (+ their calibrations): no_mv, no_mv_no_pos, no_mv_no_age
.venv_ml/bin/python train_fallback_no_mv.py --variant no_mv --input data_test.csv
.venv_ml/bin/python train_fallback_no_mv.py --variant no_mv_no_pos --input data_test.csv
.venv_ml/bin/python train_fallback_no_mv.py --variant no_mv_no_age --input data_test.csv
```

---

## Limitations

### Data limitations

1. **Severe league bias**: 75.9% of salary training data comes from top-5 leagues, but those leagues are only 14.8% of all players. Lower-division predictions are extrapolations, not interpolations.
2. **Near-zero coverage in lower leagues**: GB3 (3.2%), IT3A (0%), ES2 (0.7%) have almost no salary data. Benchmarks for these players are educated guesses, not validated estimates.
3. **Single season**: all data is 2025-2026. No historical trends or inflation adjustment.
4. **Capology dependency**: salary data relies on web scraping. If Capology changes its page structure, the parser breaks gracefully (returns nulls) but data is lost.
5. **No performance data**: goals, assists, minutes played, xG are not included. Salary is predicted purely from market/contract/demographic features.
6. **Market value semi-circularity**: Transfermarkt market values are partially influenced by salary/contract information. The model's R²=0.746 somewhat overstates predictive "intelligence" — a simpler 2-feature model (MV + age) achieves R²≈0.70.

### Model limitations

1. **Small training set**: 3,363 rows is modest for ML. The model generalizes well for top leagues but cannot be validated for lower leagues.
2. **MAPE of ~47%**: the model's average error (grouped holdout) is 47% of actual salary. Useful for ballpark orientation, not for contract precision.
3. **Superstar uncertainty**: players like Mbappé, Haaland, and Yamal have few true comparables. Confidence is correctly reported as LOW. The model underpredicts some top earners (e.g. Haaland: median €20.0M vs actual €31.5M). With the **wide** calibrated range he is **FAIRLY_PAID** (€10.5M–€37.5M); with the **normal** range he would be **OVERPAID**. Superstars may need segment-specific or wider calibration tails.
4. **Calibration coverage is nuanced**: out-of-fold grouped calibration targets ~50% / ~80% coverage, but measured coverage depends on scope:
   - **Grouped holdout** (honest): normal 50.5%, wide 77.2% — close to targets
   - **In-sample production model** (`evaluation_report.json`): optimistic (58.1% / 86.3%) because the production model saw all training rows
   Do not read in-sample coverage as validation.
5. **Static model**: the model does not update as new contracts are signed. Requires manual retraining.
6. **Two features dominate**: market_value and age account for ~85% of model signal. Other features (contract length, competition) contribute marginally. The fallback ablations confirm this empirically: removing market value drops holdout R² from 0.747 to 0.592, and additionally removing age collapses it to 0.314 (see [Fallback models](#fallback-models-players-with-missing-key-features)).
7. **Fairness — nationality is a model feature**: predictions are partly conditioned on `nationality`, so two otherwise identical players can receive different "expected" salaries because of their passport. Nationality is a protected characteristic in EU employment contexts; treat OVERPAID/UNDERPAID labels as market descriptions (the market itself may embed such biases), never as normative advice for individual contract decisions. The pool also contains ~687 players under 18, who receive benchmarks like any adult. A production deployment should measure and report nationality-conditional prediction gaps, and consider dropping the feature.

### Confidence limitations

1. **Forced LOW for leagues without salary data**: when no comparable player from the same league has a known salary, confidence is always LOW — we acknowledge we're extrapolating.
2. **HIGH confidence is rare**: requires 50+ comparables with avg similarity ≥0.75 AND narrow range. Few players meet this bar.

### Interface limitations

1. **No authentication**: the app is a local prototype, not production-secured.
2. **Bundle size**: Recharts adds ~600KB to the frontend bundle (acceptable for a prototype).
3. **No caching**: each benchmark request recomputes from scratch (fast enough for demo, but not optimized for scale).
4. **Player IDs are not stable**: the `id` used in URLs (`/benchmark/123`) is the player's row position in `player_pool.csv`. Rebuilding the pool can silently rebind an ID to a different player — do not persist or share benchmark links across pool rebuilds. A production version would key players by `transfermarkt_player_id` + club.
5. **Players with missing key features get weaker fallback estimates**: 924 of 19,476 pool players have no market value; they are benchmarked with the `no_mv` fallback (wider range, confidence capped at MEDIUM). Players also missing their position (51) or age (9) route to the `no_mv_no_pos` / `no_mv_no_age` variants (much wider ranges, always LOW confidence). The search UI labels each with exactly what is missing. Only the 2 players missing position **and** age remain unbenchmarkable, refused with a clear error.
6. **Most pool players show verdict UNKNOWN**: 16,113 of 19,476 pool players have no Capology salary, so the OVERPAID / FAIRLY PAID / UNDERPAID verdict cannot be computed for them — the range and comparables still are. A club that knows its player's real salary can get the verdict by entering the player (with salary) on the **Custom Player** page (`/manual`).
7. **Manual entry requires position and age**: the Custom Player form deliberately refuses to guess missing inputs typed by a human (unlike pool players, where missing data is a property of the dataset and the fallback cascade applies). Market value and salary stay optional.

---

## What I Would Do Differently With More Time

### Data (highest impact)

- **License salary data commercially**: replace scraping with a licensed feed (Capology's data services, Sportradar, league-official disclosures) — removes the terms-of-use risk documented in [Legality](#legality-and-terms-of-use-considerations) and adds contractual data-quality guarantees.
- **Add performance features**: goals, assists, minutes played, xG from FBref/StatsBomb. This would likely cut MAPE from ~47% substantially and reduce market-value dependency.
- **Multi-season data**: track salary evolution over 3-5 seasons to model inflation and career trajectories.
- **Lower-league salary sources**: supplement Capology with league-specific sources (LFP for Ligue 2, EFL disclosures for Championship/League One) to fix the severe league bias.
- **Transfer fee as feature**: the fee paid for a player strongly correlates with salary expectations and is available from Transfermarkt.

### Model (calibration already fixed)

- ~~Out-of-fold calibration~~ ✅ **Done** — calibration uses 5-fold grouped out-of-fold residuals (`GroupKFold` by `transfermarkt_player_id`).
- **Segment-specific calibration**: compute separate residual distributions per league/position. The model underpredicts in Italy and overpredicts in France.
- **Bayesian approach**: model salary as a distribution rather than a point estimate, giving natural uncertainty quantification without residual calibration.
- **Simpler baseline comparison**: explicitly report how much AutoGluon beats a 2-feature Ridge(MV+age) to justify the complexity.

### Pipeline

- **Containerization**: Docker Compose for reproducible deployment (backend + frontend + model).
- **CI/CD**: automated retraining pipeline triggered by new data.
- **Automated data freshness check**: alert when Capology structure changes or match rate drops.

### Monitoring (current scope: prediction logging)

- **Temporal drift scaffold**: `salary_benchmark/monitoring.py` compares the first vs second half of recent log-scale predictions (PSI + KS). Each log entry also records the input features and the `model_used` variant, so drift can later be analysed per feature and per model variant (full vs fallbacks). Note: this split-by-time check only makes sense if predictions arrive in an order unrelated to player attributes — the reference `models/monitoring_report.json` is generated from a randomly-shuffled sample for exactly that reason (logging players in file order produced a false PSI alert from league blocks).
- **Real drift detection** (future): store training-time prediction distribution and compare production predictions against that baseline.
- **Online A/B testing**: serve two models simultaneously and compare real user outcomes.
- **Observability stack**: Prometheus/Grafana for latency, error rates, prediction distributions.

---

## Project Structure

```
Soccer/
├── README.md                          # This file
├── data (1).csv                       # SoccerSolver input (19,476 players)
├── data_full.csv                      # Enriched dataset (24 columns)
├── data_test.csv                      # Model-ready features (16 columns)
├── player_pool.csv                    # Benchmark pool (35 columns)
├── capology_pipeline/                 # Data acquisition pipeline
├── build_model_features.py            # Feature engineering script
├── schemas.py                         # Pydantic validation schemas
├── train_autogluon_cpu.py             # AutoGluon training script
├── train_fallback_no_mv.py            # Fallback model variants + calibrations (--variant)
├── train_automl_cpu.py                # FLAML training script
├── train_pycaret.py                   # PyCaret training script
├── model_evaluation.py                # Rigorous model evaluation (CV + segments)
├── model_comparison.py                # Cross-validated model comparison
├── salary_benchmark/                  # Benchmark engine package
│   ├── model.py                       # AutoGluon inference
│   ├── calibration.py                 # Out-of-fold residual calibration
│   ├── comparables.py                 # 3-level comparable search
│   ├── confidence.py                  # Confidence + status logic
│   ├── benchmark.py                   # Main orchestrator
│   └── monitoring.py                  # Prediction logging + drift scaffold
├── run_benchmark.py                   # CLI benchmark tool
├── models/                            # Trained model artifacts
│   ├── autogluon/                     # Production model
│   ├── autogluon_no_mv/               # Fallback model (no market value)
│   ├── autogluon_no_mv_no_pos/        # Fallback model (no MV, no position)
│   ├── autogluon_no_mv_no_age/        # Fallback model (no MV, no age)
│   ├── calibration.json               # Out-of-fold residual percentiles
│   ├── calibration_no_mv.json         # Fallback residual percentiles (wider)
│   ├── calibration_no_mv_no_pos.json  # No-position fallback percentiles
│   ├── calibration_no_mv_no_age.json  # No-age fallback percentiles (widest)
│   ├── fallback_*_results.json        # Per-variant holdout metrics vs full model
│   ├── evaluation_report.json         # Rigorous evaluation results
│   └── plots/                         # Diagnostic plots
├── tests/                             # 400+ tests (unit + integration + data)
├── logs/                              # Prediction logs (monitoring)
└── soccer-benchmark/                  # Web interface
    ├── backend/                       # FastAPI
    └── frontend/                      # React + Vite + Tailwind
```
