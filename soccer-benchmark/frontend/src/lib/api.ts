export interface Player {
  id: number;
  player_name: string;
  main_position: string | null;
  team_name: string | null;
  competition_id: string | null;
  competition_country: string | null;
  nationality: string | null;
  age_months: number | null;
  market_value_current_eur: number | null;
  annual_fixed_eur: number | null;
}

export interface ComparablePlayer {
  id?: number;
  player_name: string;
  main_position: string | null;
  competition_id: string | null;
  competition_country: string | null;
  age_months: number | null;
  market_value_current_eur: number | null;
  annual_fixed_eur: number | null;
  similarity_score: number;
}

export interface BenchmarkResult {
  player_id: number | null;
  player_name: string;
  main_position: string | null;
  competition_id: string | null;
  competition_country: string | null;
  age_months: number | null;
  market_value_current_eur: number | null;
  expected_salary_low_eur: number;
  expected_salary_median_eur: number;
  expected_salary_high_eur: number;
  actual_salary_eur: number | null;
  salary_percentile: number | null;
  salary_status: string;
  benchmark_confidence: string;
  benchmark_n_comparables: number;
  benchmark_n_comparables_with_salary: number | null;
  benchmark_avg_similarity: number;
  comparable_level_used: number;
  range_width_used: 'normal' | 'wide';
  model_used: 'full' | 'no_mv' | 'no_mv_no_pos' | 'no_mv_no_age';
  benchmark_warning: string | null;
  comparable_players: ComparablePlayer[];
}

export interface FeatureContribution {
  feature: string;
  label: string;
  value: string | null;
  shap_log: number;
  pct_effect: number;
}

export interface ExplanationResult {
  player_name: string;
  model_used: 'full' | 'no_mv' | 'no_mv_no_pos' | 'no_mv_no_age';
  base_salary_eur: number;
  predicted_salary_eur: number;
  features: FeatureContribution[];
}

export interface ModelCardMetrics {
  split: string;
  n_train: number;
  n_test: number;
  r2: number;
  rmse_log: number;
  mae_log: number;
  median_ape_pct: number;
  within_20_pct: number;
  within_50_pct: number;
}

export interface ModelCardCalibration {
  method: string;
  n_folds: number;
  n_samples: number;
  residual_p10: number;
  residual_p25: number;
  residual_p50: number;
  residual_p75: number;
  residual_p90: number;
}

export interface ModelCardFeature {
  feature: string;
  label: string;
  importance: number;
}

export interface ModelCardCoverage {
  n_rows: number;
  n_players: number;
  n_with_salary: number;
  n_with_market_value: number;
  countries: string[];
  n_leagues: number;
  seasons: number[];
  n_positions: number;
}

export interface ModelCard {
  model_name: string;
  framework: string;
  framework_version: string | null;
  trained_at: string | null;
  target: string;
  variants: Record<string, boolean>;
  metrics: ModelCardMetrics;
  calibration: ModelCardCalibration;
  top_features: ModelCardFeature[];
  coverage: ModelCardCoverage;
}

export interface CompetitionOption {
  id: string;
  name: string;
  country: string | null;
}

export interface BenchmarkOptions {
  positions: string[];
  competitions: CompetitionOption[];
}

export interface ManualPlayerInput {
  main_position: string;
  age_months: number;
  competition_id?: string;
  competition_country?: string;
  market_value_current_eur?: number;
  annual_fixed_eur?: number;
}

async function errorFromResponse(res: Response, fallback: string): Promise<Error> {
  try {
    const body = await res.json();
    if (typeof body?.detail === 'string') return new Error(body.detail);
  } catch {
    // Non-JSON error body — fall through to the generic message.
  }
  return new Error(`${fallback} (HTTP ${res.status})`);
}

export async function searchPlayers(query: string, limit = 20): Promise<Player[]> {
  const res = await fetch(`/api/players/search?q=${encodeURIComponent(query)}&limit=${limit}`);
  if (!res.ok) throw await errorFromResponse(res, 'Search failed');
  return res.json();
}

export async function getPlayer(id: number): Promise<Player> {
  const res = await fetch(`/api/players/${id}`);
  if (!res.ok) throw await errorFromResponse(res, 'Player not found');
  return res.json();
}

export async function getBenchmarkById(playerId: number, rangeWidth: 'normal' | 'wide' = 'normal', fullComparables = false): Promise<BenchmarkResult> {
  const res = await fetch('/api/benchmark', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ player_id: playerId, range_width: rangeWidth, full_comparables: fullComparables }),
  });
  if (!res.ok) throw await errorFromResponse(res, 'Benchmark failed');
  return res.json();
}

export async function getBenchmarkOptions(): Promise<BenchmarkOptions> {
  const res = await fetch('/api/players/options');
  if (!res.ok) throw await errorFromResponse(res, 'Failed to load form options');
  return res.json();
}

export async function getManualBenchmark(input: ManualPlayerInput, rangeWidth: 'normal' | 'wide' = 'normal'): Promise<BenchmarkResult> {
  const res = await fetch('/api/benchmark', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...input, range_width: rangeWidth }),
  });
  if (!res.ok) throw await errorFromResponse(res, 'Benchmark failed');
  return res.json();
}

export async function getExplanationById(playerId: number): Promise<ExplanationResult> {
  const res = await fetch('/api/benchmark/explain', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ player_id: playerId }),
  });
  if (!res.ok) throw await errorFromResponse(res, 'Explanation failed');
  return res.json();
}

export async function getManualExplanation(input: ManualPlayerInput): Promise<ExplanationResult> {
  const res = await fetch('/api/benchmark/explain', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  });
  if (!res.ok) throw await errorFromResponse(res, 'Explanation failed');
  return res.json();
}

export async function getModelCard(): Promise<ModelCard> {
  const res = await fetch('/api/meta/model-card');
  if (!res.ok) throw await errorFromResponse(res, 'Failed to load model card');
  return res.json();
}

export function formatEur(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return 'N/A';
  if (value >= 1_000_000) {
    return `€${(value / 1_000_000).toFixed(1)}M`;
  }
  if (value >= 1_000) {
    return `€${(value / 1_000).toFixed(0)}K`;
  }
  return `€${value.toFixed(0)}`;
}

export function formatAge(months: number | null | undefined): string {
  if (months === null || months === undefined || Number.isNaN(months)) return 'N/A';
  const years = (months / 12).toFixed(1);
  return `${months.toFixed(0)} months (${years} yrs)`;
}
