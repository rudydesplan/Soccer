import type {
  BenchmarkOptions,
  BenchmarkResult,
  ComparablePlayer,
  ExplanationResult,
  Player,
} from '../lib/api';

export const players: Player[] = [
  {
    id: 0,
    player_name: 'Lionel Messi',
    main_position: 'Right Winger',
    team_name: 'Inter Miami',
    competition_id: 'MLS1',
    competition_country: 'United States',
    nationality: 'Argentina',
    age_months: 440,
    market_value_current_eur: 20_000_000,
    annual_fixed_eur: 15_000_000,
  },
  {
    id: 2,
    player_name: 'Erling Haaland',
    main_position: 'Centre-Forward',
    team_name: 'Manchester City',
    competition_id: 'GB1',
    competition_country: 'England',
    nationality: 'Norway',
    age_months: 290,
    market_value_current_eur: 170_000_000,
    annual_fixed_eur: 20_000_000,
  },
];

export const comparables: ComparablePlayer[] = [
  {
    id: 10,
    player_name: 'Kylian Mbappe',
    main_position: 'Centre-Forward',
    competition_id: 'ES1',
    competition_country: 'Spain',
    age_months: 306,
    market_value_current_eur: 180_000_000,
    annual_fixed_eur: 25_000_000,
    similarity_score: 0.93,
  },
  {
    id: 11,
    player_name: 'Harry Kane',
    main_position: 'Centre-Forward',
    competition_id: 'L1',
    competition_country: 'Germany',
    age_months: 370,
    market_value_current_eur: 90_000_000,
    annual_fixed_eur: 24_000_000,
    similarity_score: 0.88,
  },
];

export const benchmarkResult: BenchmarkResult = {
  player_id: 2,
  player_name: 'Erling Haaland',
  main_position: 'Centre-Forward',
  competition_id: 'GB1',
  competition_country: 'England',
  age_months: 290,
  market_value_current_eur: 170_000_000,
  expected_salary_low_eur: 14_000_000,
  expected_salary_median_eur: 20_000_000,
  expected_salary_high_eur: 28_000_000,
  actual_salary_eur: 31_500_000,
  salary_percentile: 95,
  salary_status: 'OVERPAID',
  benchmark_confidence: 'LOW',
  benchmark_n_comparables: 12,
  benchmark_n_comparables_with_salary: 8,
  benchmark_avg_similarity: 0.81,
  comparable_level_used: 1,
  range_width_used: 'normal',
  model_used: 'full',
  benchmark_warning: null,
  comparable_players: comparables,
};

export const explanationResult: ExplanationResult = {
  player_name: 'Erling Haaland',
  model_used: 'full',
  base_salary_eur: 366_718,
  predicted_salary_eur: 20_027_641,
  features: [
    {
      feature: 'log_market_value_current_eur',
      label: 'Market value',
      value: '€170.0M',
      shap_log: 2.753,
      pct_effect: 1467.1,
    },
    {
      feature: 'age_months',
      label: 'Age',
      value: '24.2 years',
      shap_log: 0.289,
      pct_effect: 33.5,
    },
    {
      feature: 'competition_id',
      label: 'League',
      value: 'GB1',
      shap_log: 0.318,
      pct_effect: 37.4,
    },
    {
      feature: 'nationality',
      label: 'Nationality',
      value: 'Norway',
      shap_log: -0.12,
      pct_effect: -11.3,
    },
  ],
};

export const benchmarkOptions: BenchmarkOptions = {
  positions: ['Centre-Forward', 'Goalkeeper', 'Right Winger'],
  competitions: [
    { id: 'GB1', name: 'Premier League', country: 'England' },
    { id: 'ES1', name: 'LaLiga', country: 'Spain' },
  ],
};
