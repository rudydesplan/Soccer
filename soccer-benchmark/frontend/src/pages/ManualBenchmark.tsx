import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  getBenchmarkOptions,
  getManualBenchmark,
  formatEur,
  type BenchmarkOptions,
  type BenchmarkResult,
  type ManualPlayerInput,
} from '../lib/api';
import SalaryRangeChart from '../components/SalaryRangeChart';
import ConfidenceBadge from '../components/ConfidenceBadge';
import ComparablesTable from '../components/ComparablesTable';

const statusColors: Record<string, string> = {
  OVERPAID: 'text-red-600 bg-red-50 border-red-200',
  UNDERPAID: 'text-yellow-700 bg-yellow-50 border-yellow-200',
  FAIRLY_PAID: 'text-green-700 bg-green-50 border-green-200',
  UNKNOWN: 'text-gray-700 bg-gray-50 border-gray-200',
};

const statusLabels: Record<string, string> = {
  OVERPAID: '⬆️ Overpaid',
  UNDERPAID: '⬇️ Underpaid',
  FAIRLY_PAID: '✅ Fairly Paid',
  UNKNOWN: 'Unknown salary',
};

export default function ManualBenchmark() {
  const [options, setOptions] = useState<BenchmarkOptions | null>(null);
  const [optionsError, setOptionsError] = useState<string | null>(null);

  const [position, setPosition] = useState('');
  const [competitionId, setCompetitionId] = useState('');
  const [ageYears, setAgeYears] = useState('');
  const [marketValue, setMarketValue] = useState('');
  const [actualSalary, setActualSalary] = useState('');

  const [rangeWidth, setRangeWidth] = useState<'normal' | 'wide'>('normal');
  const [result, setResult] = useState<BenchmarkResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Keep the submitted input so the range-width toggle can re-run the same query.
  const [lastInput, setLastInput] = useState<ManualPlayerInput | null>(null);

  useEffect(() => {
    getBenchmarkOptions()
      .then(setOptions)
      .catch((err) => setOptionsError(err.message));
  }, []);

  function buildInput(): ManualPlayerInput | string {
    if (!position) return 'Select a position.';
    const age = parseFloat(ageYears);
    if (!Number.isFinite(age)) return 'Enter the player\u2019s age in years.';
    if (age < 12.5 || age > 50) return 'Age must be between 12.5 and 50 years.';
    const input: ManualPlayerInput = {
      main_position: position,
      age_months: Math.round(age * 12),
    };
    if (competitionId && options) {
      const comp = options.competitions.find((c) => c.id === competitionId);
      input.competition_id = competitionId;
      if (comp?.country) input.competition_country = comp.country;
    }
    if (marketValue !== '') {
      const mv = parseFloat(marketValue);
      if (!Number.isFinite(mv) || mv < 0) return 'Market value must be a positive number (in EUR).';
      input.market_value_current_eur = mv;
    }
    if (actualSalary !== '') {
      const sal = parseFloat(actualSalary);
      if (!Number.isFinite(sal) || sal < 0) return 'Current salary must be a positive number (in EUR).';
      input.annual_fixed_eur = sal;
    }
    return input;
  }

  async function run(input: ManualPlayerInput, width: 'normal' | 'wide') {
    setLoading(true);
    setError(null);
    try {
      const res = await getManualBenchmark(input, width);
      setResult(res);
      setLastInput(input);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Benchmark failed');
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const input = buildInput();
    if (typeof input === 'string') {
      setError(input);
      setResult(null);
      return;
    }
    run(input, rangeWidth);
  }

  function handleRangeWidth(width: 'normal' | 'wide') {
    setRangeWidth(width);
    if (lastInput) run(lastInput, width);
  }

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-6">
      <Link to="/" className="text-blue-600 hover:underline text-sm">
        ← Back to search
      </Link>

      <div>
        <h1 className="text-3xl font-bold text-gray-900 mb-1">Benchmark a Custom Player</h1>
        <p className="text-gray-600">
          Enter a player&apos;s profile — including one not in our database — and get an expected
          salary range. Add their current salary to see an overpaid / fairly paid / underpaid verdict.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="bg-white rounded-lg shadow-md p-6">
        {optionsError && (
          <div className="mb-4 bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700">
            {optionsError}
          </div>
        )}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <label className="block">
            <span className="text-sm font-medium text-gray-700">Position *</span>
            <select
              value={position}
              onChange={(e) => setPosition(e.target.value)}
              required
              className="mt-1 w-full px-3 py-2 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
            >
              <option value="">Select a position…</option>
              {options?.positions.map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
          </label>

          <label className="block">
            <span className="text-sm font-medium text-gray-700">Age (years) *</span>
            <input
              type="number"
              step="0.5"
              min="12.5"
              max="50"
              value={ageYears}
              onChange={(e) => setAgeYears(e.target.value)}
              required
              placeholder="e.g. 24"
              className="mt-1 w-full px-3 py-2 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </label>

          <label className="block">
            <span className="text-sm font-medium text-gray-700">League</span>
            <select
              value={competitionId}
              onChange={(e) => setCompetitionId(e.target.value)}
              className="mt-1 w-full px-3 py-2 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
            >
              <option value="">Not specified</option>
              {options?.competitions.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name}{c.country ? ` (${c.country})` : ''}
                </option>
              ))}
            </select>
          </label>

          <label className="block">
            <span className="text-sm font-medium text-gray-700">Market value (EUR)</span>
            <input
              type="number"
              min="0"
              step="100000"
              value={marketValue}
              onChange={(e) => setMarketValue(e.target.value)}
              placeholder="e.g. 25000000 — leave empty if unknown"
              className="mt-1 w-full px-3 py-2 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <span className="text-xs text-gray-500">
              Without it, a weaker fallback model is used and the range gets wider.
            </span>
          </label>

          <label className="block md:col-span-2">
            <span className="text-sm font-medium text-gray-700">Current annual salary (EUR, gross fixed)</span>
            <input
              type="number"
              min="0"
              step="100000"
              value={actualSalary}
              onChange={(e) => setActualSalary(e.target.value)}
              placeholder="e.g. 4000000 — needed for the overpaid / underpaid verdict"
              className="mt-1 w-full px-3 py-2 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </label>
        </div>

        <button
          type="submit"
          disabled={loading}
          className="mt-5 px-6 py-2.5 rounded-lg bg-blue-600 text-white font-medium hover:bg-blue-700 disabled:opacity-50 transition-colors"
        >
          {loading ? 'Running…' : 'Run benchmark'}
        </button>
      </form>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
          {error}
        </div>
      )}

      {result && (
        <>
          {result.benchmark_warning && (
            <div className="bg-amber-50 border border-amber-300 rounded-lg p-4 flex items-start gap-3">
              <span className="text-amber-500 text-xl leading-none" aria-hidden="true">⚠</span>
              <div>
                <p className="font-semibold text-amber-800">Limited reliability</p>
                <p className="text-sm text-amber-700">{result.benchmark_warning}</p>
              </div>
            </div>
          )}

          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex flex-wrap items-center gap-4">
              <div className={`px-4 py-2 rounded-lg border font-semibold ${statusColors[result.salary_status] || ''}`}>
                {statusLabels[result.salary_status] || result.salary_status}
              </div>
              <ConfidenceBadge confidence={result.benchmark_confidence} />
              {result.model_used !== 'full' && (
                <div className="px-3 py-1.5 rounded-lg border border-amber-200 bg-amber-50 text-amber-800 text-xs font-medium">
                  {result.model_used === 'no_mv' && 'Fallback model (no market value)'}
                  {result.model_used === 'no_mv_no_pos' && 'Fallback model (no market value or position)'}
                  {result.model_used === 'no_mv_no_age' && 'Fallback model (no market value or age)'}
                </div>
              )}
              <div className="text-sm text-gray-600">
                Percentile: <span className="font-mono font-medium">{result.salary_percentile === null ? 'N/A' : `${result.salary_percentile.toFixed(0)}%`}</span>
              </div>
              <div className="text-sm text-gray-600">
                Comparables: <span className="font-mono font-medium">{result.benchmark_n_comparables}</span>
                {result.benchmark_n_comparables_with_salary !== null && (
                  <span className="text-gray-400"> ({result.benchmark_n_comparables_with_salary} with known salary)</span>
                )}
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex justify-between items-center mb-2">
              <h3 className="text-lg font-semibold text-gray-900">Salary Range</h3>
              <div className="flex gap-1 bg-gray-100 rounded-lg p-1">
                <button
                  onClick={() => handleRangeWidth('normal')}
                  className={`px-3 py-1 text-sm rounded-md transition-colors ${rangeWidth === 'normal' ? 'bg-white shadow font-medium' : 'text-gray-600 hover:text-gray-900'}`}
                >
                  Normal (50%)
                </button>
                <button
                  onClick={() => handleRangeWidth('wide')}
                  className={`px-3 py-1 text-sm rounded-md transition-colors ${rangeWidth === 'wide' ? 'bg-white shadow font-medium' : 'text-gray-600 hover:text-gray-900'}`}
                >
                  Wide (80%)
                </button>
              </div>
            </div>
            <p className="text-sm text-gray-500 mb-4">
              Actual salary: <span className="font-semibold">{formatEur(result.actual_salary_eur)}</span> vs expected range
            </p>
            <SalaryRangeChart
              low={result.expected_salary_low_eur}
              median={result.expected_salary_median_eur}
              high={result.expected_salary_high_eur}
              actual={result.actual_salary_eur}
              status={result.salary_status}
            />
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Top 5 Comparable Players</h3>
            <ComparablesTable players={result.comparable_players.slice(0, 5)} />
          </div>
        </>
      )}
    </div>
  );
}
