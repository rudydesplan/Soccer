import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getBenchmarkById, type BenchmarkResult, formatEur } from '../lib/api';
import PlayerCard from '../components/PlayerCard';
import SalaryRangeChart from '../components/SalaryRangeChart';
import ConfidenceBadge from '../components/ConfidenceBadge';
import ComparablesTable from '../components/ComparablesTable';

export default function Benchmark() {
  const { playerId } = useParams<{ playerId: string }>();
  const [result, setResult] = useState<BenchmarkResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [rangeWidth, setRangeWidth] = useState<'normal' | 'wide'>('normal');

  useEffect(() => {
    const id = Number(playerId);
    if (!Number.isInteger(id)) {
      setError('Invalid player id');
      setLoading(false);
      return;
    }
    setLoading(true);
    setError(null);
    getBenchmarkById(id, rangeWidth)
      .then(setResult)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [playerId, rangeWidth]);

  if (loading) {
    return (
      <div className="max-w-5xl mx-auto px-4 py-12 text-center">
        <div className="animate-pulse">
          <div className="text-xl text-gray-500">Loading benchmark...</div>
        </div>
      </div>
    );
  }

  if (error || !result) {
    return (
      <div className="max-w-5xl mx-auto px-4 py-12 text-center">
        <div className="bg-red-50 border border-red-200 rounded-lg p-6">
          <p className="text-red-700">{error || 'Failed to load benchmark'}</p>
          <Link to="/" className="text-blue-600 hover:underline mt-4 inline-block">
            ← Back to search
          </Link>
        </div>
      </div>
    );
  }

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

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-6">
      <Link to="/" className="text-blue-600 hover:underline text-sm">
        ← Back to search
      </Link>

      <PlayerCard
        name={result.player_name}
        position={result.main_position || 'N/A'}
        competition={result.competition_id || 'N/A'}
        age_months={result.age_months}
        market_value={result.market_value_current_eur}
      />

      {/* Warning banner */}
      {result.benchmark_warning && (
        <div className="bg-amber-50 border border-amber-300 rounded-lg p-4 flex items-start gap-3">
          <span className="text-amber-500 text-xl leading-none" aria-hidden="true">⚠</span>
          <div>
            <p className="font-semibold text-amber-800">Limited reliability</p>
            <p className="text-sm text-amber-700">{result.benchmark_warning}</p>
          </div>
        </div>
      )}

      {/* Status & Confidence */}
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
          <div className="text-sm text-gray-600">
            Avg Similarity: <span className="font-mono font-medium">{(result.benchmark_avg_similarity * 100).toFixed(1)}%</span>
          </div>
        </div>
      </div>

      {/* Salary Chart */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex justify-between items-center mb-2">
          <h3 className="text-lg font-semibold text-gray-900">Salary Range</h3>
          <div className="flex gap-1 bg-gray-100 rounded-lg p-1">
            <button
              onClick={() => setRangeWidth('normal')}
              className={`px-3 py-1 text-sm rounded-md transition-colors ${rangeWidth === 'normal' ? 'bg-white shadow font-medium' : 'text-gray-600 hover:text-gray-900'}`}
            >
              Normal (50%)
            </button>
            <button
              onClick={() => setRangeWidth('wide')}
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

      {/* Top Comparables */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Top 5 Comparable Players</h3>
          <Link
            to={`/comparables/${result.player_id}`}
            className="text-blue-600 hover:underline text-sm"
          >
            View all comparables →
          </Link>
        </div>
        <ComparablesTable players={result.comparable_players.slice(0, 5)} />
      </div>
    </div>
  );
}
