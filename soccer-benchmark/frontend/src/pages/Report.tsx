import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  getBenchmarkById,
  getExplanationById,
  type BenchmarkResult,
  type ExplanationResult,
  formatEur,
  formatAge,
} from '../lib/api';
import { formatEffect } from '../components/ExplanationPanel';

const STATUS_LABELS: Record<string, string> = {
  OVERPAID: 'Overpaid',
  UNDERPAID: 'Underpaid',
  FAIRLY_PAID: 'Fairly paid',
  UNKNOWN: 'Unknown (no salary on record)',
};

const MODEL_LABELS: Record<string, string> = {
  full: 'Full model',
  no_mv: 'Fallback model (no market value)',
  no_mv_no_pos: 'Fallback model (no market value or position)',
  no_mv_no_age: 'Fallback model (no market value or age)',
};

const MAX_REPORT_FEATURES = 8;
const MAX_REPORT_COMPARABLES = 10;

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="mb-6 break-inside-avoid">
      <h2 className="text-lg font-semibold text-gray-900 border-b border-gray-300 pb-1 mb-3">
        {title}
      </h2>
      {children}
    </section>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs uppercase tracking-wide text-gray-500">{label}</div>
      <div className="font-medium text-gray-900">{value}</div>
    </div>
  );
}

export default function Report() {
  const { playerId } = useParams<{ playerId: string }>();
  const [result, setResult] = useState<BenchmarkResult | null>(null);
  const [explanation, setExplanation] = useState<ExplanationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const id = Number(playerId);
    if (!Number.isInteger(id)) {
      setError('Invalid player id');
      setLoading(false);
      return;
    }
    setLoading(true);
    setError(null);
    // The explanation is part of the report: fail the whole page rather than
    // print a report with a silently missing section.
    Promise.all([getBenchmarkById(id), getExplanationById(id)])
      .then(([benchmark, expl]) => {
        setResult(benchmark);
        setExplanation(expl);
      })
      .catch((err) => setError(err instanceof Error ? err.message : 'Failed to load report'))
      .finally(() => setLoading(false));
  }, [playerId]);

  if (loading) {
    return (
      <div className="max-w-3xl mx-auto px-4 py-12 text-center text-gray-500" data-testid="report-loading">
        Preparing report… (running benchmark and explanation)
      </div>
    );
  }

  if (error || !result || !explanation) {
    return (
      <div className="max-w-3xl mx-auto px-4 py-12 text-center">
        <div className="bg-red-50 border border-red-200 rounded-lg p-6">
          <p className="text-red-700">{error || 'Failed to load report'}</p>
          <Link to="/" className="text-blue-600 hover:underline mt-4 inline-block">
            ← Back to search
          </Link>
        </div>
      </div>
    );
  }

  const generatedOn = new Date().toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });

  return (
    <div className="max-w-3xl mx-auto px-4 py-8">
      {/* Screen-only toolbar */}
      <div className="flex justify-between items-center mb-6 print:hidden">
        <Link to={`/benchmark/${result.player_id}`} className="text-blue-600 hover:underline text-sm">
          ← Back to benchmark
        </Link>
        <button
          onClick={() => window.print()}
          className="px-4 py-2 rounded-lg bg-blue-600 text-white text-sm font-medium hover:bg-blue-700 transition-colors"
        >
          Print / Save as PDF
        </button>
      </div>

      {/* Report document */}
      <div className="bg-white rounded-lg shadow-md p-8 print:shadow-none print:p-0">
        <header className="mb-6 border-b-2 border-gray-900 pb-4">
          <h1 className="text-2xl font-bold text-gray-900">Salary Benchmark Report</h1>
          <div className="flex justify-between text-sm text-gray-600 mt-1">
            <span>{result.player_name}</span>
            <span>Generated {generatedOn}</span>
          </div>
        </header>

        <Section title="Player">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <Field label="Position" value={result.main_position || 'N/A'} />
            <Field label="League" value={result.competition_id || 'N/A'} />
            <Field label="Age" value={formatAge(result.age_months)} />
            <Field label="Market value" value={formatEur(result.market_value_current_eur)} />
          </div>
        </Section>

        <Section title="Benchmark result">
          <div className="grid grid-cols-3 gap-4 mb-4">
            <div className="bg-gray-50 rounded-lg p-4 text-center print:border print:border-gray-300">
              <div className="text-xs uppercase tracking-wide text-gray-500">Low</div>
              <div className="text-xl font-bold text-gray-900">{formatEur(result.expected_salary_low_eur)}</div>
            </div>
            <div className="bg-blue-50 rounded-lg p-4 text-center print:border print:border-gray-400">
              <div className="text-xs uppercase tracking-wide text-blue-700">Expected (median)</div>
              <div className="text-xl font-bold text-blue-900">{formatEur(result.expected_salary_median_eur)}</div>
            </div>
            <div className="bg-gray-50 rounded-lg p-4 text-center print:border print:border-gray-300">
              <div className="text-xs uppercase tracking-wide text-gray-500">High</div>
              <div className="text-xl font-bold text-gray-900">{formatEur(result.expected_salary_high_eur)}</div>
            </div>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <Field label="Actual salary" value={formatEur(result.actual_salary_eur)} />
            <Field
              label="Market position"
              value={STATUS_LABELS[result.salary_status] || result.salary_status}
            />
            <Field
              label="Percentile"
              value={result.salary_percentile === null ? 'N/A' : `${result.salary_percentile.toFixed(0)}%`}
            />
            <Field label="Confidence" value={result.benchmark_confidence} />
          </div>
          <p className="text-xs text-gray-500 mt-3">
            Range: {result.range_width_used === 'wide' ? 'wide (80% of outcomes)' : 'normal (50% of outcomes)'} ·{' '}
            {MODEL_LABELS[result.model_used] || result.model_used} ·{' '}
            {result.benchmark_n_comparables} comparables
            {result.benchmark_n_comparables_with_salary !== null
              ? ` (${result.benchmark_n_comparables_with_salary} with known salary)`
              : ''}
            , avg similarity {(result.benchmark_avg_similarity * 100).toFixed(0)}%
          </p>
          {result.benchmark_warning && (
            <div
              className="mt-3 bg-amber-50 border border-amber-300 rounded-lg p-3 text-sm text-amber-800 print:bg-white"
              data-testid="report-warning"
            >
              <strong>Limited reliability:</strong> {result.benchmark_warning}
            </div>
          )}
        </Section>

        <Section title="Why this estimate">
          <p className="text-sm text-gray-600 mb-3">
            A typical player&apos;s estimate is {formatEur(explanation.base_salary_eur)}; the factors
            below move this player to {formatEur(explanation.predicted_salary_eur)} (raw model
            estimate — the calibrated median above can differ slightly).
          </p>
          <table className="min-w-full text-sm" data-testid="report-explanation">
            <thead>
              <tr className="text-left text-xs uppercase tracking-wide text-gray-500 border-b border-gray-300">
                <th className="py-1 pr-4">Factor</th>
                <th className="py-1 pr-4">Player value</th>
                <th className="py-1 text-right">Effect on estimate</th>
              </tr>
            </thead>
            <tbody>
              {explanation.features.slice(0, MAX_REPORT_FEATURES).map((f) => (
                <tr key={f.feature} className="border-b border-gray-100">
                  <td className="py-1.5 pr-4 text-gray-900">{f.label}</td>
                  <td className="py-1.5 pr-4 text-gray-600">{f.value ?? '—'}</td>
                  <td
                    className={`py-1.5 text-right font-mono ${f.shap_log >= 0 ? 'text-emerald-700' : 'text-red-600'}`}
                  >
                    {formatEffect(f.pct_effect)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Section>

        <Section title="Comparable players">
          <table className="min-w-full text-sm" data-testid="report-comparables">
            <thead>
              <tr className="text-left text-xs uppercase tracking-wide text-gray-500 border-b border-gray-300">
                <th className="py-1 pr-3">Name</th>
                <th className="py-1 pr-3">Position</th>
                <th className="py-1 pr-3">League</th>
                <th className="py-1 pr-3">Age</th>
                <th className="py-1 pr-3 text-right">Market value</th>
                <th className="py-1 pr-3 text-right">Salary</th>
                <th className="py-1 text-right">Similarity</th>
              </tr>
            </thead>
            <tbody>
              {result.comparable_players.slice(0, MAX_REPORT_COMPARABLES).map((p, idx) => (
                <tr key={p.id ?? `${p.player_name}-${idx}`} className="border-b border-gray-100">
                  <td className="py-1.5 pr-3 font-medium text-gray-900">{p.player_name}</td>
                  <td className="py-1.5 pr-3 text-gray-600">{p.main_position}</td>
                  <td className="py-1.5 pr-3 text-gray-600">{p.competition_id}</td>
                  <td className="py-1.5 pr-3 text-gray-600">{formatAge(p.age_months)}</td>
                  <td className="py-1.5 pr-3 text-right text-gray-600">{formatEur(p.market_value_current_eur)}</td>
                  <td className="py-1.5 pr-3 text-right text-gray-600">{formatEur(p.annual_fixed_eur)}</td>
                  <td className="py-1.5 text-right font-mono text-gray-600">
                    {(p.similarity_score * 100).toFixed(0)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Section>

        <footer className="text-xs text-gray-500 border-t border-gray-300 pt-3">
          Estimates describe market patterns learned from Capology salaries and Transfermarkt market
          values; they are not advice on what any player should earn. Individual estimates can be
          far off — read the range and confidence, not just the median. See the{' '}
          <Link to="/model" className="text-blue-600 hover:underline">
            model card
          </Link>{' '}
          for accuracy, coverage, and limitations.
        </footer>
      </div>
    </div>
  );
}
