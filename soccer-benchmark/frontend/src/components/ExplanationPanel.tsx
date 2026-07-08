import { useState } from 'react';
import { type ExplanationResult, formatEur } from '../lib/api';

interface Props {
  /** Fetches the explanation for the player currently shown. */
  fetchExplanation: () => Promise<ExplanationResult>;
  /** Reset key: when this changes, a previously loaded explanation is stale. */
  playerKey: string;
}

const MAX_BARS = 8;

/** "+35%", "−20%", or "×15.7" for very large positive effects. */
export function formatEffect(pctEffect: number): string {
  if (pctEffect >= 300) return `×${(1 + pctEffect / 100).toFixed(1)}`;
  const rounded = Math.round(pctEffect);
  return rounded >= 0 ? `+${rounded}%` : `−${Math.abs(rounded)}%`;
}

export default function ExplanationPanel({ fetchExplanation, playerKey }: Props) {
  const [result, setResult] = useState<ExplanationResult | null>(null);
  const [loadedFor, setLoadedFor] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const current = loadedFor === playerKey ? result : null;

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchExplanation();
      setResult(res);
      setLoadedFor(playerKey);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Explanation failed');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex justify-between items-center mb-2">
        <h3 className="text-lg font-semibold text-gray-900">Why this estimate?</h3>
        {!current && (
          <button
            onClick={load}
            disabled={loading}
            className="px-4 py-2 rounded-lg bg-blue-600 text-white text-sm font-medium hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            {loading ? 'Analyzing…' : 'Explain this estimate'}
          </button>
        )}
      </div>

      {!current && !loading && !error && (
        <p className="text-sm text-gray-500">
          See which features push this player&apos;s expected salary up or down
          compared to a typical player.
        </p>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {current && (
        <div>
          <p className="text-sm text-gray-600 mb-4">
            A typical player&apos;s estimate is{' '}
            <span className="font-semibold">{formatEur(current.base_salary_eur)}</span>. The
            features below move this player to{' '}
            <span className="font-semibold">{formatEur(current.predicted_salary_eur)}</span>.
            Each bar shows how much that feature multiplies the estimate.
          </p>

          <div className="space-y-2" data-testid="explanation-bars">
            {current.features.slice(0, MAX_BARS).map((f) => {
              const maxImpact = Math.max(
                ...current.features.map((x) => Math.abs(x.shap_log)),
                1e-9
              );
              const widthPct = (Math.abs(f.shap_log) / maxImpact) * 100;
              const positive = f.shap_log >= 0;
              return (
                <div key={f.feature} className="flex items-center gap-2 text-sm">
                  <div className="w-56 shrink-0 text-right">
                    <span className="text-gray-800">{f.label}</span>
                    {f.value !== null && (
                      <span className="text-gray-400"> · {f.value}</span>
                    )}
                  </div>
                  <div className="flex-1 flex items-center">
                    <div className="w-1/2 flex justify-end">
                      {!positive && (
                        <div
                          className="h-4 rounded-l bg-red-400"
                          style={{ width: `${widthPct / 2}%` }}
                        />
                      )}
                    </div>
                    <div className="w-px h-5 bg-gray-300" />
                    <div className="w-1/2">
                      {positive && (
                        <div
                          className="h-4 rounded-r bg-emerald-500"
                          style={{ width: `${widthPct / 2}%` }}
                        />
                      )}
                    </div>
                  </div>
                  <div
                    className={`w-16 shrink-0 font-mono text-right ${positive ? 'text-emerald-700' : 'text-red-600'}`}
                  >
                    {formatEffect(f.pct_effect)}
                  </div>
                </div>
              );
            })}
          </div>

          {current.features.length > MAX_BARS && (
            <p className="text-xs text-gray-400 mt-2">
              {current.features.length - MAX_BARS} smaller effects not shown.
            </p>
          )}

          <p className="text-xs text-gray-400 mt-3">
            SHAP values on the model&apos;s log-salary output; effects are multiplicative.
            The model estimate can differ slightly from the calibrated median range.
          </p>
        </div>
      )}
    </div>
  );
}
