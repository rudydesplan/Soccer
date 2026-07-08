import { useEffect, useState } from 'react';
import { type ModelCard as ModelCardData, getModelCard } from '../lib/api';

/** Multiplicative effect of a log-space residual, as "−27%" / "+41%". */
export function formatResidualPct(residualLog: number): string {
  const pct = Math.round((Math.exp(residualLog) - 1) * 100);
  return pct >= 0 ? `+${pct}%` : `−${Math.abs(pct)}%`;
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="bg-white rounded-lg shadow-md p-6 mb-6">
      <h2 className="text-xl font-semibold text-gray-900 mb-3">{title}</h2>
      {children}
    </section>
  );
}

function Stat({ value, label }: { value: string; label: string }) {
  return (
    <div className="bg-gray-50 rounded-lg p-4 text-center">
      <div className="text-2xl font-bold text-gray-900">{value}</div>
      <div className="text-sm text-gray-600 mt-1">{label}</div>
    </div>
  );
}

const LIMITATIONS = [
  'Salaries are known for only a fraction of the pool — the model learns from those players and extrapolates to the rest.',
  'Salary figures come from Capology and are estimates, not official club disclosures.',
  'Coverage is limited to men\u2019s leagues in five countries; estimates for players outside these leagues are unreliable.',
  'A single season of data: the model reflects today\u2019s market, not salary trends over time.',
  'Individual estimates can be far off — roughly 3 in 10 predictions miss the actual salary by more than 50%. Always read the range and the confidence badge, not just the median.',
  'Bonuses, image rights, and signing fees are not modeled — only fixed annual gross salary.',
];

const FAIRNESS = [
  'Nationality is a model feature because it carries real market signal. That means the model reproduces any nationality-based pay differences that exist in the market — it describes the market, it does not correct it.',
  'OVERPAID / UNDERPAID labels are statements about market patterns, never a judgment about what a player deserves or a recommendation to change anyone\u2019s pay.',
  'Use estimates as one input among many in contract decisions — alongside scouting, role, injury history, and dressing-room factors the model cannot see.',
];

export default function ModelCard() {
  const [card, setCard] = useState<ModelCardData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getModelCard()
      .then(setCard)
      .catch((err) => setError(err instanceof Error ? err.message : 'Failed to load model card'));
  }, []);

  if (error) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-16">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">{error}</div>
      </div>
    );
  }

  if (!card) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-16 text-center text-gray-500" data-testid="model-card-loading">
        Loading model card…
      </div>
    );
  }

  const m = card.metrics;
  const cal = card.calibration;
  const cov = card.coverage;
  const maxImportance = Math.max(...card.top_features.map((f) => f.importance), 1e-9);
  const trainedDate = card.trained_at ? new Date(card.trained_at).toLocaleDateString() : null;

  return (
    <div className="max-w-4xl mx-auto px-4 py-10">
      <h1 className="text-3xl font-bold text-gray-900 mb-2">About the model</h1>
      <p className="text-gray-600 mb-8">
        What powers the salary estimates, how accurate they are, and where they should not be trusted.
      </p>

      <Section title="What this tool does">
        <p className="text-gray-700">
          The benchmark estimates the <strong>expected gross annual salary</strong> for a player,
          given their market value, age, position, league, and contract situation. It is trained on{' '}
          {cov.n_with_salary.toLocaleString()} player-seasons with known salaries and compares any
          player against that market. The output is a range, a market position
          (OVERPAID / UNDERPAID / FAIRLY PAID), and a set of comparable players.
        </p>
        <p className="text-gray-500 text-sm mt-3">
          Model: {card.model_name} ({card.framework}
          {card.framework_version ? ` ${card.framework_version}` : ''})
          {trainedDate ? `, trained ${trainedDate}` : ''}. Fallback variants cover players with
          missing market value, position, or age.
        </p>
      </Section>

      <Section title="How accurate is it?">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4" data-testid="metric-cards">
          <Stat value={`${Math.round(m.r2 * 100)}%`} label="of salary variation explained" />
          <Stat value={`±${Math.round(m.median_ape_pct)}%`} label="typical estimate error (median)" />
          <Stat value={`${Math.round(m.within_50_pct)}%`} label="of estimates within ±50% of actual" />
          <Stat value={m.n_test.toLocaleString()} label="players in the held-out test set" />
        </div>
        <p className="text-gray-700 text-sm">
          Numbers come from a held-out test the model never saw during training, split so that{' '}
          <strong>no player appears in both training and test data</strong>. In plain terms: half of
          all estimates land within roughly ±{Math.round(m.median_ape_pct)}% of the player&apos;s
          real salary, and about {Math.round(m.within_50_pct / 10)} in 10 land within ±50%. Football
          salaries are noisy — two similar players can earn very different wages — so a sizable
          minority of estimates will be far off. That is exactly why every result ships with a range
          and a confidence badge.
        </p>
      </Section>

      <Section title="How the salary range is built">
        <p className="text-gray-700 text-sm mb-3">
          The model predicts a single number; the range around it is calibrated from{' '}
          {cal.n_samples.toLocaleString()} out-of-sample prediction errors ({cal.n_folds}-fold
          cross-validation, grouped by player). The range is honest by construction: it reflects how
          wrong the model actually was on players it had never seen.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-gray-50 rounded-lg p-4">
            <div className="font-semibold text-gray-900 mb-1">Normal range</div>
            <p className="text-sm text-gray-600">
              Spans {formatResidualPct(cal.residual_p25)} to {formatResidualPct(cal.residual_p75)}{' '}
              around the median — half of real salaries fall inside it.
            </p>
          </div>
          <div className="bg-gray-50 rounded-lg p-4">
            <div className="font-semibold text-gray-900 mb-1">Wide range</div>
            <p className="text-sm text-gray-600">
              Spans {formatResidualPct(cal.residual_p10)} to {formatResidualPct(cal.residual_p90)}{' '}
              around the median — 8 in 10 real salaries fall inside it.
            </p>
          </div>
        </div>
      </Section>

      <Section title="What drives the estimates">
        <p className="text-gray-700 text-sm mb-4">
          Feature importance from the trained model (how much accuracy drops when a feature is
          shuffled). Market value and age dominate; everything else fine-tunes.
        </p>
        <div className="space-y-2" data-testid="importance-bars">
          {card.top_features.map((f) => (
            <div key={f.feature} className="flex items-center gap-3 text-sm">
              <div className="w-56 shrink-0 text-right text-gray-800">{f.label}</div>
              <div className="flex-1">
                <div
                  className="h-4 rounded bg-blue-500"
                  style={{ width: `${(f.importance / maxImportance) * 100}%` }}
                />
              </div>
              <div className="w-14 shrink-0 font-mono text-right text-gray-600">
                {f.importance.toFixed(2)}
              </div>
            </div>
          ))}
        </div>
      </Section>

      <Section title="Data coverage">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-3">
          <Stat value={cov.n_players.toLocaleString()} label="players in the pool" />
          <Stat value={cov.n_with_salary.toLocaleString()} label="with a known salary" />
          <Stat value={String(cov.n_leagues)} label={`leagues in ${cov.countries.length} countries`} />
          <Stat value={cov.seasons.map((s) => `${s}/${(s + 1) % 100}`).join(', ')} label="season covered" />
        </div>
        <p className="text-gray-700 text-sm">
          Countries: {cov.countries.join(', ')}. Market values come from Transfermarkt (
          {Math.round((cov.n_with_market_value / cov.n_rows) * 100)}% of the pool), salaries from
          Capology. Only {Math.round((cov.n_with_salary / cov.n_rows) * 100)}% of players have a
          known salary — those are the players the model learns from.
        </p>
      </Section>

      <Section title="Limitations">
        <ul className="list-disc pl-5 space-y-2 text-sm text-gray-700" data-testid="limitations-list">
          {LIMITATIONS.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </Section>

      <Section title="Fairness & responsible use">
        <ul className="list-disc pl-5 space-y-2 text-sm text-gray-700" data-testid="fairness-list">
          {FAIRNESS.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </Section>
    </div>
  );
}
