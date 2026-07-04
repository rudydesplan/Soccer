import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getBenchmarkById, type BenchmarkResult } from '../lib/api';
import ComparablesTable from '../components/ComparablesTable';

export default function Comparables() {
  const { playerId } = useParams<{ playerId: string }>();
  const [result, setResult] = useState<BenchmarkResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const id = Number(playerId);
    if (!Number.isInteger(id)) {
      setError('Invalid player id');
      setLoading(false);
      return;
    }
    setLoading(true);
    setError(null);
    getBenchmarkById(id, 'normal', true)
      .then(setResult)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [playerId]);

  if (loading) {
    return (
      <div className="max-w-6xl mx-auto px-4 py-12 text-center">
        <div className="text-xl text-gray-500 animate-pulse">Loading comparables...</div>
      </div>
    );
  }

  if (error || !result) {
    return (
      <div className="max-w-6xl mx-auto px-4 py-12 text-center">
        <div className="bg-red-50 border border-red-200 rounded-lg p-6">
          <p className="text-red-700">{error || 'Failed to load comparables'}</p>
          <Link to="/" className="text-blue-600 hover:underline mt-4 inline-block">
            ← Back to search
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-6">
      <div className="flex items-center gap-4">
        <Link
          to={`/benchmark/${result.player_id}`}
          className="text-blue-600 hover:underline text-sm"
        >
          ← Back to benchmark
        </Link>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">
          Comparable Players for {result.player_name}
        </h2>
        <p className="text-sm text-gray-600 mb-6">
          {result.benchmark_n_comparables} comparable players found at level: {result.comparable_level_used}
        </p>
        <ComparablesTable players={result.comparable_players} />
      </div>
    </div>
  );
}
