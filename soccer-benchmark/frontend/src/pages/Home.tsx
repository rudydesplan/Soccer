import { Link } from 'react-router-dom';
import PlayerSearch from '../components/PlayerSearch';

export default function Home() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-16">
      <div className="text-center">
        <h1 className="text-5xl font-bold text-gray-900 mb-4">
          Soccer Salary Benchmark
        </h1>
        <p className="text-xl text-gray-600 mb-8">
          Compare player salaries against similar players using AI-powered benchmarking.
          Find out if a player is underpaid, fairly paid, or overpaid.
        </p>
        <div className="flex justify-center mb-3">
          <PlayerSearch />
        </div>
        <p className="text-sm text-gray-500 mb-12">
          Player not in the list?{' '}
          <Link to="/manual" className="text-blue-600 hover:underline font-medium">
            Benchmark a custom player →
          </Link>
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-12">
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="text-3xl mb-2">📊</div>
            <h3 className="font-semibold text-gray-900 mb-1">Salary Benchmarking</h3>
            <p className="text-sm text-gray-600">
              Get expected salary ranges based on comparable players
            </p>
          </div>
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="text-3xl mb-2">🔍</div>
            <h3 className="font-semibold text-gray-900 mb-1">Comparable Players</h3>
            <p className="text-sm text-gray-600">
              See which players are most similar by market value, league, position, age, and contract
            </p>
          </div>
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="text-3xl mb-2">⚡</div>
            <h3 className="font-semibold text-gray-900 mb-1">Instant Analysis</h3>
            <p className="text-sm text-gray-600">
              Get results in seconds with confidence scoring
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
