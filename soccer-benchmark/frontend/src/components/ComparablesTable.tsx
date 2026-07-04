import { useState } from 'react';
import { type ComparablePlayer, formatEur, formatAge } from '../lib/api';

interface Props {
  players: ComparablePlayer[];
}

export default function ComparablesTable({ players }: Props) {
  const [sortAsc, setSortAsc] = useState(false);

  const sorted = [...players].sort((a, b) =>
    sortAsc ? a.similarity_score - b.similarity_score : b.similarity_score - a.similarity_score
  );

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Name</th>
            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Position</th>
            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">League</th>
            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Country</th>
            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Age</th>
            <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Market Value</th>
            <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Salary</th>
            <th
              className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase cursor-pointer hover:text-blue-600"
              onClick={() => setSortAsc(!sortAsc)}
            >
              Similarity {sortAsc ? '↑' : '↓'}
            </th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {sorted.map((player, idx) => (
            <tr key={player.id ?? `${player.player_name}-${idx}`} className="hover:bg-gray-50">
              <td className="px-4 py-3 text-sm font-medium text-gray-900">{player.player_name}</td>
              <td className="px-4 py-3 text-sm text-gray-600">{player.main_position}</td>
              <td className="px-4 py-3 text-sm text-gray-600">{player.competition_id}</td>
              <td className="px-4 py-3 text-sm text-gray-600">{player.competition_country}</td>
              <td className="px-4 py-3 text-sm text-gray-600">{formatAge(player.age_months)}</td>
              <td className="px-4 py-3 text-sm text-gray-600 text-right">{formatEur(player.market_value_current_eur)}</td>
              <td className="px-4 py-3 text-sm text-gray-600 text-right">{formatEur(player.annual_fixed_eur)}</td>
              <td className="px-4 py-3 text-sm text-gray-600 text-right font-mono">
                {(player.similarity_score * 100).toFixed(1)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
