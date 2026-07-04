import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { searchPlayers, type Player } from '../lib/api';

export default function PlayerSearch() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<Player[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const wrapperRef = useRef<HTMLDivElement>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);

    if (query.trim().length < 2) {
      setResults([]);
      setIsOpen(false);
      return;
    }

    debounceRef.current = setTimeout(async () => {
      setLoading(true);
      try {
        const data = await searchPlayers(query, 10);
        setResults(data);
        setIsOpen(true);
      } catch {
        setResults([]);
      } finally {
        setLoading(false);
      }
    }, 300);

    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [query]);

  function handleSelect(player: Player) {
    setIsOpen(false);
    setQuery('');
    navigate(`/benchmark/${player.id}`);
  }

  return (
    <div ref={wrapperRef} className="relative w-full max-w-lg">
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search for a player (e.g. Lamine Yamal)..."
        className="w-full px-4 py-3 rounded-lg border border-gray-300 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-lg"
      />
      {loading && (
        <div className="absolute right-3 top-3.5 text-gray-400">
          <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
        </div>
      )}
      {isOpen && results.length > 0 && (
        <ul className="absolute z-50 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg max-h-80 overflow-y-auto">
          {results.map((player) => {
            const missing: string[] = [];
            if (player.market_value_current_eur === null) missing.push('market value');
            if (player.main_position === null) missing.push('position');
            if (player.age_months === null) missing.push('age');
            const unbenchmarkable = player.main_position === null && player.age_months === null;
            return (
              <li
                key={player.id}
                onClick={() => handleSelect(player)}
                className="px-4 py-3 hover:bg-blue-50 cursor-pointer border-b border-gray-100 last:border-b-0"
              >
                <div className="font-medium text-gray-900">{player.player_name}</div>
                <div className="text-sm text-gray-500">
                  {player.main_position ?? 'Unknown position'} · {player.team_name} · {player.competition_country}
                  {missing.length > 0 && (
                    <span className="ml-2 text-amber-600 font-medium">
                      {unbenchmarkable
                        ? `no ${missing.join(', ')} — cannot benchmark`
                        : `no ${missing.join(', ')} — wider fallback estimate`}
                    </span>
                  )}
                </div>
              </li>
            );
          })}
        </ul>
      )}
      {isOpen && results.length === 0 && !loading && query.trim().length >= 2 && (
        <div className="absolute z-50 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg p-4 text-gray-500">
          No players found
        </div>
      )}
    </div>
  );
}
