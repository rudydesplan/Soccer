import { formatEur, formatAge } from '../lib/api';

interface Props {
  name: string;
  position: string;
  competition: string;
  age_months: number | null;
  market_value: number | null;
}

export default function PlayerCard({ name, position, competition, age_months, market_value }: Props) {
  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-bold text-gray-900">{name}</h2>
      <div className="mt-3 grid grid-cols-2 md:grid-cols-4 gap-4">
        <div>
          <p className="text-xs text-gray-500 uppercase">Position</p>
          <p className="text-sm font-medium text-gray-800">{position}</p>
        </div>
        <div>
          <p className="text-xs text-gray-500 uppercase">Competition</p>
          <p className="text-sm font-medium text-gray-800">{competition}</p>
        </div>
        <div>
          <p className="text-xs text-gray-500 uppercase">Age</p>
          <p className="text-sm font-medium text-gray-800">{formatAge(age_months)}</p>
        </div>
        <div>
          <p className="text-xs text-gray-500 uppercase">Market Value</p>
          <p className="text-sm font-medium text-gray-800">{formatEur(market_value)}</p>
        </div>
      </div>
    </div>
  );
}
