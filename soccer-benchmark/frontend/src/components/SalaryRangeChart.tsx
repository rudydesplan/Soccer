import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer, Cell } from 'recharts';
import { formatEur } from '../lib/api';

interface Props {
  low: number;
  median: number;
  high: number;
  actual: number | null;
  status: string;
}

export default function SalaryRangeChart({ low, median, high, actual, status }: Props) {
  const data = [
    { name: 'Expected Range', low, median: median - low, high: high - median },
  ];

  const actualColor = status === 'OVERPAID' ? '#ef4444' : status === 'UNDERPAID' ? '#f59e0b' : '#3b82f6';
  const xMax = Math.max(high, actual ?? 0) * 1.1;

  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={120}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            type="number"
            tickFormatter={(v) => formatEur(v)}
            domain={[0, xMax]}
          />
          <YAxis type="category" dataKey="name" hide />
          {/* Bars are stacked segments (low, median-low, high-median); show the
              actual boundary values in the tooltip instead of segment deltas. */}
          <Tooltip
            formatter={(_value, name) => {
              const actualValues: Record<string, number> = { Low: low, Median: median, High: high };
              const actualValue = actualValues[String(name)];
              return [formatEur(actualValue ?? (_value as number)), name];
            }}
          />
          <Bar dataKey="low" stackId="range" fill="#d1fae5" name="Low">
            <Cell fill="#d1fae5" />
          </Bar>
          <Bar dataKey="median" stackId="range" fill="#10b981" name="Median">
            <Cell fill="#10b981" />
          </Bar>
          <Bar dataKey="high" stackId="range" fill="#6ee7b7" name="High">
            <Cell fill="#6ee7b7" />
          </Bar>
          {actual !== null && actual !== undefined && (
            <ReferenceLine
              x={actual}
              stroke={actualColor}
              strokeWidth={3}
              strokeDasharray="5 5"
              label={{ value: `Actual: ${formatEur(actual)}`, position: 'top', fill: actualColor, fontSize: 12 }}
            />
          )}
        </BarChart>
      </ResponsiveContainer>
      <div className="flex justify-between text-sm text-gray-600 mt-2 px-5">
        <span>Low: {formatEur(low)}</span>
        <span className="font-semibold">Median: {formatEur(median)}</span>
        <span>High: {formatEur(high)}</span>
      </div>
    </div>
  );
}
