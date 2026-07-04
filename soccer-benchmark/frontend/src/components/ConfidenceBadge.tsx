interface Props {
  confidence: string;
}

export default function ConfidenceBadge({ confidence }: Props) {
  const colors: Record<string, string> = {
    HIGH: 'bg-green-100 text-green-800 border-green-300',
    MEDIUM: 'bg-yellow-100 text-yellow-800 border-yellow-300',
    LOW: 'bg-red-100 text-red-800 border-red-300',
  };

  const colorClass = colors[confidence] || colors.MEDIUM;

  return (
    <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium border ${colorClass}`}>
      Confidence: {confidence}
    </span>
  );
}
