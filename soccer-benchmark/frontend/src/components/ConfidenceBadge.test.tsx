import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import ConfidenceBadge from './ConfidenceBadge';

describe('ConfidenceBadge', () => {
  it.each([
    ['HIGH', 'bg-green-100'],
    ['MEDIUM', 'bg-yellow-100'],
    ['LOW', 'bg-red-100'],
  ])('renders %s with its color', (confidence, colorClass) => {
    render(<ConfidenceBadge confidence={confidence} />);
    const badge = screen.getByText(`Confidence: ${confidence}`);
    expect(badge).toBeInTheDocument();
    expect(badge.className).toContain(colorClass);
  });

  it('falls back to MEDIUM colors for unknown values', () => {
    render(<ConfidenceBadge confidence="WHATEVER" />);
    expect(screen.getByText('Confidence: WHATEVER').className).toContain('bg-yellow-100');
  });
});
