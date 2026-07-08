import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import ExplanationPanel, { formatEffect } from './ExplanationPanel';
import { explanationResult } from '../test/fixtures';

describe('formatEffect', () => {
  it('formats moderate positive effects as +N%', () => {
    expect(formatEffect(33.5)).toBe('+34%');
  });

  it('formats negative effects with a minus sign', () => {
    expect(formatEffect(-11.3)).toBe('−11%');
  });

  it('switches to a multiplier for very large effects', () => {
    expect(formatEffect(1467.1)).toBe('×15.7');
  });
});

describe('ExplanationPanel', () => {
  it('shows the explain button and teaser before loading', () => {
    render(<ExplanationPanel playerKey="2" fetchExplanation={vi.fn()} />);
    expect(screen.getByText('Why this estimate?')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Explain this estimate' })).toBeInTheDocument();
    expect(screen.getByText(/push this player's expected salary/)).toBeInTheDocument();
  });

  it('loads and renders contributions on click', async () => {
    const fetcher = vi.fn().mockResolvedValue(explanationResult);
    render(<ExplanationPanel playerKey="2" fetchExplanation={fetcher} />);

    fireEvent.click(screen.getByRole('button', { name: 'Explain this estimate' }));

    expect(await screen.findByText('Market value')).toBeInTheDocument();
    expect(fetcher).toHaveBeenCalledTimes(1);
    // Baseline → prediction sentence
    expect(screen.getByText('€367K')).toBeInTheDocument();
    expect(screen.getByText('€20.0M')).toBeInTheDocument();
    // Effects: large positive as multiplier, negative with minus
    expect(screen.getByText('×15.7')).toBeInTheDocument();
    expect(screen.getByText('−11%')).toBeInTheDocument();
    // Feature values shown next to labels
    expect(screen.getByText(/€170.0M/)).toBeInTheDocument();
    // Button disappears once loaded
    expect(screen.queryByRole('button', { name: 'Explain this estimate' })).not.toBeInTheDocument();
  });

  it('shows the error message when the fetch fails', async () => {
    const fetcher = vi.fn().mockRejectedValue(new Error('Explanation failed (HTTP 500)'));
    render(<ExplanationPanel playerKey="2" fetchExplanation={fetcher} />);

    fireEvent.click(screen.getByRole('button', { name: 'Explain this estimate' }));
    expect(await screen.findByText('Explanation failed (HTTP 500)')).toBeInTheDocument();
    // Button remains for retry
    expect(screen.getByRole('button', { name: 'Explain this estimate' })).toBeInTheDocument();
  });

  it('resets to the button when playerKey changes', async () => {
    const fetcher = vi.fn().mockResolvedValue(explanationResult);
    const { rerender } = render(
      <ExplanationPanel playerKey="2" fetchExplanation={fetcher} />
    );
    fireEvent.click(screen.getByRole('button', { name: 'Explain this estimate' }));
    expect(await screen.findByText('Market value')).toBeInTheDocument();

    rerender(<ExplanationPanel playerKey="7" fetchExplanation={fetcher} />);
    expect(screen.queryByText('Market value')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Explain this estimate' })).toBeInTheDocument();
  });
});
