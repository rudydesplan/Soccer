import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import ModelCard, { formatResidualPct } from './ModelCard';
import { modelCard } from '../test/fixtures';

function mockFetchOnce(body: unknown, ok = true, status = 200) {
  vi.mocked(globalThis.fetch).mockResolvedValueOnce({
    ok,
    status,
    json: () => Promise.resolve(body),
  } as Response);
}

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn());
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('formatResidualPct', () => {
  it('converts log residuals to multiplicative percentages', () => {
    expect(formatResidualPct(0)).toBe('+0%');
    // exp(-0.3086) ≈ 0.734 → −27%
    expect(formatResidualPct(-0.3086)).toBe('−27%');
    // exp(0.3422) ≈ 1.408 → +41%
    expect(formatResidualPct(0.3422)).toBe('+41%');
  });
});

describe('ModelCard', () => {
  it('shows a loading state then renders all sections', async () => {
    mockFetchOnce(modelCard);
    render(<ModelCard />);
    expect(screen.getByTestId('model-card-loading')).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByRole('heading', { name: 'About the model' })).toBeInTheDocument();
    });
    expect(screen.getByRole('heading', { name: 'What this tool does' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'How accurate is it?' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'How the salary range is built' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'What drives the estimates' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Data coverage' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Limitations' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Fairness & responsible use' })).toBeInTheDocument();
  });

  it('renders metrics in business language', async () => {
    mockFetchOnce(modelCard);
    render(<ModelCard />);
    await waitFor(() => {
      // r2 0.7465 → 75%
      expect(screen.getByText('75%')).toBeInTheDocument();
    });
    // median APE 31.4 → ±31%
    expect(screen.getByText('±31%')).toBeInTheDocument();
    // within_50 71.1 → 71%
    expect(screen.getByText('71%')).toBeInTheDocument();
    // model identity line
    expect(screen.getByText(/WeightedEnsemble_L2_FULL/)).toBeInTheDocument();
  });

  it('renders one importance bar per feature, sorted', async () => {
    mockFetchOnce(modelCard);
    render(<ModelCard />);
    await waitFor(() => {
      expect(screen.getByTestId('importance-bars')).toBeInTheDocument();
    });
    const bars = screen.getByTestId('importance-bars');
    expect(bars.children).toHaveLength(modelCard.top_features.length);
    expect(bars.children[0].textContent).toContain('Market value');
  });

  it('renders coverage numbers', async () => {
    mockFetchOnce(modelCard);
    render(<ModelCard />);
    await waitFor(() => {
      expect(screen.getByText('18,490')).toBeInTheDocument();
    });
    expect(screen.getByText('3,363')).toBeInTheDocument();
    expect(screen.getByText(/England, France, Germany, Italy, Spain/)).toBeInTheDocument();
  });

  it('shows the API error message on failure', async () => {
    mockFetchOnce({ detail: 'Model artifact missing' }, false, 503);
    render(<ModelCard />);
    await waitFor(() => {
      expect(screen.getByText('Model artifact missing')).toBeInTheDocument();
    });
  });
});
