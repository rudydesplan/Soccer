import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MemoryRouter, Routes, Route } from 'react-router-dom';
import Benchmark from './Benchmark';
import { benchmarkResult, explanationResult } from '../test/fixtures';
import * as api from '../lib/api';

vi.mock('../lib/api', async (importOriginal) => {
  const original = await importOriginal<typeof api>();
  return { ...original, getBenchmarkById: vi.fn(), getExplanationById: vi.fn() };
});

const getBenchmarkByIdMock = vi.mocked(api.getBenchmarkById);
const getExplanationByIdMock = vi.mocked(api.getExplanationById);

function renderBenchmark(playerId: string) {
  return render(
    <MemoryRouter initialEntries={[`/benchmark/${playerId}`]}>
      <Routes>
        <Route path="/benchmark/:playerId" element={<Benchmark />} />
      </Routes>
    </MemoryRouter>
  );
}

beforeEach(() => {
  getBenchmarkByIdMock.mockReset();
  getExplanationByIdMock.mockReset();
});

describe('Benchmark page', () => {
  it('shows a loading state first', () => {
    getBenchmarkByIdMock.mockReturnValue(new Promise(() => {}));
    renderBenchmark('2');
    expect(screen.getByText('Loading benchmark...')).toBeInTheDocument();
  });

  it('renders the full result', async () => {
    getBenchmarkByIdMock.mockResolvedValue(benchmarkResult);
    renderBenchmark('2');

    expect(await screen.findByText('Erling Haaland')).toBeInTheDocument();
    expect(screen.getByText('⬆️ Overpaid')).toBeInTheDocument();
    expect(screen.getByText('Confidence: LOW')).toBeInTheDocument();
    expect(screen.getByText('95%')).toBeInTheDocument();
    expect(screen.getByText('12')).toBeInTheDocument();
    expect(screen.getByText('(8 with known salary)')).toBeInTheDocument();
    expect(screen.getByText('81.0%')).toBeInTheDocument();
    // Salary section + comparables
    expect(screen.getByText('Salary Range')).toBeInTheDocument();
    expect(screen.getByText('Kylian Mbappe')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /View all comparables/ })).toHaveAttribute(
      'href',
      '/comparables/2'
    );
    expect(screen.getByRole('link', { name: 'Printable report' })).toHaveAttribute(
      'href',
      '/report/2'
    );
    expect(getBenchmarkByIdMock).toHaveBeenCalledWith(2, 'normal');
  });

  it('shows the warning banner and fallback badge', async () => {
    getBenchmarkByIdMock.mockResolvedValue({
      ...benchmarkResult,
      model_used: 'no_mv',
      benchmark_warning: 'Fallback model used. Expect a wider, less precise range.',
    });
    renderBenchmark('2');
    expect(await screen.findByText('Limited reliability')).toBeInTheDocument();
    expect(screen.getByText('Fallback model (no market value)')).toBeInTheDocument();
  });

  it('re-fetches with wide range when toggled', async () => {
    getBenchmarkByIdMock.mockResolvedValue(benchmarkResult);
    renderBenchmark('2');
    await screen.findByText('Erling Haaland');

    fireEvent.click(screen.getByRole('button', { name: 'Wide (80%)' }));
    await waitFor(() => {
      expect(getBenchmarkByIdMock).toHaveBeenCalledWith(2, 'wide');
    });
  });

  it('loads the SHAP explanation on demand', async () => {
    getBenchmarkByIdMock.mockResolvedValue(benchmarkResult);
    getExplanationByIdMock.mockResolvedValue(explanationResult);
    renderBenchmark('2');
    await screen.findByText('Erling Haaland');

    expect(screen.getByText('Why this estimate?')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Explain this estimate' }));

    expect(await screen.findByText('Market value')).toBeInTheDocument();
    expect(getExplanationByIdMock).toHaveBeenCalledWith(2);
  });

  it('shows the API error message', async () => {
    getBenchmarkByIdMock.mockRejectedValue(new Error('Player 99999 not found'));
    renderBenchmark('99999');
    expect(await screen.findByText('Player 99999 not found')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Back to search/ })).toBeInTheDocument();
  });

  it('rejects a non-numeric player id without calling the API', async () => {
    renderBenchmark('abc');
    expect(await screen.findByText('Invalid player id')).toBeInTheDocument();
    expect(getBenchmarkByIdMock).not.toHaveBeenCalled();
  });
});
