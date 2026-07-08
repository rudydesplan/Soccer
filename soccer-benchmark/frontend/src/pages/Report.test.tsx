import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { MemoryRouter, Routes, Route } from 'react-router-dom';
import Report from './Report';
import { benchmarkResult, explanationResult } from '../test/fixtures';
import * as api from '../lib/api';

vi.mock('../lib/api', async (importOriginal) => {
  const original = await importOriginal<typeof api>();
  return { ...original, getBenchmarkById: vi.fn(), getExplanationById: vi.fn() };
});

const getBenchmarkByIdMock = vi.mocked(api.getBenchmarkById);
const getExplanationByIdMock = vi.mocked(api.getExplanationById);

function renderReport(playerId: string) {
  return render(
    <MemoryRouter initialEntries={[`/report/${playerId}`]}>
      <Routes>
        <Route path="/report/:playerId" element={<Report />} />
      </Routes>
    </MemoryRouter>
  );
}

beforeEach(() => {
  getBenchmarkByIdMock.mockReset();
  getExplanationByIdMock.mockReset();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('Report page', () => {
  it('shows a loading state while both requests are in flight', () => {
    getBenchmarkByIdMock.mockReturnValue(new Promise(() => {}));
    getExplanationByIdMock.mockReturnValue(new Promise(() => {}));
    renderReport('2');
    expect(screen.getByTestId('report-loading')).toBeInTheDocument();
  });

  it('renders benchmark, confidence, explanation, and comparables together', async () => {
    getBenchmarkByIdMock.mockResolvedValue(benchmarkResult);
    getExplanationByIdMock.mockResolvedValue(explanationResult);
    renderReport('2');

    expect(await screen.findByText('Salary Benchmark Report')).toBeInTheDocument();
    // Player + result
    expect(screen.getByText('Erling Haaland')).toBeInTheDocument();
    expect(screen.getByText('Overpaid')).toBeInTheDocument();
    expect(screen.getByText('LOW')).toBeInTheDocument();
    expect(screen.getByText('€20.0M')).toBeInTheDocument();
    // SHAP table
    const explanationTable = screen.getByTestId('report-explanation');
    expect(explanationTable).toHaveTextContent('Market value');
    expect(explanationTable).toHaveTextContent('×15.7');
    // Comparables table
    const comparablesTable = screen.getByTestId('report-comparables');
    expect(comparablesTable).toHaveTextContent('Kylian Mbappe');
    expect(comparablesTable).toHaveTextContent('Harry Kane');
    // Both requests fired for the same player
    expect(getBenchmarkByIdMock).toHaveBeenCalledWith(2);
    expect(getExplanationByIdMock).toHaveBeenCalledWith(2);
  });

  it('includes the benchmark warning when present', async () => {
    getBenchmarkByIdMock.mockResolvedValue({
      ...benchmarkResult,
      benchmark_warning: 'Fallback model used. Expect a wider, less precise range.',
    });
    getExplanationByIdMock.mockResolvedValue(explanationResult);
    renderReport('2');

    expect(await screen.findByTestId('report-warning')).toHaveTextContent(
      'Fallback model used'
    );
  });

  it('omits the warning block when there is none', async () => {
    getBenchmarkByIdMock.mockResolvedValue(benchmarkResult);
    getExplanationByIdMock.mockResolvedValue(explanationResult);
    renderReport('2');

    await screen.findByText('Salary Benchmark Report');
    expect(screen.queryByTestId('report-warning')).not.toBeInTheDocument();
  });

  it('triggers window.print from the toolbar button', async () => {
    getBenchmarkByIdMock.mockResolvedValue(benchmarkResult);
    getExplanationByIdMock.mockResolvedValue(explanationResult);
    const printMock = vi.fn();
    vi.stubGlobal('print', printMock);
    renderReport('2');

    await screen.findByText('Salary Benchmark Report');
    fireEvent.click(screen.getByRole('button', { name: 'Print / Save as PDF' }));
    expect(printMock).toHaveBeenCalledOnce();
  });

  it('fails the whole report when the explanation fails', async () => {
    getBenchmarkByIdMock.mockResolvedValue(benchmarkResult);
    getExplanationByIdMock.mockRejectedValue(new Error('Explanation failed'));
    renderReport('2');

    expect(await screen.findByText('Explanation failed')).toBeInTheDocument();
    expect(screen.queryByText('Salary Benchmark Report')).not.toBeInTheDocument();
  });

  it('rejects a non-numeric player id without calling the API', async () => {
    renderReport('abc');
    expect(await screen.findByText('Invalid player id')).toBeInTheDocument();
    expect(getBenchmarkByIdMock).not.toHaveBeenCalled();
  });
});
