import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import ManualBenchmark from './ManualBenchmark';
import { benchmarkOptions, benchmarkResult, explanationResult } from '../test/fixtures';
import * as api from '../lib/api';

vi.mock('../lib/api', async (importOriginal) => {
  const original = await importOriginal<typeof api>();
  return {
    ...original,
    getBenchmarkOptions: vi.fn(),
    getManualBenchmark: vi.fn(),
    getManualExplanation: vi.fn(),
  };
});

const getOptionsMock = vi.mocked(api.getBenchmarkOptions);
const getManualMock = vi.mocked(api.getManualBenchmark);
const getManualExplanationMock = vi.mocked(api.getManualExplanation);

function renderPage() {
  return render(
    <MemoryRouter>
      <ManualBenchmark />
    </MemoryRouter>
  );
}

async function fillRequiredFields() {
  await screen.findByRole('option', { name: 'Centre-Forward' });
  fireEvent.change(screen.getByLabelText(/Position \*/), {
    target: { value: 'Centre-Forward' },
  });
  fireEvent.change(screen.getByLabelText(/Age \(years\) \*/), { target: { value: '25' } });
}

beforeEach(() => {
  getOptionsMock.mockReset();
  getManualMock.mockReset();
  getManualExplanationMock.mockReset();
  getOptionsMock.mockResolvedValue(benchmarkOptions);
});

describe('ManualBenchmark page', () => {
  it('loads positions and competitions into the form', async () => {
    renderPage();
    expect(await screen.findByRole('option', { name: 'Centre-Forward' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Premier League (England)' })).toBeInTheDocument();
  });

  it('shows an error when the options endpoint fails', async () => {
    getOptionsMock.mockRejectedValue(new Error('Failed to load form options (HTTP 503)'));
    renderPage();
    expect(
      await screen.findByText('Failed to load form options (HTTP 503)')
    ).toBeInTheDocument();
  });

  // Note: fireEvent.submit bypasses the browser's native constraint
  // validation (min/max/required), exercising the component's own
  // buildInput() checks — which is exactly what these tests target.
  function submitForm() {
    fireEvent.submit(
      screen.getByRole('button', { name: 'Run benchmark' }).closest('form')!
    );
  }

  it('validates the age range client-side', async () => {
    renderPage();
    await fillRequiredFields();
    fireEvent.change(screen.getByLabelText(/Age \(years\) \*/), { target: { value: '60' } });
    submitForm();

    expect(await screen.findByText('Age must be between 12.5 and 50 years.')).toBeInTheDocument();
    expect(getManualMock).not.toHaveBeenCalled();
  });

  it('rejects a negative market value', async () => {
    renderPage();
    await fillRequiredFields();
    fireEvent.change(screen.getByLabelText(/Market value \(EUR\)/), { target: { value: '-5' } });
    submitForm();

    expect(
      await screen.findByText('Market value must be a positive number (in EUR).')
    ).toBeInTheDocument();
    expect(getManualMock).not.toHaveBeenCalled();
  });

  it('submits the form and renders the result', async () => {
    getManualMock.mockResolvedValue({
      ...benchmarkResult,
      player_name: 'custom_player',
      salary_status: 'UNKNOWN',
      actual_salary_eur: null,
      salary_percentile: null,
    });
    renderPage();
    await fillRequiredFields();
    fireEvent.change(screen.getByLabelText(/League/), { target: { value: 'GB1' } });
    fireEvent.change(screen.getByLabelText(/Market value \(EUR\)/), {
      target: { value: '50000000' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Run benchmark' }));

    expect(await screen.findByText('Unknown salary')).toBeInTheDocument();
    expect(getManualMock).toHaveBeenCalledWith(
      {
        main_position: 'Centre-Forward',
        age_months: 300,
        competition_id: 'GB1',
        competition_country: 'England',
        market_value_current_eur: 50_000_000,
      },
      'normal'
    );
    expect(screen.getByText('Salary Range')).toBeInTheDocument();
    expect(screen.getByText('Kylian Mbappe')).toBeInTheDocument();
  });

  it('shows the backend error message when the benchmark fails', async () => {
    getManualMock.mockRejectedValue(new Error('age_months is required for manual benchmarks'));
    renderPage();
    await fillRequiredFields();
    fireEvent.click(screen.getByRole('button', { name: 'Run benchmark' }));

    expect(
      await screen.findByText('age_months is required for manual benchmarks')
    ).toBeInTheDocument();
  });

  it('offers a SHAP explanation for the submitted custom player', async () => {
    getManualMock.mockResolvedValue(benchmarkResult);
    getManualExplanationMock.mockResolvedValue({
      ...explanationResult,
      model_used: 'no_mv',
      player_name: 'custom_player',
    });
    renderPage();
    await fillRequiredFields();
    fireEvent.click(screen.getByRole('button', { name: 'Run benchmark' }));
    await screen.findByText('Salary Range');

    fireEvent.click(screen.getByRole('button', { name: 'Explain this estimate' }));
    expect(await screen.findByText('Market value')).toBeInTheDocument();
    expect(getManualExplanationMock).toHaveBeenCalledWith({
      main_position: 'Centre-Forward',
      age_months: 300,
    });
  });

  it('re-runs the last input when the range width is toggled', async () => {
    getManualMock.mockResolvedValue(benchmarkResult);
    renderPage();
    await fillRequiredFields();
    fireEvent.click(screen.getByRole('button', { name: 'Run benchmark' }));
    await screen.findByText('Salary Range');

    fireEvent.click(screen.getByRole('button', { name: 'Wide (80%)' }));
    await waitFor(() => {
      expect(getManualMock).toHaveBeenCalledTimes(2);
    });
    expect(getManualMock.mock.calls[1][1]).toBe('wide');
  });
});
