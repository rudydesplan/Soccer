import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter, Routes, Route } from 'react-router-dom';
import Comparables from './Comparables';
import { benchmarkResult } from '../test/fixtures';
import * as api from '../lib/api';

vi.mock('../lib/api', async (importOriginal) => {
  const original = await importOriginal<typeof api>();
  return { ...original, getBenchmarkById: vi.fn() };
});

const getBenchmarkByIdMock = vi.mocked(api.getBenchmarkById);

function renderComparables(playerId: string) {
  return render(
    <MemoryRouter initialEntries={[`/comparables/${playerId}`]}>
      <Routes>
        <Route path="/comparables/:playerId" element={<Comparables />} />
      </Routes>
    </MemoryRouter>
  );
}

beforeEach(() => {
  getBenchmarkByIdMock.mockReset();
});

describe('Comparables page', () => {
  it('requests all comparables and renders them', async () => {
    getBenchmarkByIdMock.mockResolvedValue(benchmarkResult);
    renderComparables('2');

    expect(
      await screen.findByText('Comparable Players for Erling Haaland')
    ).toBeInTheDocument();
    expect(screen.getByText(/12 comparable players found at level: 1/)).toBeInTheDocument();
    expect(screen.getByText('Kylian Mbappe')).toBeInTheDocument();
    expect(screen.getByText('Harry Kane')).toBeInTheDocument();
    expect(getBenchmarkByIdMock).toHaveBeenCalledWith(2, 'normal', true);
    expect(screen.getByRole('link', { name: /Back to benchmark/ })).toHaveAttribute(
      'href',
      '/benchmark/2'
    );
  });

  it('shows the error state', async () => {
    getBenchmarkByIdMock.mockRejectedValue(new Error('Player 5 not found'));
    renderComparables('5');
    expect(await screen.findByText('Player 5 not found')).toBeInTheDocument();
  });

  it('rejects a non-numeric player id', async () => {
    renderComparables('nope');
    expect(await screen.findByText('Invalid player id')).toBeInTheDocument();
    expect(getBenchmarkByIdMock).not.toHaveBeenCalled();
  });
});
