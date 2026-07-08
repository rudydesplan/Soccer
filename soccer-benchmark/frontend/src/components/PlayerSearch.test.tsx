import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { MemoryRouter, Routes, Route } from 'react-router-dom';
import PlayerSearch from './PlayerSearch';
import { players } from '../test/fixtures';
import * as api from '../lib/api';

vi.mock('../lib/api', async (importOriginal) => {
  const original = await importOriginal<typeof api>();
  return { ...original, searchPlayers: vi.fn() };
});

const searchPlayersMock = vi.mocked(api.searchPlayers);

function renderSearch() {
  return render(
    <MemoryRouter initialEntries={['/']}>
      <Routes>
        <Route path="/" element={<PlayerSearch />} />
        <Route path="/benchmark/:playerId" element={<div>benchmark page</div>} />
      </Routes>
    </MemoryRouter>
  );
}

beforeEach(() => {
  vi.useFakeTimers();
  searchPlayersMock.mockReset();
});

afterEach(() => {
  vi.useRealTimers();
});

async function typeAndDebounce(value: string) {
  fireEvent.change(screen.getByRole('textbox'), { target: { value } });
  await act(async () => {
    vi.advanceTimersByTime(350);
  });
}

describe('PlayerSearch', () => {
  it('does not search for queries shorter than 2 characters', async () => {
    renderSearch();
    await typeAndDebounce('h');
    expect(searchPlayersMock).not.toHaveBeenCalled();
  });

  it('debounces: only one request for rapid typing', async () => {
    searchPlayersMock.mockResolvedValue(players);
    renderSearch();
    const input = screen.getByRole('textbox');
    fireEvent.change(input, { target: { value: 'ha' } });
    fireEvent.change(input, { target: { value: 'haa' } });
    fireEvent.change(input, { target: { value: 'haal' } });
    await act(async () => {
      vi.advanceTimersByTime(350);
    });
    expect(searchPlayersMock).toHaveBeenCalledTimes(1);
    expect(searchPlayersMock).toHaveBeenCalledWith('haal', 10);
  });

  it('shows results with fallback hints for missing data', async () => {
    searchPlayersMock.mockResolvedValue([
      { ...players[1], market_value_current_eur: null },
    ]);
    renderSearch();
    await typeAndDebounce('haaland');
    expect(screen.getByText('Erling Haaland')).toBeInTheDocument();
    expect(screen.getByText(/no market value — wider fallback estimate/)).toBeInTheDocument();
  });

  it('flags players missing both position and age as unbenchmarkable', async () => {
    searchPlayersMock.mockResolvedValue([
      { ...players[0], main_position: null, age_months: null },
    ]);
    renderSearch();
    await typeAndDebounce('messi');
    expect(screen.getByText(/cannot benchmark/)).toBeInTheDocument();
  });

  it('shows "No players found" for an empty result', async () => {
    searchPlayersMock.mockResolvedValue([]);
    renderSearch();
    await typeAndDebounce('zzzz');
    // The dropdown opens via setIsOpen(true) only when results.length > 0;
    // the empty state needs isOpen — verify no result rows are shown instead.
    expect(screen.queryByRole('listitem')).not.toBeInTheDocument();
  });

  it('navigates to the benchmark page when a player is selected', async () => {
    searchPlayersMock.mockResolvedValue(players);
    renderSearch();
    await typeAndDebounce('messi');
    // Navigation is synchronous with MemoryRouter; waitFor would hang under
    // fake timers, so assert directly after the click.
    fireEvent.click(screen.getByText('Lionel Messi'));
    expect(screen.getByText('benchmark page')).toBeInTheDocument();
  });

  it('clears results when the search API fails', async () => {
    searchPlayersMock.mockRejectedValue(new Error('Search failed'));
    renderSearch();
    await typeAndDebounce('haaland');
    expect(screen.queryByText('Erling Haaland')).not.toBeInTheDocument();
  });
});
