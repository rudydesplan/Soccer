import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  formatEur,
  formatAge,
  searchPlayers,
  getPlayer,
  getBenchmarkById,
  getBenchmarkOptions,
  getManualBenchmark,
  getExplanationById,
  getManualExplanation,
} from './api';
import { players, benchmarkResult, benchmarkOptions, explanationResult } from '../test/fixtures';

function mockFetchOnce(body: unknown, ok = true, status = 200) {
  const res = {
    ok,
    status,
    json: () => Promise.resolve(body),
  } as Response;
  vi.mocked(globalThis.fetch).mockResolvedValueOnce(res);
  return res;
}

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn());
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('formatEur', () => {
  it('formats millions with one decimal', () => {
    expect(formatEur(31_500_000)).toBe('€31.5M');
  });

  it('formats thousands without decimals', () => {
    expect(formatEur(850_000)).toBe('€850K');
  });

  it('formats small values as plain euros', () => {
    expect(formatEur(999)).toBe('€999');
  });

  it('returns N/A for null, undefined, and NaN', () => {
    expect(formatEur(null)).toBe('N/A');
    expect(formatEur(undefined)).toBe('N/A');
    expect(formatEur(Number.NaN)).toBe('N/A');
  });
});

describe('formatAge', () => {
  it('shows months and years', () => {
    expect(formatAge(300)).toBe('300 months (25.0 yrs)');
  });

  it('returns N/A for null and undefined', () => {
    expect(formatAge(null)).toBe('N/A');
    expect(formatAge(undefined)).toBe('N/A');
  });
});

describe('searchPlayers', () => {
  it('encodes the query and returns players', async () => {
    mockFetchOnce(players);
    const result = await searchPlayers('haaland & co', 10);
    expect(globalThis.fetch).toHaveBeenCalledWith(
      '/api/players/search?q=haaland%20%26%20co&limit=10'
    );
    expect(result).toEqual(players);
  });

  it('throws the API detail message on error', async () => {
    mockFetchOnce({ detail: 'Player pool not available' }, false, 503);
    await expect(searchPlayers('xx')).rejects.toThrow('Player pool not available');
  });

  it('falls back to a generic message when the error body is not JSON', async () => {
    vi.mocked(globalThis.fetch).mockResolvedValueOnce({
      ok: false,
      status: 502,
      json: () => Promise.reject(new Error('not json')),
    } as unknown as Response);
    await expect(searchPlayers('xx')).rejects.toThrow('Search failed (HTTP 502)');
  });
});

describe('getPlayer', () => {
  it('fetches by id', async () => {
    mockFetchOnce(players[0]);
    const player = await getPlayer(0);
    expect(globalThis.fetch).toHaveBeenCalledWith('/api/players/0');
    expect(player.player_name).toBe('Lionel Messi');
  });

  it('throws detail on 404', async () => {
    mockFetchOnce({ detail: 'Player not found' }, false, 404);
    await expect(getPlayer(99999)).rejects.toThrow('Player not found');
  });
});

describe('getBenchmarkById', () => {
  it('POSTs player_id with defaults', async () => {
    mockFetchOnce(benchmarkResult);
    await getBenchmarkById(2);
    expect(globalThis.fetch).toHaveBeenCalledWith('/api/benchmark', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ player_id: 2, range_width: 'normal', full_comparables: false }),
    });
  });

  it('passes range width and full comparables', async () => {
    mockFetchOnce(benchmarkResult);
    await getBenchmarkById(2, 'wide', true);
    const body = JSON.parse(
      (vi.mocked(globalThis.fetch).mock.calls[0][1] as RequestInit).body as string
    );
    expect(body).toEqual({ player_id: 2, range_width: 'wide', full_comparables: true });
  });
});

describe('getBenchmarkOptions', () => {
  it('fetches options', async () => {
    mockFetchOnce(benchmarkOptions);
    const options = await getBenchmarkOptions();
    expect(globalThis.fetch).toHaveBeenCalledWith('/api/players/options');
    expect(options.positions).toContain('Centre-Forward');
  });
});

describe('getExplanationById', () => {
  it('POSTs the player id to the explain endpoint', async () => {
    mockFetchOnce(explanationResult);
    const result = await getExplanationById(2);
    expect(globalThis.fetch).toHaveBeenCalledWith('/api/benchmark/explain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ player_id: 2 }),
    });
    expect(result.features[0].label).toBe('Market value');
  });

  it('throws the API detail on error', async () => {
    mockFetchOnce({ detail: 'Player id 99999 not found in player_pool.csv' }, false, 404);
    await expect(getExplanationById(99999)).rejects.toThrow('not found');
  });
});

describe('getManualExplanation', () => {
  it('POSTs the manual fields to the explain endpoint', async () => {
    mockFetchOnce(explanationResult);
    await getManualExplanation({ main_position: 'Centre-Forward', age_months: 300 });
    const body = JSON.parse(
      (vi.mocked(globalThis.fetch).mock.calls[0][1] as RequestInit).body as string
    );
    expect(body).toEqual({ main_position: 'Centre-Forward', age_months: 300 });
  });
});

describe('getManualBenchmark', () => {
  it('POSTs manual fields plus range width', async () => {
    mockFetchOnce(benchmarkResult);
    await getManualBenchmark(
      { main_position: 'Centre-Forward', age_months: 300, market_value_current_eur: 50_000_000 },
      'wide'
    );
    const body = JSON.parse(
      (vi.mocked(globalThis.fetch).mock.calls[0][1] as RequestInit).body as string
    );
    expect(body).toEqual({
      main_position: 'Centre-Forward',
      age_months: 300,
      market_value_current_eur: 50_000_000,
      range_width: 'wide',
    });
  });

  it('surfaces the backend detail for a 400', async () => {
    mockFetchOnce({ detail: 'age_months is required for manual benchmarks' }, false, 400);
    await expect(
      getManualBenchmark({ main_position: 'Centre-Forward', age_months: 0 })
    ).rejects.toThrow('age_months is required for manual benchmarks');
  });
});
