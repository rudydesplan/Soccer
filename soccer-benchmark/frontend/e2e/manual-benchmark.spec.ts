import { test, expect } from '@playwright/test';
import { mockApi, benchmarkResult } from './fixtures';

test.describe('Custom player (manual benchmark) flow', () => {
  test.beforeEach(async ({ page }) => {
    await mockApi(page);
  });

  test('navbar links to the custom player form', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('link', { name: 'Custom Player', exact: true }).click();
    await expect(page).toHaveURL(/\/manual/);
    await expect(page.getByRole('heading', { name: 'Benchmark a Custom Player' })).toBeVisible();
  });

  test('form loads options and runs a benchmark', async ({ page }) => {
    await page.route('**/api/benchmark', (route) =>
      route.fulfill({
        json: {
          ...benchmarkResult,
          player_name: 'custom_player',
          salary_status: 'UNKNOWN',
          actual_salary_eur: null,
          salary_percentile: null,
        },
      })
    );

    await page.goto('/manual');
    await page.getByLabel(/Position \*/).selectOption('Centre-Forward');
    await page.getByLabel(/Age \(years\) \*/).fill('25');
    await page.getByLabel(/League/).selectOption('GB1');
    await page.getByLabel(/Market value \(EUR\)/).fill('50000000');

    const benchmarkRequest = page.waitForRequest((req) =>
      req.url().includes('/api/benchmark')
    );
    await page.getByRole('button', { name: 'Run benchmark' }).click();
    const req = await benchmarkRequest;

    expect(req.postDataJSON()).toMatchObject({
      main_position: 'Centre-Forward',
      age_months: 300,
      competition_id: 'GB1',
      competition_country: 'England',
      market_value_current_eur: 50_000_000,
    });

    await expect(page.getByText('Unknown salary')).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Salary Range' })).toBeVisible();
    await expect(page.getByRole('cell', { name: 'Kylian Mbappe' })).toBeVisible();
  });

  test('custom player explanation reuses the submitted input', async ({ page }) => {
    await page.goto('/manual');
    await page.getByLabel(/Position \*/).selectOption('Centre-Forward');
    await page.getByLabel(/Age \(years\) \*/).fill('25');
    await page.getByRole('button', { name: 'Run benchmark' }).click();
    await expect(page.getByRole('heading', { name: 'Salary Range' })).toBeVisible();

    const explainRequest = page.waitForRequest(
      (req) =>
        req.url().includes('/api/benchmark/explain') &&
        req.postDataJSON()?.main_position === 'Centre-Forward'
    );
    await page.getByRole('button', { name: 'Explain this estimate' }).click();
    await explainRequest;
    await expect(page.getByText('×15.7')).toBeVisible();
  });

  test('salary verdict appears when the actual salary is provided', async ({ page }) => {
    await page.goto('/manual');
    await page.getByLabel(/Position \*/).selectOption('Centre-Forward');
    await page.getByLabel(/Age \(years\) \*/).fill('24');
    await page.getByLabel(/Current annual salary/).fill('31500000');
    await page.getByRole('button', { name: 'Run benchmark' }).click();

    await expect(page.getByText('⬆️ Overpaid')).toBeVisible();
  });

  test('backend validation errors are surfaced', async ({ page }) => {
    await page.route('**/api/benchmark', (route) =>
      route.fulfill({
        status: 400,
        json: { detail: 'age_months is required for manual benchmarks' },
      })
    );
    await page.goto('/manual');
    await page.getByLabel(/Position \*/).selectOption('Centre-Forward');
    await page.getByLabel(/Age \(years\) \*/).fill('25');
    await page.getByRole('button', { name: 'Run benchmark' }).click();

    await expect(page.getByText('age_months is required for manual benchmarks')).toBeVisible();
  });

  test('options endpoint failure shows an error in the form', async ({ page }) => {
    await page.route('**/api/players/options', (route) =>
      route.fulfill({ status: 503, json: { detail: 'Player pool not available' } })
    );
    await page.goto('/manual');
    await expect(page.getByText('Player pool not available')).toBeVisible();
  });
});
