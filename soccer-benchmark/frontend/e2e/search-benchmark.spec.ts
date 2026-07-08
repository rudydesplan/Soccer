import { test, expect } from '@playwright/test';
import { mockApi, benchmarkResult } from './fixtures';

test.describe('Search → benchmark flow', () => {
  test.beforeEach(async ({ page }) => {
    await mockApi(page);
  });

  test('home page renders the hero and search box', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByRole('heading', { name: 'Soccer Salary Benchmark' })).toBeVisible();
    await expect(page.getByPlaceholder(/Search for a player/)).toBeVisible();
    await expect(page.getByRole('link', { name: /Benchmark a custom player/ })).toBeVisible();
  });

  test('searching shows results and selecting navigates to the benchmark', async ({ page }) => {
    await page.goto('/');
    await page.getByPlaceholder(/Search for a player/).fill('haaland');

    const result = page.getByText('Erling Haaland');
    await expect(result).toBeVisible();
    await result.click();

    await expect(page).toHaveURL(/\/benchmark\/2/);
    await expect(page.getByRole('heading', { name: 'Erling Haaland' })).toBeVisible();
    await expect(page.getByText('⬆️ Overpaid')).toBeVisible();
    await expect(page.getByText('Confidence: LOW')).toBeVisible();
    await expect(page.getByText('Salary Range')).toBeVisible();
    await expect(page.getByRole('cell', { name: 'Kylian Mbappe' })).toBeVisible();
  });

  test('range width toggle re-queries the API with wide', async ({ page }) => {
    await page.goto('/benchmark/2');
    await expect(page.getByText('⬆️ Overpaid')).toBeVisible();

    const wideRequest = page.waitForRequest(
      (req) => req.url().includes('/api/benchmark') && req.postDataJSON()?.range_width === 'wide'
    );
    await page.getByRole('button', { name: 'Wide (80%)' }).click();
    await wideRequest;
  });

  test('explanation panel loads SHAP contributions on demand', async ({ page }) => {
    await page.goto('/benchmark/2');
    await expect(page.getByText('Why this estimate?')).toBeVisible();

    const explainRequest = page.waitForRequest(
      (req) => req.url().includes('/api/benchmark/explain') && req.postDataJSON()?.player_id === 2
    );
    await page.getByRole('button', { name: 'Explain this estimate' }).click();
    await explainRequest;

    await expect(page.getByText(/A typical player's estimate is/)).toBeVisible();
    await expect(page.getByText('×15.7')).toBeVisible();
    await expect(page.getByText('−11%')).toBeVisible();
  });

  test('explanation error is surfaced with a retry button', async ({ page }) => {
    await page.route('**/api/benchmark/explain', (route) =>
      route.fulfill({ status: 500, json: { detail: 'Internal server error' } })
    );
    await page.goto('/benchmark/2');
    await page.getByRole('button', { name: 'Explain this estimate' }).click();
    await expect(page.getByText('Internal server error')).toBeVisible();
    await expect(page.getByRole('button', { name: 'Explain this estimate' })).toBeVisible();
  });

  test('view all comparables page lists every comparable', async ({ page }) => {
    await page.goto('/benchmark/2');
    await page.getByRole('link', { name: /View all comparables/ }).click();

    await expect(page).toHaveURL(/\/comparables\/2/);
    await expect(page.getByText('Comparable Players for Erling Haaland')).toBeVisible();
    await expect(page.getByRole('cell', { name: 'Kylian Mbappe' })).toBeVisible();
    await expect(page.getByRole('cell', { name: 'Harry Kane' })).toBeVisible();
  });

  test('fallback model shows the warning banner', async ({ page }) => {
    await page.route('**/api/benchmark', (route) =>
      route.fulfill({
        json: {
          ...benchmarkResult,
          model_used: 'no_mv',
          benchmark_warning: 'Fallback model used. Expect a wider, less precise range.',
        },
      })
    );
    await page.goto('/benchmark/2');
    await expect(page.getByText('Limited reliability')).toBeVisible();
    await expect(page.getByText('Fallback model (no market value)')).toBeVisible();
  });

  test('API error shows the error state with a way back', async ({ page }) => {
    await page.route('**/api/benchmark', (route) =>
      route.fulfill({ status: 404, json: { detail: 'Player 99999 not found' } })
    );
    await page.goto('/benchmark/99999');
    await expect(page.getByText('Player 99999 not found')).toBeVisible();
    await expect(page.getByRole('link', { name: /Back to search/ })).toBeVisible();
  });
});
