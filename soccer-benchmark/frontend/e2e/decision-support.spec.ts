import { test, expect } from '@playwright/test';
import { mockApi } from './fixtures';

test.describe('Model card page', () => {
  test.beforeEach(async ({ page }) => {
    await mockApi(page);
  });

  test('navbar link opens the model card with all sections', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('link', { name: 'About the Model' }).click();

    await expect(page).toHaveURL(/\/model/);
    await expect(page.getByRole('heading', { name: 'About the model' })).toBeVisible();
    await expect(page.getByRole('heading', { name: 'How accurate is it?' })).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Limitations' })).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Fairness & responsible use' })).toBeVisible();
    // Metrics rendered in business language
    await expect(page.getByTestId('metric-cards')).toContainText('75%');
    await expect(page.getByTestId('metric-cards')).toContainText('±31%');
    // Importance bars
    await expect(page.getByTestId('importance-bars')).toContainText('Market value');
  });

  test('backend failure shows the error message', async ({ page }) => {
    await page.route('**/api/meta/model-card', (route) =>
      route.fulfill({ status: 503, json: { detail: 'Model artifact not found' } })
    );
    await page.goto('/model');
    await expect(page.getByText('Model artifact not found')).toBeVisible();
  });
});

test.describe('Printable report', () => {
  test.beforeEach(async ({ page }) => {
    await mockApi(page);
  });

  test('benchmark page links to a complete report', async ({ page }) => {
    await page.goto('/benchmark/2');
    await page.getByRole('link', { name: 'Printable report' }).click();

    await expect(page).toHaveURL(/\/report\/2/);
    await expect(page.getByRole('heading', { name: 'Salary Benchmark Report' })).toBeVisible();
    // Result summary
    await expect(page.getByText('Overpaid', { exact: true })).toBeVisible();
    await expect(page.getByText('€20.0M', { exact: true })).toBeVisible();
    // SHAP explanation is fetched automatically (no click needed)
    await expect(page.getByTestId('report-explanation')).toContainText('Market value');
    await expect(page.getByTestId('report-explanation')).toContainText('×15.7');
    // Comparables included
    await expect(page.getByTestId('report-comparables')).toContainText('Kylian Mbappe');
    // Print button present
    await expect(page.getByRole('button', { name: 'Print / Save as PDF' })).toBeVisible();
  });

  test('report fails loudly when the explanation cannot be computed', async ({ page }) => {
    await page.route('**/api/benchmark/explain', (route) =>
      route.fulfill({ status: 500, json: { detail: 'Internal server error' } })
    );
    await page.goto('/report/2');
    await expect(page.getByText('Internal server error')).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Salary Benchmark Report' })).toBeHidden();
  });
});
