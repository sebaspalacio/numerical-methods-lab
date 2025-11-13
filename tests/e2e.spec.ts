import { test, expect } from '@playwright/test';

test('tabs and method select are interactive', async ({ page }) => {
  await page.goto('http://localhost:8080/index.html');
  const method = page.locator('#method');

  await page.click('#tabRoot');
  await expect(method).toHaveCount(1);
  const rootCount = await method.locator('option').count();
  expect(rootCount).toBeGreaterThan(3);

  await page.click('#tabDirect');
  const directCount = await method.locator('option').count();
  expect(directCount).toBeGreaterThan(3);
  expect(directCount).not.toBe(rootCount);

  await page.click('#tabIter');
  const iterCount = await method.locator('option').count();
  expect(iterCount).toBeGreaterThan(1);
});

test('bisection runs and invalid char error is explicit', async ({ page }) => {
  await page.goto('http://localhost:8080/index.html');
  await page.click('#tabRoot');
  await page.selectOption('#method', { value: 'BISE' });
  await page.click('#runBtn');
  await expect(page.locator('#status')).toContainText('OK');
  await expect(page.locator('#iters table')).toBeVisible();

  const fx = page.locator('#fxInput');
  await fx.fill('x×x - 7*x + 6');
  await page.click('#runBtn');
  await expect(page.locator('#status')).toContainText('U+00D7');
});

test('LU simple prints L and U', async ({ page }) => {
  await page.goto('http://localhost:8080/index.html');
  await page.click('#tabDirect');
  await page.selectOption('#method', { value: 'LUS' });
  await page.click('#runBtn');
  await expect(page.locator('#outputs')).toContainText('L:');
  await expect(page.locator('#outputs')).toContainText('U:');
});
