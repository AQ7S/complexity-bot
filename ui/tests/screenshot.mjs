import { chromium } from '@playwright/test';
import { mkdirSync } from 'node:fs';
import { resolve } from 'node:path';

const ROUTES = [
  ['/',          'command'],
  ['/trace',     'trace'],
  ['/charts',    'charts'],
  ['/scanner',   'scanner'],
  ['/journal',   'journal'],
  ['/ai',        'ai'],
  ['/settings',  'settings'],
];

const OUT = resolve('screenshots');
mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const page = await ctx.newPage();

const errors = [];
page.on('pageerror', (e) => errors.push(['pageerror', e.message]));
page.on('console', (msg) => { if (msg.type() === 'error') errors.push(['console.error', msg.text()]); });

for (const [path, name] of ROUTES) {
  errors.length = 0;
  const url = `http://localhost:4173/#${path}`;
  await page.goto(url, { waitUntil: 'networkidle' });
  await page.waitForTimeout(400);
  const file = `${OUT}/${name}.png`;
  await page.screenshot({ path: file, fullPage: true });
  console.log(`OK ${path.padEnd(12)} -> ${file}  errors=${errors.length}`);
  for (const [k, m] of errors) console.log(`   [${k}] ${m.slice(0, 200)}`);
}

// Bonus: load demo signals on /trace and re-shoot
await page.goto(`http://localhost:4173/#/trace`, { waitUntil: 'networkidle' });
const btn = await page.$('[data-testid="load-demo"]');
if (btn) {
  await btn.click();
  await page.waitForTimeout(500);
  await page.screenshot({ path: `${OUT}/trace-demo.png`, fullPage: true });
  console.log('OK /trace [demo loaded] -> screenshots/trace-demo.png');
  const cards = await page.$$('[data-testid="decision-card"]');
  if (cards[0]) {
    const toggle = await cards[0].$('[data-testid="toggle-reasoning"]');
    if (toggle) {
      await toggle.click();
      await page.waitForTimeout(300);
      await page.screenshot({ path: `${OUT}/trace-expanded.png`, fullPage: true });
      console.log('OK /trace [expanded] -> screenshots/trace-expanded.png');
    }
  }
}

await browser.close();
