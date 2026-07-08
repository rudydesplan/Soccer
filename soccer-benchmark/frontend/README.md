# Soccer Salary Benchmark — Frontend

React + TypeScript + Vite + Tailwind CSS interface for the salary benchmark tool.

## Development

```bash
npm install
npm run dev
```

Requires the backend running on port 8001 (Vite proxies `/api` requests).

## Build

```bash
npm run build
```

## Tests

Unit/component tests (Vitest + Testing Library, jsdom):

```bash
npm test              # run once
npm run test:watch    # watch mode
npm run test:coverage # with coverage
```

End-to-end tests (Playwright, API mocked — no backend needed):

```bash
npx playwright install chromium   # once
npm run test:e2e
npm run test:e2e:ui               # interactive UI mode
```

The Playwright config starts the Vite dev server automatically; all `/api`
calls are intercepted with `page.route()`, so no trained models or Python
environment are required.

## Stack

- React 19
- TypeScript 6
- Vite 8
- Tailwind CSS 4
- Recharts (salary range visualisation)
- React Router (client-side routing)
