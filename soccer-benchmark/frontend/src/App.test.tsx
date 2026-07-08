import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import App from './App';

describe('App', () => {
  it('renders the navbar and the home page on /', () => {
    window.history.pushState({}, '', '/');
    render(<App />);
    expect(screen.getByRole('heading', { name: 'Soccer Salary Benchmark' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Custom Player' })).toBeInTheDocument();
  });
});
