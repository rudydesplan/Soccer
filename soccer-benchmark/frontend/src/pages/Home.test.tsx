import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import Home from './Home';

describe('Home', () => {
  it('renders the title, search box, and custom player link', () => {
    render(
      <MemoryRouter>
        <Home />
      </MemoryRouter>
    );
    expect(screen.getByRole('heading', { name: 'Soccer Salary Benchmark' })).toBeInTheDocument();
    expect(screen.getByRole('textbox')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Benchmark a custom player/ })).toHaveAttribute(
      'href',
      '/manual'
    );
    expect(screen.getByText('Salary Benchmarking')).toBeInTheDocument();
    expect(screen.getByText('Comparable Players')).toBeInTheDocument();
    expect(screen.getByText('Instant Analysis')).toBeInTheDocument();
  });
});
