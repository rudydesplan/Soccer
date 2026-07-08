import { describe, it, expect } from 'vitest';
import { render, screen, within, fireEvent } from '@testing-library/react';
import ComparablesTable from './ComparablesTable';
import { comparables } from '../test/fixtures';

function rowNames() {
  const rows = screen.getAllByRole('row').slice(1); // skip header
  return rows.map((row) => within(row).getAllByRole('cell')[0].textContent);
}

describe('ComparablesTable', () => {
  it('renders one row per player with formatted values', () => {
    render(<ComparablesTable players={comparables} />);
    expect(screen.getByText('Kylian Mbappe')).toBeInTheDocument();
    expect(screen.getByText('Harry Kane')).toBeInTheDocument();
    expect(screen.getByText('93.0%')).toBeInTheDocument();
    expect(screen.getByText('€180.0M')).toBeInTheDocument();
    expect(screen.getByText('€24.0M')).toBeInTheDocument();
  });

  it('sorts by similarity descending by default and toggles on click', () => {
    render(<ComparablesTable players={comparables} />);
    expect(rowNames()).toEqual(['Kylian Mbappe', 'Harry Kane']);

    fireEvent.click(screen.getByText(/Similarity/));
    expect(rowNames()).toEqual(['Harry Kane', 'Kylian Mbappe']);
  });

  it('renders an empty table without crashing', () => {
    render(<ComparablesTable players={[]} />);
    expect(screen.getAllByRole('row')).toHaveLength(1); // header only
  });
});
