import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import PlayerCard from './PlayerCard';

describe('PlayerCard', () => {
  it('renders all player fields', () => {
    render(
      <PlayerCard
        name="Erling Haaland"
        position="Centre-Forward"
        competition="GB1"
        age_months={290}
        market_value={170_000_000}
      />
    );
    expect(screen.getByText('Erling Haaland')).toBeInTheDocument();
    expect(screen.getByText('Centre-Forward')).toBeInTheDocument();
    expect(screen.getByText('GB1')).toBeInTheDocument();
    expect(screen.getByText('290 months (24.2 yrs)')).toBeInTheDocument();
    expect(screen.getByText('€170.0M')).toBeInTheDocument();
  });

  it('shows N/A for missing age and market value', () => {
    render(
      <PlayerCard name="Unknown" position="N/A" competition="N/A" age_months={null} market_value={null} />
    );
    expect(screen.getAllByText('N/A').length).toBeGreaterThanOrEqual(2);
  });
});
