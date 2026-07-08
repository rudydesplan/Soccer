import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import SalaryRangeChart from './SalaryRangeChart';

describe('SalaryRangeChart', () => {
  it('renders the low / median / high labels', () => {
    render(
      <SalaryRangeChart
        low={14_000_000}
        median={20_000_000}
        high={28_000_000}
        actual={31_500_000}
        status="OVERPAID"
      />
    );
    expect(screen.getByText('Low: €14.0M')).toBeInTheDocument();
    expect(screen.getByText('Median: €20.0M')).toBeInTheDocument();
    expect(screen.getByText('High: €28.0M')).toBeInTheDocument();
  });

  it('renders without an actual salary', () => {
    render(
      <SalaryRangeChart low={1_000_000} median={2_000_000} high={3_000_000} actual={null} status="UNKNOWN" />
    );
    expect(screen.getByText('Median: €2.0M')).toBeInTheDocument();
  });
});
