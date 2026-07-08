import { Link } from 'react-router-dom';

export default function Navbar() {
  return (
    <nav className="bg-[#1e293b] text-white shadow-lg print:hidden">
      <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2">
          <span className="text-2xl">⚽</span>
          <span className="text-xl font-bold">Soccer Salary Benchmark</span>
        </Link>
        <div className="flex gap-6">
          <Link to="/" className="hover:text-blue-300 transition-colors">
            Home
          </Link>
          <Link to="/manual" className="hover:text-blue-300 transition-colors">
            Custom Player
          </Link>
          <Link to="/model" className="hover:text-blue-300 transition-colors">
            About the Model
          </Link>
        </div>
      </div>
    </nav>
  );
}
