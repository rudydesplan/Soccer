import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Benchmark from './pages/Benchmark';
import Comparables from './pages/Comparables';
import ManualBenchmark from './pages/ManualBenchmark';
import './App.css';

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-50">
        <Navbar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/manual" element={<ManualBenchmark />} />
          <Route path="/benchmark/:playerId" element={<Benchmark />} />
          <Route path="/comparables/:playerId" element={<Comparables />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;
