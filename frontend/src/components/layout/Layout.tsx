import { Routes, Route } from 'react-router-dom';
import { Header } from './Header';
import { Footer } from './Footer';
import { Home } from '../../pages/Home';
import { Reconstruct } from '../../pages/Reconstruct';
import { Upload } from '../../pages/Upload';
import { Evaluation } from '../../pages/Evaluation';
import { Settings } from '../../pages/Settings';

export function Layout() {
  return (
    <div className="flex min-h-screen flex-col">
      <Header />
      <main className="flex-1 pt-16">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/reconstruct" element={<Reconstruct />} />
          <Route path="/upload" element={<Upload />} />
          <Route path="/evaluation" element={<Evaluation />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </main>
      <Footer />
    </div>
  );
}