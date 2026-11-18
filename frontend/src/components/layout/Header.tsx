import { Link, useLocation } from 'react-router-dom';
import { cn } from '../../lib/utils';
import {
  LayoutDashboard,
  MapPin,
  Upload,
  Settings,
  GitBranch,
  Send,
  Sun,
  Moon,
  Menu,
  X,
  Target,
} from 'lucide-react';
import { useState, useEffect } from 'react';

export function Header() {
  const location = useLocation();
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isDark, setIsDark] = useState(true);

  useEffect(() => {
    const saved = localStorage.getItem('theme');
    if (saved) {
      setIsDark(saved === 'dark');
    } else {
      setIsDark(window.matchMedia('(prefers-color-scheme: dark)').matches);
    }
  }, []);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', isDark);
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
  }, [isDark]);

  const navItems = [
    { path: '/', label: 'Home', icon: LayoutDashboard },
    { path: '/reconstruct', label: 'Reconstruct', icon: MapPin },
    { path: '/upload', label: 'Upload', icon: Upload },
    { path: '/evaluation', label: 'Evaluation', icon: Target },
    { path: '/settings', label: 'Settings', icon: Settings },
  ];

  return (
    <header className="fixed top-0 left-0 right-0 z-50 glass-strong border-b border-border-primary">
      <nav className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8" aria-label="Main navigation">
        <div className="flex h-16 items-center justify-between">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-2 text-text-primary hover:opacity-80 transition-opacity" aria-label="PATHSNAP Home">
            <svg className="w-8 h-8 text-accent-primary" viewBox="0 0 32 32" fill="none" aria-hidden="true">
              <circle cx="16" cy="16" r="14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M16 6v12M10 12h12M6 16h20" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            </svg>
            <span className="font-semibold text-xl tracking-tight">PATHSNAP</span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex md:items-center md:gap-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path || (item.path !== '/' && location.pathname.startsWith(item.path));
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={cn(
                    'flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-all duration-150',
                    isActive
                      ? 'bg-bg-tertiary text-accent-primary'
                      : 'text-text-secondary hover:text-text-primary hover:bg-bg-tertiary'
                  )}
                  aria-current={isActive ? 'page' : undefined}
                >
                  <Icon className="w-4 h-4" aria-hidden="true" />
                  {item.label}
                </Link>
              );
            })}
          </div>

          {/* Right side actions */}
          <div className="flex items-center gap-3">
            {/* Theme toggle */}
            <button
              onClick={() => setIsDark(!isDark)}
              className="p-2 rounded-lg text-text-secondary hover:text-text-primary hover:bg-bg-tertiary transition-colors"
              aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
            >
              {isDark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </button>

            {/* GitHub link */}
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 rounded-lg text-text-secondary hover:text-text-primary hover:bg-bg-tertiary transition-colors"
              aria-label="GitHub Repository"
            >
              <GitBranch className="w-5 h-5" />
            </a>

            {/* Twitter link */}
            <a
              href="https://twitter.com"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 rounded-lg text-text-secondary hover:text-text-primary hover:bg-bg-tertiary transition-colors"
              aria-label="Twitter"
            >
              <Send className="w-5 h-5" />
            </a>

            {/* Mobile menu button */}
            <button
              className="md:hidden p-2 rounded-lg text-text-secondary hover:text-text-primary hover:bg-bg-tertiary transition-colors"
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              aria-expanded={isMenuOpen}
              aria-controls="mobile-menu"
              aria-label={isMenuOpen ? 'Close menu' : 'Open menu'}
            >
              {isMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {isMenuOpen && (
          <div id="mobile-menu" className="md:hidden py-4 border-t border-border-primary animate-slide-down">
            <div className="flex flex-col gap-1">
              {navItems.map((item) => {
                const Icon = item.icon;
                const isActive = location.pathname === item.path || (item.path !== '/' && location.pathname.startsWith(item.path));
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    onClick={() => setIsMenuOpen(false)}
                    className={cn(
                      'flex items-center gap-3 px-3 py-2.5 rounded-md text-base font-medium transition-all duration-150',
                      isActive
                        ? 'bg-bg-tertiary text-accent-primary'
                        : 'text-text-secondary hover:text-text-primary hover:bg-bg-tertiary'
                    )}
                    aria-current={isActive ? 'page' : undefined}
                  >
                    <Icon className="w-5 h-5" aria-hidden="true" />
                    {item.label}
                  </Link>
                );
              })}
            </div>
          </div>
        )}
      </nav>
    </header>
  );
}