import { Link } from 'react-router-dom';
import { GitBranch, Send, User, Mail } from 'lucide-react';

export function Footer() {
  const currentYear = new Date().getFullYear();

  const links = {
    product: [
      { label: 'Reconstruction', href: '/reconstruct' },
      { label: 'File Upload', href: '/upload' },
      { label: 'API Docs', href: '/docs' },
      { label: 'Pricing', href: '/pricing' },
    ],
    company: [
      { label: 'About', href: '/about' },
      { label: 'Blog', href: '/blog' },
      { label: 'Careers', href: '/careers' },
      { label: 'Contact', href: '/contact' },
    ],
    resources: [
      { label: 'Documentation', href: '/docs' },
      { label: 'API Reference', href: '/api-reference' },
      { label: 'Status', href: '/status' },
      { label: 'Changelog', href: '/changelog' },
    ],
    legal: [
      { label: 'Privacy', href: '/privacy' },
      { label: 'Terms', href: '/terms' },
      { label: 'Security', href: '/security' },
      { label: 'Cookies', href: '/cookies' },
    ],
  };

  const socialLinks = [
    { icon: GitBranch, href: 'https://github.com', label: 'GitHub' },
    { icon: Send, href: 'https://twitter.com', label: 'Twitter' },
    { icon: User, href: 'https://linkedin.com', label: 'LinkedIn' },
    { icon: Mail, href: 'mailto:hello@pathsnap.dev', label: 'Email' },
  ];

  return (
    <footer className="border-t border-border-primary bg-bg-secondary/50">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-12 lg:py-16">
        <div className="xl:grid xl:grid-cols-3 xl:gap-8">
          {/* Brand */}
          <div className="space-y-6">
            <Link to="/" className="flex items-center gap-2 text-text-primary hover:opacity-80 transition-opacity" aria-label="PATHSNAP Home">
              <svg className="w-8 h-8 text-accent-primary" viewBox="0 0 32 32" fill="none" aria-hidden="true">
                <circle cx="16" cy="16" r="14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                <path d="M16 6v12M10 12h12M6 16h20" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
              <span className="font-semibold text-xl tracking-tight">PATHSNAP</span>
            </Link>
            <p className="text-text-secondary text-sm max-w-xs leading-relaxed">
              AI-powered GPS trajectory reconstruction and map matching.
              Transform noisy GPS tracks into accurate road routes.
            </p>
            <div className="flex gap-4">
              {socialLinks.map((item) => (
                <a
                  key={item.label}
                  href={item.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="p-2 rounded-lg text-text-muted hover:text-text-primary hover:bg-bg-tertiary transition-all duration-150"
                  aria-label={item.label}
                >
                  <item.icon className="w-5 h-5" aria-hidden="true" />
                </a>
              ))}
            </div>
          </div>

          {/* Navigation Links */}
          <div className="mt-10 grid grid-cols-2 gap-8 xl:col-span-2 xl:mt-0">
            {Object.entries(links).map(([category, items]) => (
              <div key={category}>
                <h3 className="text-text-primary font-semibold text-sm uppercase tracking-wider mb-4">
                  {category.charAt(0).toUpperCase() + category.slice(1)}
                </h3>
                <ul className="space-y-3" role="list">
                  {items.map((item) => (
                    <li key={item.label}>
                      <Link
                        to={item.href}
                        className="text-text-secondary text-sm hover:text-text-primary transition-colors"
                      >
                        {item.label}
                      </Link>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>

        {/* Bottom */}
        <div className="mt-12 pt-8 border-t border-border-primary flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-text-muted text-sm">
            © {currentYear} PATHSNAP. All rights reserved.
          </p>
          <div className="flex items-center gap-6 text-sm text-text-muted">
            <span>Built with React, TypeScript, and Tailwind CSS</span>
            <span>v1.0.0</span>
          </div>
        </div>
      </div>
    </footer>
  );
}