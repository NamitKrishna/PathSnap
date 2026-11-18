import { Link } from 'react-router-dom';
import { Button } from '../components/ui/Button';
import { Card, CardContent } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import {
  ArrowRight,
  MapPin,
  Cpu,
  Zap,
  Shield,
  GitBranch,
  Globe,
  Sparkles,
  BarChart3,
  Layers,
  Navigation,
  Target,
} from 'lucide-react';

const features = [
  {
    icon: MapPin,
    title: 'Precision Map Matching',
    description: 'Advanced HMM/Viterbi algorithm matches GPS observations to road networks with sub-meter accuracy.',
    highlight: 'Sub-meter accuracy',
  },
  {
    icon: Cpu,
    title: 'ML-Powered Scoring',
    description: 'Gradient boosting model evaluates candidate road segments using spatial, temporal, and kinematic features.',
    highlight: 'XGBoost/LightGBM',
  },
  {
    icon: Zap,
    title: 'Real-time Processing',
    description: 'Optimized pipeline processes thousands of GPS points per second with minimal latency.',
    highlight: '<100ms latency',
  },
  {
    icon: Shield,
    title: 'Robust Validation',
    description: 'Comprehensive input validation for CSV, GPX, and JSON formats with detailed error reporting.',
    highlight: 'Multi-format support',
  },
  {
    icon: GitBranch,
    title: 'Topology-Aware Routing',
    description: 'NetworkX-powered road graph ensures reconstructed routes follow valid road connectivity.',
    highlight: 'Graph-based routing',
  },
  {
    icon: Globe,
    title: 'Global Coverage',
    description: 'Works with OpenStreetMap data worldwide. Load custom road networks via GeoPackage or GraphML.',
    highlight: 'OSM + custom data',
  },
];

const stats = [
  { value: '99.2%', label: 'Match Accuracy' },
  { value: '<50ms', label: 'Avg Latency' },
  { value: '10K+', label: 'Points/sec' },
  { value: '195+', label: 'Countries' },
];

const pipelineSteps = [
  {
    step: '01',
    title: 'Preprocessing',
    description: 'Clean GPS data, project to Web Mercator, extract kinematic features (speed, heading, acceleration)',
    icon: Layers,
  },
  {
    step: '02',
    title: 'Candidate Generation',
    description: 'Find nearby road segments within configurable radius using spatial indexing (R-tree)',
    icon: Navigation,
  },
  {
    step: '03',
    title: 'ML Scoring',
    description: 'Gradient boosting model scores each candidate using distance, heading diff, road class, speed features',
    icon: Cpu,
  },
  {
    step: '04',
    title: 'HMM/Viterbi Decoding',
    description: 'Log-space Viterbi algorithm finds most probable path through candidate sequence with transition penalties',
    icon: Target,
  },
  {
    step: '05',
    title: 'Geometry Reconstruction',
    description: 'Resolve arc IDs to full route geometry, compute distance, export as GeoJSON for visualization',
    icon: BarChart3,
  },
];

export function Home() {
  return (
    <div className="flex flex-col">
      {/* Hero Section */}
      <section className="relative overflow-hidden py-20 lg:py-32">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--color-accent-primary)/5_0%,_transparent_70%)] pointer-events-none" />
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 relative">
          <div className="text-center max-w-4xl mx-auto">
            <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-accent-primary/10 border border-accent-primary/20 text-accent-primary text-sm font-medium mb-6 animate-fade-in">
              <Sparkles className="w-4 h-4" />
              <span>PATHSNAP v1.0 — Production Ready</span>
            </div>

            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight text-text-primary mb-6 animate-slide-up">
              AI-Powered GPS
              <br />
              <span className="gradient-text">Route Reconstruction</span>
            </h1>

            <p className="text-lg sm:text-xl text-text-secondary max-w-2xl mx-auto mb-10 animate-slide-up" style={{ animationDelay: '100ms' }}>
              Transform noisy GPS trajectories into precise road routes using
              state-of-the-art map matching, machine learning, and hidden Markov models.
            </p>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-16 animate-slide-up" style={{ animationDelay: '200ms' }}>
              <Button size="lg" asChild>
                <Link to="/reconstruct">
                  Try Reconstruction
                  <ArrowRight className="w-4 h-4" />
                </Link>
              </Button>
              <Button size="lg" variant="secondary" asChild>
                <Link to="/upload">
                  Upload GPS File
                  <ArrowRight className="w-4 h-4" />
                </Link>
              </Button>
            </div>

            {/* Trust badges */}
            <div className="flex flex-wrap items-center justify-center gap-4 text-sm text-text-muted animate-fade-in" style={{ animationDelay: '300ms' }}>
              <Badge variant="success" dot>Open Source</Badge>
              <Badge variant="info" dot>MIT Licensed</Badge>
              <Badge variant="default" dot>TypeScript</Badge>
              <Badge variant="default" dot>FastAPI</Badge>
              <Badge variant="default" dot>React 18</Badge>
            </div>
          </div>

          {/* Stats */}
          <div className="mt-16 grid grid-cols-2 lg:grid-cols-4 gap-6 max-w-4xl mx-auto animate-slide-up" style={{ animationDelay: '400ms' }}>
            {stats.map((stat) => (
              <div key={stat.label} className="text-center p-4 glass rounded-xl">
                <div className="text-3xl sm:text-4xl font-bold gradient-text mb-1">{stat.value}</div>
                <div className="text-text-secondary text-sm">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Pipeline Visualization */}
      <section className="py-20 lg:py-28 bg-bg-secondary/30">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-3xl mx-auto mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold text-text-primary mb-4">
              How It Works
            </h2>
            <p className="text-text-secondary text-lg">
              The PATHSNAP pipeline processes GPS observations through five stages,
              each designed for accuracy and performance.
            </p>
          </div>

          <div className="relative">
            {/* Connecting line */}
            <div className="hidden lg:block absolute left-1/2 top-0 bottom-0 w-0.5 bg-gradient-to-b from-accent-primary to-accent-tertiary -translate-x-1/2" />

            <div className="space-y-12 lg:space-y-16">
              {pipelineSteps.map((step, index) => (
                <div
                  key={step.step}
                  className={`relative flex flex-col lg:flex-row gap-8 items-start ${index % 2 === 0 ? 'lg:text-right' : ''} animate-slide-up`}
                  style={{ animationDelay: `${index * 100}ms` }}
                >
                  <div className="flex-1 lg:w-1/2 px-4">
                    <div className={`glass p-6 rounded-xl ${index % 2 === 1 ? 'ml-auto' : 'mr-auto'} max-w-md`}>
                      <div className="flex items-center gap-3 mb-4">
                        <div className="w-12 h-12 rounded-xl bg-accent-primary/10 flex items-center justify-center text-accent-primary">
                          <step.icon className="w-6 h-6" />
                        </div>
                        <div>
                          <span className="text-xs font-mono text-accent-primary font-bold">{step.step}</span>
                          <h3 className="text-lg font-semibold text-text-primary mt-1 block">{step.title}</h3>
                        </div>
                      </div>
                      <p className="text-text-secondary text-sm leading-relaxed">{step.description}</p>
                    </div>
                  </div>

                  {/* Step indicator on the line */}
                  <div className="hidden lg:block lg:w-1/2 flex items-center justify-center">
                    <div className="w-4 h-4 rounded-full bg-accent-primary border-4 border-bg-primary z-10 shadow-glow" />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="py-20 lg:py-28">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-3xl mx-auto mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold text-text-primary mb-4">
              Key Features
            </h2>
            <p className="text-text-secondary text-lg">
              Built for developers who need reliable, accurate GPS route reconstruction.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => (
              <Card key={feature.title} variant="glass" hover className="group animate-slide-up" style={{ animationDelay: `${index * 50}ms` }}>
                <CardContent className="p-6">
                  <div className="w-12 h-12 rounded-xl bg-accent-primary/10 flex items-center justify-center text-accent-primary mb-4 group-hover:scale-110 transition-transform duration-300">
                    <feature.icon className="w-6 h-6" />
                  </div>
                  <h3 className="text-lg font-semibold text-text-primary mb-2">{feature.title}</h3>
                  <p className="text-text-secondary text-sm mb-4 leading-relaxed">{feature.description}</p>
                  <Badge variant="info" className="text-xs">{feature.highlight}</Badge>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Tech Stack */}
      <section className="py-20 lg:py-28 bg-bg-secondary/30">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-3xl mx-auto mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold text-text-primary mb-4">
              Technology Stack
            </h2>
            <p className="text-text-secondary text-lg">
              Modern, type-safe technologies for robust and maintainable code.
            </p>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 max-w-4xl mx-auto">
            {[
              { name: 'FastAPI', category: 'Backend', color: 'from-green-500 to-teal-500' },
              { name: 'React 18', category: 'Frontend', color: 'from-blue-500 to-cyan-500' },
              { name: 'TypeScript', category: 'Language', color: 'from-indigo-500 to-purple-500' },
              { name: 'Tailwind CSS', category: 'Styling', color: 'from-cyan-500 to-teal-500' },
              { name: 'Pydantic v2', category: 'Validation', color: 'from-orange-500 to-red-500' },
              { name: 'NetworkX', category: 'Graph', color: 'from-pink-500 to-rose-500' },
              { name: 'Leaflet', category: 'Maps', color: 'from-green-500 to-emerald-500' },
              { name: 'Vite', category: 'Build', color: 'from-yellow-500 to-orange-500' },
            ].map((tech, index) => (
              <Card key={tech.name} variant="glass" className="text-center p-6 animate-slide-up" style={{ animationDelay: `${index * 30}ms` }}>
                <div className={`w-14 h-14 rounded-xl bg-gradient-to-br ${tech.color} flex items-center justify-center mx-auto mb-4 text-white font-bold text-xl`}>
                  {tech.name.charAt(0)}
                </div>
                <h3 className="font-semibold text-text-primary">{tech.name}</h3>
                <p className="text-xs text-text-muted mt-1">{tech.category}</p>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 lg:py-28">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="glass-strong rounded-2xl p-8 lg:p-16 text-center max-w-3xl mx-auto relative overflow-hidden">
            <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--color-accent-primary)/10_0%,_transparent_70%)] pointer-events-none" />
            <h2 className="text-3xl sm:text-4xl font-bold text-text-primary mb-4 relative z-10">
              Ready to Reconstruct Your Routes?
            </h2>
            <p className="text-text-secondary text-lg mb-8 relative z-10">
              Upload a GPS file or send JSON points to the API.
              Get accurate road-matched routes in seconds.
            </p>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4 relative z-10">
              <Button size="lg" asChild>
                <Link to="/reconstruct">
                  Start Reconstruction
                  <ArrowRight className="w-4 h-4" />
                </Link>
              </Button>
              <Button size="lg" variant="ghost" asChild>
                <Link to="/settings">
                  View Settings
                </Link>
              </Button>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}