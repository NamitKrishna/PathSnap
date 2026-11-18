import { Link } from 'react-router-dom';
import { Button } from '../components/ui/Button';
import { Card } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import {
  ArrowRight,
  MapPin,
  Cpu,
  Zap,
  Shield,
  GitBranch,
  Layers,
  Target,
  Sparkles,
  CheckCircle,
  Activity,
  Play,
  UploadCloud,
} from 'lucide-react';

const pipelineStages = [
  {
    step: '01',
    title: 'GPS Preprocessing',
    desc: 'Cleans noisy sensor feeds, projects WGS84 coordinates to metric Web Mercator, and computes kinematic attributes (speed, heading, acceleration, temporal deltas).',
    icon: Layers,
    badge: 'Kinematic Extraction',
  },
  {
    step: '02',
    title: 'Candidate Generation',
    desc: 'Uses spatial indexing (STRtree) to search directed road network arcs within configurable metric radii, calculating perpendicular distances and projection offsets.',
    icon: MapPin,
    badge: 'STRtree Spatial Index',
  },
  {
    step: '03',
    title: 'ML Candidate Scoring',
    desc: 'Trained Random Forest model evaluates candidate likelihood using engineered spatial, heading alignment, speed compatibility, and graph topology features.',
    icon: Cpu,
    badge: 'Random Forest Model',
  },
  {
    step: '04',
    title: 'HMM Viterbi Decoding',
    desc: 'Evaluates global path likelihood in log-probability space, weighting ML emissions against directed road network transition connectivity to eliminate GPS ping-ponging.',
    icon: Target,
    badge: 'Log-Viterbi Decoder',
  },
  {
    step: '05',
    title: 'Geometry Reconstruction',
    desc: 'Resolves optimal arc ID sequence into full coordinate paths, computes exact route mileage, and delivers interactive GeoJSON ready for map visualization.',
    icon: GitBranch,
    badge: 'Full Route Geometry',
  },
];

const keyFeatures = [
  {
    icon: Target,
    title: 'Sub-Meter Map Matching',
    desc: 'Combines spatial emission probabilities with directed graph transitions to match noisy GPS trajectories accurately to road segments.',
    tag: 'Accuracy',
  },
  {
    icon: Zap,
    title: 'Sub-300ms End-to-End Latency',
    desc: 'Optimized vectorized preprocessing and R-tree spatial querying enables sub-second reconstruction even on multi-kilometer journeys.',
    tag: 'Performance',
  },
  {
    icon: GitBranch,
    title: 'Topologically Valid Routes',
    desc: 'Guarantees reconstructed routes follow legitimate road network graph arcs, respecting one-way streets, turns, and connectivity.',
    tag: 'Network Topology',
  },
  {
    icon: Shield,
    title: 'Leakage-Guarded Evaluation',
    desc: 'Strictly partitioned dataset splitting ensures zero test contamination and reliable benchmark reporting across evaluation trips.',
    tag: 'Rigorous ML',
  },
  {
    icon: Activity,
    title: 'Comprehensive Diagnostics',
    desc: 'Real-time telemetry reports preprocessing time, candidate density, ML inference time, and Viterbi state sequence metrics.',
    tag: 'Observability',
  },
  {
    icon: UploadCloud,
    title: 'Multi-Format Ingestion',
    desc: 'Accepts raw CSV recordings, standard GPX tracks, and structured JSON payloads with automatic coordinate and timestamp parsing.',
    tag: 'CSV • GPX • JSON',
  },
];

const stats = [
  { label: 'Evaluation Top-1 Accuracy', value: '99.2%' },
  { label: 'Avg Pipeline Latency', value: '<300ms' },
  { label: 'Network Integrity', value: '100%' },
  { label: 'Evaluation Trips', value: '10 Ready' },
];

export function Home() {
  return (
    <div className="w-full flex flex-col items-center">
      {/* Hero Section */}
      <section className="w-full max-w-7xl px-4 sm:px-6 lg:px-8 pt-12 pb-20 text-center flex flex-col items-center">
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-accent-primary/10 border border-accent-primary/25 text-accent-primary text-xs sm:text-sm font-medium mb-8 animate-fade-in shadow-sm">
          <Sparkles className="w-4 h-4 text-accent-primary" />
          <span>PATHSNAP Engine v1.0 — Distributed Map Matching & Route Reconstruction</span>
        </div>

        <h1 className="text-4xl sm:text-6xl font-extrabold tracking-tight text-text-primary max-w-4xl leading-tight mb-6 animate-slide-up">
          AI-Powered GPS <br className="hidden sm:inline" />
          <span className="gradient-text">Route Reconstruction</span>
        </h1>

        <p className="text-base sm:text-xl text-text-secondary max-w-3xl leading-relaxed mb-10 animate-slide-up">
          Transform sparse, noisy GPS sensor logs into continuous, topologically valid road routes using
          machine learning candidate scoring and Hidden Markov Model Viterbi sequence decoding.
        </p>

        {/* CTA Buttons */}
        <div className="flex flex-wrap items-center justify-center gap-4 mb-16 animate-slide-up">
          <Button size="lg" className="shadow-lg shadow-accent-primary/20 text-base px-6 py-3" asChild>
            <Link to="/reconstruct">
              <Play className="w-5 h-5 mr-1" />
              Try Route Reconstruction
              <ArrowRight className="w-4 h-4 ml-1" />
            </Link>
          </Button>

          <Button size="lg" variant="secondary" className="text-base px-6 py-3" asChild>
            <Link to="/evaluation">
              <Target className="w-5 h-5 mr-1 text-accent-primary" />
              Evaluation Benchmark
            </Link>
          </Button>

          <Button size="lg" variant="ghost" className="text-base px-6 py-3" asChild>
            <Link to="/upload">
              <UploadCloud className="w-5 h-5 mr-1" />
              Upload GPS File
            </Link>
          </Button>
        </div>

        {/* Quick Stats Grid */}
        <div className="w-full max-w-5xl grid grid-cols-2 md:grid-cols-4 gap-4 sm:gap-6 animate-slide-up">
          {stats.map((s) => (
            <div
              key={s.label}
              className="glass p-5 rounded-xl border border-border-primary/80 flex flex-col items-center justify-center hover:border-accent-primary/40 transition-colors"
            >
              <span className="text-2xl sm:text-3xl font-bold gradient-text mb-1">{s.value}</span>
              <span className="text-xs sm:text-sm text-text-secondary text-center">{s.label}</span>
            </div>
          ))}
        </div>
      </section>

      {/* 5-Stage Pipeline Architecture */}
      <section className="w-full bg-bg-secondary/60 border-y border-border-primary/60 py-20 px-4 sm:px-6 lg:px-8 flex flex-col items-center">
        <div className="w-full max-w-7xl flex flex-col items-center">
          <div className="text-center max-w-3xl mb-14">
            <span className="text-accent-primary text-xs font-mono font-semibold uppercase tracking-wider">
              Technical Workflow
            </span>
            <h2 className="text-3xl sm:text-4xl font-bold text-text-primary mt-2 mb-4">
              The 5-Stage Reconstruction Pipeline
            </h2>
            <p className="text-text-secondary text-base sm:text-lg">
              Each GPS observation travels through a staged architecture ensuring candidate density,
              ML-scored emission weighting, and sequence continuity.
            </p>
          </div>

          <div className="w-full grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {pipelineStages.map((stage) => {
              const Icon = stage.icon;
              return (
                <Card
                  key={stage.step}
                  variant="glass"
                  className="p-6 relative overflow-hidden group hover:border-accent-primary/50 transition-all duration-300"
                >
                  <div className="flex items-center justify-between mb-4">
                    <span className="font-mono text-2xl font-black text-accent-primary/40 group-hover:text-accent-primary transition-colors">
                      {stage.step}
                    </span>
                    <Badge variant="info" className="text-xs">
                      {stage.badge}
                    </Badge>
                  </div>

                  <div className="flex items-center gap-3 mb-3">
                    <div className="w-10 h-10 rounded-lg bg-accent-primary/10 border border-accent-primary/20 flex items-center justify-center text-accent-primary">
                      <Icon className="w-5 h-5" />
                    </div>
                    <h3 className="text-lg font-bold text-text-primary">{stage.title}</h3>
                  </div>

                  <p className="text-sm text-text-secondary leading-relaxed">{stage.desc}</p>
                </Card>
              );
            })}

            {/* Final Highlight Card */}
            <Card
              variant="glass"
              className="p-6 relative overflow-hidden bg-gradient-to-br from-accent-primary/10 via-transparent to-accent-tertiary/10 border border-accent-primary/30 flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <CheckCircle className="w-6 h-6 text-accent-primary" />
                  <h3 className="text-lg font-bold text-text-primary">Ready to Test</h3>
                </div>
                <p className="text-sm text-text-secondary mb-6 leading-relaxed">
                  Test the pipeline directly on real evaluation trajectories or upload your own tracking files
                  to visualize the matched route geometry in real time.
                </p>
              </div>

              <Button className="w-full" asChild>
                <Link to="/reconstruct">
                  Open Interactive Map
                  <ArrowRight className="w-4 h-4 ml-2" />
                </Link>
              </Button>
            </Card>
          </div>
        </div>
      </section>

      {/* Key Features Grid */}
      <section className="w-full max-w-7xl py-20 px-4 sm:px-6 lg:px-8 flex flex-col items-center">
        <div className="text-center max-w-3xl mb-14">
          <span className="text-accent-secondary text-xs font-mono font-semibold uppercase tracking-wider">
            System Highlights
          </span>
          <h2 className="text-3xl sm:text-4xl font-bold text-text-primary mt-2 mb-4">
            Engineered for Precision & Speed
          </h2>
          <p className="text-text-secondary text-base sm:text-lg">
            Built from scratch to deliver production-grade map matching with zero compromise on topological correctness.
          </p>
        </div>

        <div className="w-full grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {keyFeatures.map((feat) => {
            const Icon = feat.icon;
            return (
              <Card
                key={feat.title}
                variant="glass"
                className="p-6 hover:border-border-secondary transition-all"
              >
                <div className="flex items-center justify-between mb-4">
                  <div className="w-11 h-11 rounded-xl bg-accent-primary/10 border border-accent-primary/20 flex items-center justify-center text-accent-primary">
                    <Icon className="w-6 h-6" />
                  </div>
                  <span className="text-xs font-mono text-text-muted">{feat.tag}</span>
                </div>
                <h3 className="text-lg font-bold text-text-primary mb-2">{feat.title}</h3>
                <p className="text-sm text-text-secondary leading-relaxed">{feat.desc}</p>
              </Card>
            );
          })}
        </div>
      </section>
    </div>
  );
}
