import { useState, useEffect } from 'react';
import { Button } from '../components/ui/Button';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card';
import { Input } from '../components/ui/Input';
import { Badge } from '../components/ui/Badge';
import { cn } from '../lib/utils';
import { api, type HealthResponse } from '../lib/api';
import {
  Server,
  Database,
  Cpu,
  MapPin,
  Save,
  RotateCcw,
  CheckCircle,
  Wifi,
  WifiOff,
  Info,
} from 'lucide-react';

export function Settings() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isChecking, setIsChecking] = useState(false);
  const [apiBase, setApiBase] = useState(import.meta.env.VITE_API_BASE || '/api/v1');
  const [saved, setSaved] = useState(false);

  const checkHealth = async () => {
    setIsChecking(true);
    try {
      const response = await api.checkHealth();
      setHealth(response);
    } catch (err) {
      setHealth(null);
    } finally {
      setIsChecking(false);
    }
  };

  const saveSettings = () => {
    localStorage.setItem('apiBase', apiBase);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  useEffect(() => {
    const stored = localStorage.getItem('apiBase');
    if (stored) {
      setApiBase(stored);
    }
    checkHealth();
  }, []);

  return (
    <div className="min-h-[calc(100vh-4rem)] py-8 lg:py-12">
      <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8">
        <div className="mb-8 animate-slide-up">
          <h1 className="text-4xl font-bold tracking-tight text-text-primary mb-2">Settings</h1>
          <p className="text-text-secondary">Configure API connection and view system status.</p>
        </div>

        {/* API Configuration */}
        <Card variant="glass" className="animate-slide-up">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Server className="w-5 h-5 text-accent-primary" />
              API Configuration
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-text-secondary mb-2">
                API Base URL
              </label>
              <div className="flex gap-3">
                <Input
                  value={apiBase}
                  onChange={(e) => setApiBase(e.target.value)}
                  placeholder="/api/v1 or http://laptop1:8000/api/v1"
                  className="flex-1"
                />
                <Button onClick={saveSettings} disabled={saved}>
                  {saved ? (
                    <>
                      <CheckCircle className="w-4 h-4" />
                      Saved
                    </>
                  ) : (
                    <>
                      <Save className="w-4 h-4" />
                      Save
                    </>
                  )}
                </Button>
              </div>
              <p className="text-xs text-text-muted mt-1">
                For two-laptop setup, use the full URL of Laptop 1 (e.g., http://192.168.1.100:8000/api/v1)
              </p>
            </div>

            <div className="pt-6 border-t border-border-primary">
              <h3 className="text-lg font-medium text-text-primary mb-4">Quick Actions</h3>
              <div className="flex flex-wrap gap-3">
                <Button variant="secondary" onClick={checkHealth} disabled={isChecking}>
                  {isChecking ? (
                    <>
                      <RotateCcw className="w-4 h-4 animate-spin" />
                      Checking...
                    </>
                  ) : (
                    <>
                      <Wifi className="w-4 h-4" />
                      Check Connection
                    </>
                  )}
                </Button>
                <Button variant="ghost" onClick={() => setApiBase('/api/v1')}>
                  <RotateCcw className="w-4 h-4" />
                  Reset to Default
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* System Status */}
        <Card variant="glass" className="mt-6 animate-slide-up">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Cpu className="w-5 h-5 text-accent-primary" />
              System Status
            </CardTitle>
          </CardHeader>
          <CardContent>
            {health ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 bg-bg-tertiary rounded-lg">
                  <div className="flex items-center gap-3">
                    <div className={cn(
                      'w-3 h-3 rounded-full',
                      health.status === 'healthy' && 'bg-success',
                      health.status === 'degraded' && 'bg-warning',
                      health.status === 'unhealthy' && 'bg-error'
                    )} />
                    <span className="font-medium text-text-primary capitalize">{health.status}</span>
                  </div>
                  <Badge variant={health.status === 'healthy' ? 'success' : health.status === 'degraded' ? 'warning' : 'error'}>
                    {health.status}
                  </Badge>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {Object.entries(health.components).map(([component, status]) => (
                    <div key={component} className="p-4 bg-bg-tertiary rounded-lg text-center">
                      <div className="text-2xl font-mono font-bold text-text-primary capitalize">{status}</div>
                      <div className="text-xs text-text-muted uppercase tracking-wider mt-1">{component.replace('_', ' ')}</div>
                    </div>
                  ))}
                </div>

                {health.uptime_seconds && (
                  <div className="pt-4 border-t border-border-primary flex items-center justify-between">
                    <span className="text-text-secondary">Uptime</span>
                    <span className="font-mono text-text-primary">
                      {Math.floor(health.uptime_seconds / 3600)}h {Math.floor((health.uptime_seconds % 3600) / 60)}m
                    </span>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex items-center justify-center py-8">
                <Button variant="secondary" onClick={checkHealth} disabled={isChecking}>
                  {isChecking ? (
                    <>
                      <RotateCcw className="w-4 h-4 animate-spin mr-2" />
                      Checking...
                    </>
                  ) : (
                    <>
                      <WifiOff className="w-4 h-4 mr-2" />
                      Check Connection
                    </>
                  )}
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Pipeline Components */}
        <Card variant="glass" className="mt-6 animate-slide-up">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="w-5 h-5 text-accent-primary" />
              Pipeline Components
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {[
                { name: 'Preprocessor', icon: Cpu, description: 'Cleans GPS data, projects to Web Mercator, extracts kinematic features' },
                { name: 'Candidate Generator', icon: MapPin, description: 'Finds nearby road segments using spatial indexing (R-tree)' },
                { name: 'ML Model', icon: Cpu, description: 'Gradient boosting (XGBoost/LightGBM) scores candidates', status: 'mock' },
                { name: 'HMM/Viterbi Decoder', icon: MapPin, description: 'Log-space Viterbi algorithm for sequence decoding' },
                { name: 'Road Graph', icon: Database, description: 'NetworkX MultiDiGraph with geometry (GeoPackage/GraphML/GeoJSON)' },
              ].map((comp) => (
                <div key={comp.name} className="flex items-center gap-4 p-4 bg-bg-tertiary rounded-lg">
                  <div className="w-10 h-10 rounded-lg bg-accent-primary/10 flex items-center justify-center text-accent-primary flex-shrink-0">
                    <comp.icon className="w-5 h-5" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-text-primary">{comp.name}</span>
                      {comp.status === 'mock' && <Badge variant="warning" size="sm">Mock</Badge>}
                    </div>
                    <p className="text-sm text-text-secondary">{comp.description}</p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* About */}
        <Card variant="glass" className="mt-6 animate-slide-up">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Info className="w-5 h-5 text-accent-primary" />
              About
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="grid grid-cols-2 gap-4 text-center">
              <div className="p-3 bg-bg-tertiary rounded-lg">
                <div className="text-2xl font-bold text-accent-primary">1.0.0</div>
                <div className="text-text-muted">Version</div>
              </div>
              <div className="p-3 bg-bg-tertiary rounded-lg">
                <div className="text-2xl font-bold text-accent-primary">FastAPI</div>
                <div className="text-text-muted">Backend</div>
              </div>
              <div className="p-3 bg-bg-tertiary rounded-lg">
                <div className="text-2xl font-bold text-accent-primary">React 18</div>
                <div className="text-text-muted">Frontend</div>
              </div>
              <div className="p-3 bg-bg-tertiary rounded-lg">
                <div className="text-2xl font-bold text-accent-primary">MIT</div>
                <div className="text-text-muted">License</div>
              </div>
            </div>

            <div className="pt-4 border-t border-border-primary">
              <p className="text-text-secondary">
                PATHSNAP - AI-powered GPS trajectory reconstruction and map matching.
                Built for the PATHSNAP team project.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}