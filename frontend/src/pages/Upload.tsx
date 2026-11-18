import { useState, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '../components/ui/Button';
import { Card, CardContent } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { Skeleton } from '../components/ui/Skeleton';
import { cn } from '../lib/utils';
import { api, type GPSPoint, type FileUploadResponse } from '../lib/api';
import {
  Upload as UploadIcon,
  FileText,
  MapPin,
  CheckCircle,
  AlertCircle,
  X,
  Eye,
  Download,
  Trash2,
} from 'lucide-react';

interface UploadedFile {
  file: File;
  response: FileUploadResponse | null;
  points: GPSPoint[];
  error: string | null;
  isProcessing: boolean;
}

export function Upload() {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [isDragActive, setIsDragActive] = useState(false);

  const handleFiles = useCallback((fileList: FileList) => {
    const newFiles: UploadedFile[] = Array.from(fileList).map(file => ({
      file,
      response: null,
      points: [],
      error: null,
      isProcessing: false,
    }));
    setFiles(prev => [...prev, ...newFiles]);
    processFiles(newFiles);
  }, []);

  const processFiles = async (newFiles: UploadedFile[]) => {
    for (const fileData of newFiles) {
      setFiles(prev => prev.map(f => f === fileData ? { ...f, isProcessing: true } : f));

      try {
        const response = await api.uploadGPSFile(fileData.file);
        let points: GPSPoint[] = [];
        try {
          points = await api.parseGPSFile(fileData.file);
        } catch {
          // Fallback if client parser encounters format variation
        }
        setFiles(prev => prev.map(f => f === fileData ? { ...f, response, points, isProcessing: false } : f));
      } catch (err) {
        setFiles(prev => prev.map(f => f === fileData ? {
          ...f,
          error: err instanceof Error ? err.message : 'Upload failed',
          isProcessing: false
        } : f));
      }
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setIsDragActive(true);
    } else if (e.type === 'dragleave') {
      setIsDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
    if (e.dataTransfer.files.length > 0) {
      handleFiles(e.dataTransfer.files);
    }
  };

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleReconstruct = (points: GPSPoint[]) => {
    navigate('/reconstruct', { state: { gpsPoints: points } });
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
  };

  const formatDuration = (ms: number) => {
    if (ms < 1000) return `${ms.toFixed(1)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  return (
    <div className="min-h-[calc(100vh-4rem)] py-8 lg:py-12">
      <div className="mx-auto max-w-5xl px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12 animate-slide-up">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-accent-primary/10 border border-accent-primary/20 text-accent-primary text-sm font-medium mb-6">
            <UploadIcon className="w-4 h-4" />
            <span>File Upload & Validation</span>
          </div>
          <h1 className="text-4xl sm:text-5xl font-bold tracking-tight text-text-primary mb-4">
            Upload GPS Files
          </h1>
          <p className="text-lg text-text-secondary max-w-2xl mx-auto">
            Drag and drop CSV, GPX, or JSON files to validate and parse GPS tracks.
            Validated files can be sent directly to the reconstruction pipeline.
          </p>
        </div>

        {/* Drop Zone */}
        <Card variant="glass" className={cn('animate-slide-up', isDragActive && 'border-accent-primary/50 bg-accent-primary/5')}>
          <CardContent className="p-8 lg:p-12">
            <div
              className={cn(
                'border-2 border-dashed rounded-xl text-center transition-all duration-200',
                isDragActive
                  ? 'border-accent-primary bg-accent-primary/5'
                  : 'border-border-primary hover:border-border-secondary'
              )}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv,.gpx,.json"
                multiple
                className="hidden"
                id="file-upload"
                onChange={(e) => e.target.files && handleFiles(e.target.files)}
              />

              <label htmlFor="file-upload" className="cursor-pointer flex flex-col items-center gap-4">
                <div className="w-20 h-20 rounded-2xl bg-accent-primary/10 flex items-center justify-center">
                  <UploadIcon className="w-10 h-10 text-accent-primary" />
                </div>
                <div>
                  <p className="text-xl font-medium text-text-primary mb-1">
                    Drag & drop GPS files here
                  </p>
                  <p className="text-text-secondary">
                    or <span className="text-accent-primary hover:underline">click to browse</span>
                  </p>
                  <p className="text-xs text-text-muted mt-2">
                    Supports CSV, GPX, JSON • Max 50MB per file
                  </p>
                </div>
              </label>
            </div>
          </CardContent>
        </Card>

        {/* Uploaded Files */}
        {files.length > 0 && (
          <div className="mt-8 animate-slide-up">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-semibold text-text-primary">
                Uploaded Files ({files.length})
              </h2>
              <Button variant="ghost" onClick={() => setFiles([])}>
                <Trash2 className="w-4 h-4" />
                Clear All
              </Button>
            </div>

            <div className="space-y-4">
              {files.map((fileData, index) => (
                <Card key={`${fileData.file.name}-${index}`} variant="glass" className="overflow-hidden">
                  <CardContent className="p-4">
                    {fileData.isProcessing ? (
                      <div className="flex items-center gap-4">
                        <div className="w-12 h-12 rounded-xl bg-accent-primary/10 flex items-center justify-center flex-shrink-0">
                          <div className="w-6 h-6 border-2 border-accent-primary border-t-transparent rounded-full animate-spin" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="font-medium text-text-primary truncate">{fileData.file.name}</p>
                          <p className="text-sm text-text-secondary">{formatFileSize(fileData.file.size)}</p>
                        </div>
                        <Skeleton className="h-8 w-24" variant="rectangular" />
                      </div>
                    ) : fileData.error ? (
                      <div className="flex items-center gap-4 p-4 bg-error/10 border border-error/30 rounded-lg">
                        <AlertCircle className="w-6 h-6 text-error flex-shrink-0" />
                        <div className="flex-1">
                          <p className="font-medium text-error">{fileData.file.name}</p>
                          <p className="text-sm text-text-secondary">{fileData.error}</p>
                        </div>
                        <Button variant="ghost" size="sm" onClick={() => removeFile(index)}>
                          <X className="w-4 h-4" />
                        </Button>
                      </div>
                    ) : fileData.response ? (
                      <div className="flex items-start gap-4">
                        <div className="w-12 h-12 rounded-xl bg-success/10 flex items-center justify-center flex-shrink-0">
                          <CheckCircle className="w-6 h-6 text-success" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-2">
                            <p className="font-medium text-text-primary truncate">{fileData.file.name}</p>
                            <Badge variant="info" className="text-xs capitalize">{fileData.response.format}</Badge>
                            <Badge variant="success" dot className="text-xs">{fileData.response.point_count} points</Badge>
                          </div>
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm mb-3">
                            <div className="p-2 bg-bg-tertiary rounded-lg">
                              <div className="text-text-muted text-xs">Duration</div>
                              <div className="font-mono text-text-primary">{formatDuration(fileData.response.duration_ms)}</div>
                            </div>
                            {fileData.response.bounds && (
                              <>
                                <div className="p-2 bg-bg-tertiary rounded-lg">
                                  <div className="text-text-muted text-xs">Lat Range</div>
                                  <div className="font-mono text-text-primary text-xs">
                                    {fileData.response.bounds.min_lat.toFixed(4)} – {fileData.response.bounds.max_lat.toFixed(4)}
                                  </div>
                                </div>
                                <div className="p-2 bg-bg-tertiary rounded-lg">
                                  <div className="text-text-muted text-xs">Lon Range</div>
                                  <div className="font-mono text-text-primary text-xs">
                                    {fileData.response.bounds.min_lon.toFixed(4)} – {fileData.response.bounds.max_lon.toFixed(4)}
                                  </div>
                                </div>
                              </>
                            )}
                          </div>
                          <div className="flex gap-2">
                            <Button
                              variant="secondary"
                              size="sm"
                              onClick={() => handleReconstruct(fileData.points)}
                            >
                              <MapPin className="w-4 h-4" />
                              Reconstruct
                            </Button>
                            <Button variant="ghost" size="sm">
                              <Eye className="w-4 h-4" />
                              Preview
                            </Button>
                            <Button variant="ghost" size="sm">
                              <Download className="w-4 h-4" />
                              Download
                            </Button>
                            <Button variant="ghost" size="sm" onClick={() => removeFile(index)}>
                              <Trash2 className="w-4 h-4" />
                            </Button>
                          </div>
                        </div>
                      </div>
                    ) : null}
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        )}

        {/* Supported Formats */}
        <section className="mt-16 animate-slide-up">
          <h2 className="text-2xl font-semibold text-text-primary mb-6">Supported Formats</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[
              {
                name: 'CSV',
                icon: FileText,
                description: 'Comma-separated values with latitude, longitude, timestamp columns',
                example: 'latitude,longitude,timestamp,speed,heading,accuracy\n37.7749,-122.4194,2024-01-15T10:00:00,10.5,45,5.0',
                required: ['latitude', 'longitude', 'timestamp'],
                optional: ['speed', 'heading', 'accuracy'],
              },
              {
                name: 'GPX',
                icon: MapPin,
                description: 'GPS Exchange Format - standard for GPS devices and apps',
                example: '<trk><trkseg><trkpt lat="37.7749" lon="-122.4194"><time>2024-01-15T10:00:00Z</time></trkpt></trkseg></trk>',
                required: ['lat', 'lon', 'time'],
                optional: ['ele', 'speed'],
              },
              {
                name: 'JSON',
                icon: FileText,
                description: 'Flexible JSON structure with points array',
                example: '{"points": [{"latitude": 37.7749, "longitude": -122.4194, "timestamp": "2024-01-15T10:00:00"}]}',
                required: ['latitude', 'longitude', 'timestamp'],
                optional: ['speed', 'heading', 'accuracy'],
              },
            ].map((format) => (
              <Card key={format.name} variant="glass" hover className="p-6">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-12 h-12 rounded-xl bg-accent-primary/10 flex items-center justify-center text-accent-primary">
                    <format.icon className="w-6 h-6" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-text-primary">{format.name}</h3>
                    <p className="text-sm text-text-secondary">{format.description}</p>
                  </div>
                </div>

                <div className="space-y-3 mb-4">
                  <div>
                    <p className="text-xs font-medium text-text-muted uppercase tracking-wider mb-2">Required Fields</p>
                    <div className="flex flex-wrap gap-1">
                      {format.required.map(field => (
                        <Badge key={field} variant="default" size="sm">{field}</Badge>
                      ))}
                    </div>
                  </div>
                  <div>
                    <p className="text-xs font-medium text-text-muted uppercase tracking-wider mb-2">Optional Fields</p>
                    <div className="flex flex-wrap gap-1">
                      {format.optional.map(field => (
                        <Badge key={field} variant="info" size="sm">{field}</Badge>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="bg-bg-tertiary rounded-lg p-3 font-mono text-xs text-text-secondary overflow-x-auto">
                  {format.example}
                </div>
              </Card>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}