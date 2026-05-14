'use client';

import { useState } from 'react';
import { Upload, FileText, Clock, CheckCircle, AlertCircle, RefreshCw, Video, Loader2, Youtube, Link2 } from 'lucide-react';
import { motion } from 'framer-motion';

const ML_BACKEND_URL = process.env.NEXT_PUBLIC_ML_BACKEND_URL || 'http://localhost:8000';

interface ModelPrediction {
  probability: number;
  risk_level: string;
}

interface PredictionResult {
  success: boolean;
  ensemble_probability: number;
  risk_level: string;
  num_frames_processed?: number;
  source?: string;
  youtube_url?: string;
  model_predictions: Record<string, ModelPrediction>;
  error?: string;
}

export default function ProfessionalAssessmentsPage() {
  const [selectedPatient, setSelectedPatient] = useState('');
  const [uploading, setUploading] = useState(false);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [inputMode, setInputMode] = useState<'file' | 'youtube'>('file');
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState('');

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!file.name.match(/\.(mp4|mov|avi)$/i)) {
        setError('Please select an MP4, MOV, or AVI video file.');
        return;
      }
      setVideoFile(file);
      setError('');
      setResult(null);
    }
  };

  const handleRunAssessment = async () => {
    if (!selectedPatient) {
      setError('Please select a patient.');
      return;
    }

    if (inputMode === 'file' && !videoFile) {
      setError('Please upload a video file.');
      return;
    }

    if (inputMode === 'youtube' && !youtubeUrl.trim()) {
      setError('Please enter a YouTube URL.');
      return;
    }

    setUploading(true);
    setError('');
    setResult(null);

    try {
      let res;

      if (inputMode === 'youtube') {
        res = await fetch(`${ML_BACKEND_URL}/api/predict-youtube`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ youtube_url: youtubeUrl.trim(), fps: 15 }),
        });
      } else {
        const formData = new FormData();
        formData.append('video', videoFile!);
        formData.append('fps', '15');
        res = await fetch(`${ML_BACKEND_URL}/api/predict`, {
          method: 'POST',
          body: formData,
        });
      }

      const data: PredictionResult = await res.json();

      if (!data.success) {
        setError(data.error || 'Assessment failed.');
        return;
      }

      setResult(data);
    } catch {
      setError(
        'Could not connect to the ML backend. Make sure the server is running on port 8000.\n\n' +
        'Run: cd jasmine-next && pip install -r backend/requirements.txt && uvicorn backend.main:app --reload --port 8000'
      );
    } finally {
      setUploading(false);
    }
  };

  const riskColor = (risk: string) => {
    switch (risk) {
      case 'High Risk': return { bg: 'rgba(220, 38, 38, 0.1)', text: '#dc2626' };
      case 'Moderate Risk': return { bg: 'rgba(217, 119, 6, 0.1)', text: '#d97706' };
      default: return { bg: 'rgba(22, 163, 74, 0.1)', text: '#16a34a' };
    }
  };

  const isValidUrl = (url: string) => {
    return url.match(/^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.be)\/.+/i);
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold" style={{ color: 'var(--foreground)' }}>Assessments</h1>
        <p style={{ color: 'var(--text-muted)' }}>Run ASD screening on patient video</p>
      </div>

      {/* Upload Card */}
      <div className="p-6 rounded-2xl border-2 border-dashed" style={{ backgroundColor: 'var(--background)', borderColor: 'var(--border)' }}>
        <div className="flex flex-col items-center text-center">
          <div className="w-16 h-16 rounded-full flex items-center justify-center mb-4" style={{ backgroundColor: 'var(--primary-light)' }}>
            <Video className="w-8 h-8" style={{ color: 'var(--primary)' }} />
          </div>
          <h3 className="text-lg font-semibold mb-2" style={{ color: 'var(--foreground)' }}>Run New Assessment</h3>
          <p className="mb-4" style={{ color: 'var(--text-muted)' }}>Analyze movement patterns to assess ASD risk</p>

          {/* Input Mode Toggle */}
          <div className="flex items-center gap-2 mb-4 p-1 rounded-xl" style={{ backgroundColor: 'var(--background-alt)' }}>
            <button
              onClick={() => setInputMode('file')}
              className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all"
              style={{
                backgroundColor: inputMode === 'file' ? 'var(--primary)' : 'transparent',
                color: inputMode === 'file' ? 'white' : 'var(--text-muted)',
              }}
            >
              <Upload className="w-4 h-4" />
              Upload MP4
            </button>
            <button
              onClick={() => setInputMode('youtube')}
              className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all"
              style={{
                backgroundColor: inputMode === 'youtube' ? 'var(--primary)' : 'transparent',
                color: inputMode === 'youtube' ? 'white' : 'var(--text-muted)',
              }}
            >
              <Youtube className="w-4 h-4" />
              YouTube URL
            </button>
          </div>

          <div className="flex flex-col gap-3 w-full max-w-md">
            <select
              value={selectedPatient}
              onChange={(e) => setSelectedPatient(e.target.value)}
              className="w-full px-4 py-3 rounded-xl"
              style={{ backgroundColor: 'var(--background)', border: '1px solid var(--border)', color: 'var(--foreground)' }}
            >
              <option value="">Select patient...</option>
              <option value="1">Emma Thompson</option>
              <option value="2">Liam Johnson</option>
              <option value="3">Sophie Williams</option>
            </select>

            {!result && inputMode === 'file' && (
              <label className="w-full px-4 py-3 rounded-xl text-center cursor-pointer"
                style={{ backgroundColor: 'var(--background-alt)', border: '1px solid var(--border)' }}>
                <input
                  type="file"
                  accept=".mp4,.mov,.avi"
                  onChange={handleFileChange}
                  className="hidden"
                />
                {videoFile ? (
                  <span style={{ color: 'var(--foreground)' }}>{videoFile.name}</span>
                ) : (
                  <span style={{ color: 'var(--text-muted)' }}>Click to select MP4 video</span>
                )}
              </label>
            )}

            {!result && inputMode === 'youtube' && (
              <div className="relative">
                <Link2 className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5" style={{ color: 'var(--text-muted)' }} />
                <input
                  type="url"
                  value={youtubeUrl}
                  onChange={(e) => { setYoutubeUrl(e.target.value); setError(''); setResult(null); }}
                  placeholder="https://youtube.com/watch?v=..."
                  className="w-full pl-12 pr-4 py-3 rounded-xl"
                  style={{ backgroundColor: 'var(--background-alt)', border: '1px solid var(--border)', color: 'var(--foreground)' }}
                />
              </div>
            )}

            <button
              onClick={handleRunAssessment}
              disabled={
                !selectedPatient ||
                uploading ||
                (inputMode === 'file' && !videoFile) ||
                (inputMode === 'youtube' && (!youtubeUrl.trim() || !isValidUrl(youtubeUrl)))
              }
              className="w-full px-6 py-3 text-white font-medium rounded-xl transition-all disabled:opacity-50 flex items-center justify-center gap-2"
              style={{ backgroundColor: 'var(--primary)' }}
            >
              {uploading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  {inputMode === 'youtube' ? <Youtube className="w-5 h-5" /> : <Upload className="w-5 h-5" />}
                  Run Assessment
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="p-4 rounded-xl whitespace-pre-line" style={{ backgroundColor: 'rgba(220, 38, 38, 0.1)', border: '1px solid #dc2626', color: '#dc2626' }}>
          <p className="text-sm">{error}</p>
        </div>
      )}

      {/* Results */}
      {result && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-6 rounded-2xl"
          style={{ backgroundColor: 'var(--background)', border: '1px solid var(--border)' }}
        >
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold" style={{ color: 'var(--foreground)' }}>Assessment Result</h2>
            {result.source === 'youtube' && (
              <a
                href={result.youtube_url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs"
                style={{ backgroundColor: 'var(--background-alt)', color: 'var(--primary)' }}
              >
                <Youtube className="w-3 h-3" />
                Source Video
              </a>
            )}
          </div>

          {/* Ensemble Score */}
          <div className="text-center mb-6">
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Ensemble ASD Probability</p>
            <p className="text-5xl font-bold my-2" style={{ color: riskColor(result.risk_level).text }}>
              {(result.ensemble_probability * 100).toFixed(1)}%
            </p>
            <span className="inline-block px-4 py-1.5 rounded-full text-lg font-semibold"
              style={{ backgroundColor: riskColor(result.risk_level).bg, color: riskColor(result.risk_level).text }}>
              {result.risk_level}
            </span>
            {result.num_frames_processed && (
              <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>
                {result.num_frames_processed} frames processed
              </p>
            )}
          </div>

          {/* Model Breakdown */}
          <div className="pt-4 border-t" style={{ borderColor: 'var(--border)' }}>
            <p className="text-sm font-medium mb-3" style={{ color: 'var(--foreground)' }}>Model Predictions:</p>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {Object.entries(result.model_predictions).map(([model, pred]) => (
                <div key={model} className="p-3 rounded-lg" style={{ backgroundColor: 'var(--background-alt)' }}>
                  <p className="text-xs uppercase" style={{ color: 'var(--text-muted)' }}>{model}</p>
                  <p className="text-lg font-semibold" style={{ color: 'var(--foreground)' }}>{(pred.probability * 100).toFixed(0)}%</p>
                  <p className="text-xs" style={{ color: riskColor(pred.risk_level).text }}>{pred.risk_level}</p>
                </div>
              ))}
            </div>
          </div>
        </motion.div>
      )}

      {/* Recent Assessments placeholder */}
      <div className="p-6 rounded-2xl" style={{ backgroundColor: 'var(--background)', border: '1px solid var(--border)' }}>
        <h2 className="text-lg font-semibold mb-4" style={{ color: 'var(--foreground)' }}>Recent Assessments</h2>
        <p style={{ color: 'var(--text-muted)' }}>No assessments run yet. Upload a video or paste a YouTube link above to begin.</p>
      </div>
    </div>
  );
}
