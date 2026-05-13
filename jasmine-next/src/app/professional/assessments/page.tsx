'use client';

import { useState } from 'react';
import { Upload, FileText, Clock, CheckCircle, AlertCircle, RefreshCw, Video, Loader2 } from 'lucide-react';
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
  model_predictions: Record<string, ModelPrediction>;
  error?: string;
}

export default function ProfessionalAssessmentsPage() {
  const [selectedPatient, setSelectedPatient] = useState('');
  const [uploading, setUploading] = useState(false);
  const [videoFile, setVideoFile] = useState<File | null>(null);
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
    if (!videoFile || !selectedPatient) {
      setError('Please select a patient and upload a video file.');
      return;
    }

    setUploading(true);
    setError('');
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('video', videoFile);
      formData.append('fps', '15');

      const res = await fetch(`${ML_BACKEND_URL}/api/predict`, {
        method: 'POST',
        body: formData,
      });

      const data: PredictionResult = await res.json();

      if (!data.success) {
        setError(data.error || 'Assessment failed.');
        return;
      }

      setResult(data);
    } catch {
      setError(
        'Could not connect to the ML backend. Make sure the server is running on port 8000.\n\n' +
        'Run: cd jasmine-next && pip install backend/requirements.txt && uvicorn backend.main:app --reload --port 8000'
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
          <p className="mb-4" style={{ color: 'var(--text-muted)' }}>Upload an MP4 video of the patient to analyze movement patterns</p>

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

            {!result && (
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

            <button
              onClick={handleRunAssessment}
              disabled={!videoFile || !selectedPatient || uploading}
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
                  <Upload className="w-5 h-5" />
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
          <h2 className="text-lg font-semibold mb-4" style={{ color: 'var(--foreground)' }}>Assessment Result</h2>

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
        <p style={{ color: 'var(--text-muted)' }}>No assessments run yet. Upload a video above to begin.</p>
      </div>
    </div>
  );
}