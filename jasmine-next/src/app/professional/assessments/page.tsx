'use client';

import { useState, useEffect } from 'react';
import { Upload, Video, Youtube, Loader2, Link2, CheckCircle, Play, Layers, BarChart3, Brain, Activity } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { getCurrentUser } from '@/lib/auth';
import { getPatients, Patient } from '@/lib/patients';

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

const pipelineStages = [
  { key: 'video', label: 'Video Input', icon: Video, desc: 'MP4 or YouTube link' },
  { key: 'pose', label: 'Pose Detection', icon: Activity, desc: 'MediaPipe → 25 body keypoints' },
  { key: 'features', label: 'Feature Extraction', icon: BarChart3, desc: 'Kinematic + Statistical features' },
  { key: 'models', label: 'ML Models', icon: Layers, desc: 'RF · SVM · LSTM · Transformer' },
  { key: 'ensemble', label: 'Ensemble', icon: Brain, desc: 'Risk score aggregation' },
];

export default function ProfessionalAssessmentsPage() {
  const [selectedPatient, setSelectedPatient] = useState('');
  const [patients, setPatients] = useState<Patient[]>([]);
  const [patientsLoading, setPatientsLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [inputMode, setInputMode] = useState<'file' | 'youtube'>('file');
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState('');
  const [currentStage, setCurrentStage] = useState(-1);
  const [showPipeline, setShowPipeline] = useState(false);

  useEffect(() => {
    const loadPatients = async () => {
      const user = getCurrentUser();
      if (!user) {
        setPatientsLoading(false);
        return;
      }
      try {
        const patientsData = await getPatients(user.id);
        setPatients(patientsData);
      } catch (err) {
        console.error('Error loading patients:', err);
      } finally {
        setPatientsLoading(false);
      }
    };
    loadPatients();
  }, []);

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

  const simulatePipeline = async () => {
    // Animate through pipeline stages
    for (let i = 0; i < pipelineStages.length; i++) {
      setCurrentStage(i);
      await new Promise(r => setTimeout(r, 800));
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
    setShowPipeline(true);
    setCurrentStage(-1);

    // Start pipeline animation
    simulatePipeline();

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
      setCurrentStage(pipelineStages.length); // mark all complete
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

          <div className="flex items-center gap-2 mb-4 p-1 rounded-xl" style={{ backgroundColor: 'var(--background-alt)' }}>
            <button onClick={() => setInputMode('file')}
              className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all"
              style={{ backgroundColor: inputMode === 'file' ? 'var(--primary)' : 'transparent', color: inputMode === 'file' ? 'white' : 'var(--text-muted)' }}>
              <Upload className="w-4 h-4" /> Upload MP4
            </button>
            <button onClick={() => setInputMode('youtube')}
              className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all"
              style={{ backgroundColor: inputMode === 'youtube' ? 'var(--primary)' : 'transparent', color: inputMode === 'youtube' ? 'white' : 'var(--text-muted)' }}>
              <Youtube className="w-4 h-4" /> YouTube URL
            </button>
          </div>

          <div className="flex flex-col gap-3 w-full max-w-md">
            <select value={selectedPatient} onChange={(e) => setSelectedPatient(e.target.value)}
              className="w-full px-4 py-3 rounded-xl"
              style={{ backgroundColor: 'var(--background)', border: '1px solid var(--border)', color: 'var(--foreground)' }}
              disabled={patientsLoading}>
              <option value="">{patientsLoading ? 'Loading patients...' : 'Select patient...'}</option>
              {!patientsLoading && patients.length === 0 && (
                <option value="" disabled>No patients found. Add patients first.</option>
              )}
              {patients.map((patient) => (
                <option key={patient.id} value={patient.id}>
                  {patient.name}
                </option>
              ))}
            </select>

            {!result && inputMode === 'file' && (
              <label className="w-full px-4 py-3 rounded-xl text-center cursor-pointer" style={{ backgroundColor: 'var(--background-alt)', border: '1px solid var(--border)' }}>
                <input type="file" accept=".mp4,.mov,.avi" onChange={handleFileChange} className="hidden" />
                <span style={{ color: videoFile ? 'var(--foreground)' : 'var(--text-muted)' }}>
                  {videoFile ? videoFile.name : 'Click to select MP4 video'}
                </span>
              </label>
            )}

            {!result && inputMode === 'youtube' && (
              <div className="relative">
                <Link2 className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5" style={{ color: 'var(--text-muted)' }} />
                <input type="url" value={youtubeUrl} onChange={(e) => { setYoutubeUrl(e.target.value); setError(''); setResult(null); }}
                  placeholder="https://youtube.com/watch?v=..."
                  className="w-full pl-12 pr-4 py-3 rounded-xl" style={{ backgroundColor: 'var(--background-alt)', border: '1px solid var(--border)', color: 'var(--foreground)' }} />
              </div>
            )}

            <button onClick={handleRunAssessment}
              disabled={!selectedPatient || uploading || (inputMode === 'file' && !videoFile) || (inputMode === 'youtube' && (!youtubeUrl.trim() || !isValidUrl(youtubeUrl)))}
              className="w-full px-6 py-3 text-white font-medium rounded-xl transition-all disabled:opacity-50 flex items-center justify-center gap-2"
              style={{ backgroundColor: 'var(--primary)' }}>
              {uploading ? <><Loader2 className="w-5 h-5 animate-spin" /> Processing...</> : <>{inputMode === 'youtube' ? <Youtube className="w-5 h-5" /> : <Upload className="w-5 h-5" />} Run Assessment</>}
            </button>
          </div>
        </div>
      </div>

      {/* Pipeline Visualization */}
      <AnimatePresence>
        {showPipeline && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }}
            className="p-6 rounded-2xl" style={{ backgroundColor: 'var(--background)', border: '1px solid var(--border)' }}>
            <h2 className="text-lg font-semibold mb-6" style={{ color: 'var(--foreground)' }}>Processing Pipeline</h2>

            <div className="relative">
              {/* Connecting line */}
              <div className="absolute top-8 left-6 right-6 h-0.5" style={{ backgroundColor: 'var(--border)' }} />

              {/* Stages */}
              <div className="flex justify-between relative">
                {pipelineStages.map((stage, i) => {
                  const isActive = currentStage >= i;
                  const isCurrent = currentStage === i;
                  const isComplete = currentStage > i || currentStage === pipelineStages.length;

                  return (
                    <motion.div key={stage.key} className="flex flex-col items-center gap-2 z-10" style={{ width: 120 }}>
                      <motion.div
                        animate={isCurrent ? { scale: [1, 1.15, 1], transition: { repeat: Infinity, duration: 1.5 } } : {}}
                        className="w-14 h-14 rounded-full flex items-center justify-center relative"
                        style={{
                          backgroundColor: isComplete ? '#16a34a' : isActive ? 'var(--primary)' : 'var(--background-alt)',
                          border: isActive ? 'none' : '2px solid var(--border)',
                        }}
                      >
                        {isComplete ? (
                          <CheckCircle className="w-6 h-6 text-white" />
                        ) : (
                          <stage.icon className="w-6 h-6" style={{ color: isActive ? 'white' : 'var(--text-muted)' }} />
                        )}
                        {isCurrent && (
                          <motion.span
                            animate={{ scale: [1, 1.3, 1], opacity: [0.5, 0, 0.5] }}
                            transition={{ repeat: Infinity, duration: 1.5 }}
                            className="absolute inset-0 rounded-full" style={{ backgroundColor: 'var(--primary)', opacity: 0.3 }}
                          />
                        )}
                      </motion.div>
                      <p className="text-xs font-medium text-center" style={{ color: isActive ? 'var(--foreground)' : 'var(--text-muted)' }}>
                        {stage.label}
                      </p>
                      <p className="text-[10px] text-center leading-tight" style={{ color: 'var(--text-muted)' }}>
                        {stage.desc}
                      </p>
                    </motion.div>
                  );
                })}
              </div>
            </div>

            {/* Current action message */}
            <motion.p key={currentStage} initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-sm text-center mt-4" style={{ color: 'var(--text-muted)' }}>
              {currentStage < pipelineStages.length && currentStage >= 0
                ? `Running: ${pipelineStages[currentStage].label} — ${pipelineStages[currentStage].desc}`
                : currentStage === pipelineStages.length
                  ? 'Pipeline complete!'
                  : 'Initializing...'}
            </motion.p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error */}
      {error && (
        <div className="p-4 rounded-xl whitespace-pre-line" style={{ backgroundColor: 'rgba(220, 38, 38, 0.1)', border: '1px solid #dc2626', color: '#dc2626' }}>
          <p className="text-sm">{error}</p>
        </div>
      )}

      {/* Results */}
      {result && (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
          className="p-6 rounded-2xl" style={{ backgroundColor: 'var(--background)', border: '1px solid var(--border)' }}>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold" style={{ color: 'var(--foreground)' }}>Assessment Result</h2>
            {result.source === 'youtube' && (
              <a href={result.youtube_url} target="_blank" rel="noopener noreferrer"
                className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs" style={{ backgroundColor: 'var(--background-alt)', color: 'var(--primary)' }}>
                <Youtube className="w-3 h-3" /> Source Video
              </a>
            )}
          </div>

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
              <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>{result.num_frames_processed} frames processed</p>
            )}
          </div>

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

      {/* Recent Assessments */}
      <div className="p-6 rounded-2xl" style={{ backgroundColor: 'var(--background)', border: '1px solid var(--border)' }}>
        <h2 className="text-lg font-semibold mb-4" style={{ color: 'var(--foreground)' }}>Recent Assessments</h2>
        <p style={{ color: 'var(--text-muted)' }}>No assessments run yet. Upload a video or paste a YouTube link above to begin.</p>
      </div>
    </div>
  );
}
