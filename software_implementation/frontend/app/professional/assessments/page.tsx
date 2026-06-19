'use client';

import { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import { Upload, Video, Youtube, Loader2, Link2, CheckCircle, Play, Layers, BarChart3, Brain, Activity, Calendar, Eye, Share2, Info, ZoomIn, MessageSquare, Search, Printer, X, AlertTriangle, StopCircle, ChevronDown } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { getCurrentUser } from '@/lib/auth';
import { saveAssessment, getAssessments, reviewAssessment, shareAssessment, AssessmentResult } from '@/lib/assessments';
import { getPatients, Patient } from '@/lib/patients';
import { addNotification } from '@/lib/notifications';
import { showToast } from '@/components/ui/toast';
import { PoseViewer } from '@/components/ui/pose-viewer';
import { ErrorBoundary } from '@/components/ui/error-boundary';
import { isDemoUser, getDemoLinksByProfessional } from '@/lib/demo-data';

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
  viz_keypoints?: { frame: number; keypoints: number[][] }[];
  feature_importance?: Record<string, number>;
  ensemble_weights?: Record<string, number>;
  model_confidence?: Record<string, number>;
}

const pipelineStages = [
  { key: 'video', label: 'Video Input', icon: Video, desc: 'MP4 or YouTube link' },
  { key: 'pose', label: 'Pose Detection', icon: Activity, desc: 'MediaPipe -> 25 body keypoints' },
  { key: 'features', label: 'Feature Extraction', icon: BarChart3, desc: 'Kinematic + Statistical features' },
  { key: 'models', label: 'ML Models', icon: Layers, desc: 'RF . SVM . LSTM . Transformer' },
  { key: 'ensemble', label: 'Ensemble', icon: Brain, desc: 'Risk score aggregation' },
];

const ITEMS_PER_PAGE = 10;

const readSSEStream = async (
  response: Response,
  onProgress: (stage: number, message: string) => void,
  onResult: (data: PredictionResult) => void,
  onError: (message: string) => void,
) => {
  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const parts = buffer.split('\n\n');
    buffer = parts.pop() || '';

    for (const part of parts) {
      const lines = part.split('\n');
      let event = 'message';
      let dataStr = '';

      for (const line of lines) {
        if (line.startsWith('event: ')) event = line.slice(7);
        else if (line.startsWith('data: ')) dataStr = line.slice(6);
      }

      if (dataStr) {
        try {
          const parsed = JSON.parse(dataStr);
          if (event === 'progress') {
            onProgress(parsed.stage, parsed.message);
          } else if (event === 'result') {
            onResult(parsed);
          } else if (event === 'error') {
            onError(parsed.message);
          }
        } catch {
          // skip malformed JSON
        }
      }
    }
  }
};

const formatFeatureName = (name: string) => {
  return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
};

const modelExplanations = [
  { model: 'rf', label: 'Random Forest (RF)', explanation: 'Analyzes feature patterns across all extracted movement characteristics. Identifies which joint angles, velocities, and symmetry measures are most indicative of ASD.' },
  { model: 'svm', label: 'Support Vector Machine (SVM)', explanation: 'Finds the optimal boundary between typical and atypical movement patterns in the high-dimensional feature space.' },
  { model: 'lstm', label: 'LSTM (Long Short-Term Memory)', explanation: 'Processes the movement sequence temporally to detect patterns in how body positions change over time and movement flow.' },
  { model: 'transformer', label: 'Transformer', explanation: 'Uses attention mechanisms to identify which temporal segments of the movement are most diagnostically relevant.' },
];

export default function ProfessionalAssessmentsPage() {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [assessments, setAssessments] = useState<AssessmentResult[]>([]);
  const [patientsLoading, setPatientsLoading] = useState(true);
  const [selectedPatient, setSelectedPatient] = useState('');
  const [uploading, setUploading] = useState(false);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [inputMode, setInputMode] = useState<'file' | 'youtube'>('file');
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState('');
  const [currentStage, setCurrentStage] = useState(-1);
  const [showPipeline, setShowPipeline] = useState(false);
  const [lastAssessmentId, setLastAssessmentId] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<PredictionResult | null>(null);
  const [selectedSample, setSelectedSample] = useState(0);
  const [showShareModal, setShowShareModal] = useState(false);
  const [shareNotes, setShareNotes] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [showConfirmReview, setShowConfirmReview] = useState(false);
  const [showConfirmShare, setShowConfirmShare] = useState(false);
  const [showExplain, setShowExplain] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const user = getCurrentUser();
    if (!user) { setPatientsLoading(false); return; }
    if (isDemoUser(user.id)) {
      const links = getDemoLinksByProfessional() as any[];
      setPatients(links.map((l: any) => ({
        id: l.patientId, name: l.patientName, dob: '', parentName: l.parentName,
        email: '', phone: '', lastVisit: '', risk: '',
      })));
      setAssessments([]);
      setPatientsLoading(false);
    } else {
      Promise.all([getPatients(user.id), getAssessments(user.id)])
        .then(([patientsData, assessmentsData]) => { setPatients(patientsData); setAssessments(assessmentsData); })
        .catch(console.error)
        .finally(() => setPatientsLoading(false));
    }
  }, []);

  const selectedPatientName = patients.find(p => p.id === selectedPatient)?.name || '';

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!file.name.match(/\.(mp4|mov|avi)$/i)) { setError('Please select an MP4, MOV, or AVI video file.'); return; }
      setVideoFile(file); setError(''); setResult(null);
    }
  };

  const saveAssessmentResult = async (data: PredictionResult) => {
    const user = getCurrentUser();
    if (!user) return;
    try {
      const id = await saveAssessment(user.id, {
        userId: user.id, patientId: selectedPatient, patientName: selectedPatientName,
        date: new Date().toISOString().split('T')[0],
        ensemble_probability: data.ensemble_probability, risk_level: data.risk_level,
        num_frames_processed: data.num_frames_processed,
        source: data.source || inputMode, youtube_url: data.youtube_url,
        model_predictions: data.model_predictions,
      });
      setLastAssessmentId(id); setLastResult(data);
      const updated = await getAssessments(user.id);
      setAssessments(updated);
      showToast('success', 'Assessment Saved', 'Result has been saved to the patient record.');
    } catch (err) { console.error('Failed to save assessment:', err); }
  };

  const handleReview = async () => {
    const user = getCurrentUser();
    if (!user || !lastAssessmentId) return;
    try {
      await reviewAssessment(user.id, lastAssessmentId);
      showToast('success', 'Assessment Reviewed', 'You have reviewed this assessment.');
      try {
        await addNotification({
          userId: user.id, type: 'assessment_complete',
          title: 'Assessment Reviewed',
          message: `Assessment for ${selectedPatientName} has been reviewed.`,
          link: '/professional/assessments',
        });
      } catch { /* skip */ }
    } catch (err) { console.error('Failed to review:', err); }
    setShowConfirmReview(false);
  };

  const handleShareWithNotes = async (notes?: string) => {
    const user = getCurrentUser();
    if (!user || !lastAssessmentId) return;
    try {
      await shareAssessment(user.id, lastAssessmentId, notes);
      showToast('success', 'Assessment Shared', `Results for ${selectedPatientName} are now visible to the parent.`);
      const { getPatientLinksByPatientId } = await import('@/lib/patient-access');
      const links = await getPatientLinksByPatientId(selectedPatient);
      const notesSuffix = notes ? ` Notes: ${notes}` : '';
      for (const link of links) {
        await addNotification({
          userId: link.parentId, type: 'assessment_complete',
          title: 'New Assessment Results',
          message: `${user.name} has shared assessment results for ${selectedPatientName}.${notesSuffix}`,
          link: '/parent/results',
        });
      }
      setShowShareModal(false); setShowConfirmShare(false); setShareNotes('');
    } catch (err) { console.error('Failed to share:', err); }
  };

  // Upload via XHR with progress tracking (item 27)
  const uploadViaXHR = (url: string, formData: FormData, signal: AbortSignal): Promise<Response> => {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', url);
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) setUploadProgress(Math.round((e.loaded / e.total) * 100));
      };
      xhr.onload = () => {
        const headers = new Headers();
        const contentType = xhr.getResponseHeader('content-type') || '';
        headers.set('content-type', contentType);
        resolve(new Response(xhr.responseText, { status: xhr.status, statusText: xhr.statusText, headers }));
      };
      xhr.onerror = () => reject(new Error('Network error'));
      xhr.onabort = () => reject(new DOMException('Aborted', 'AbortError'));
      signal.addEventListener('abort', () => xhr.abort());
      xhr.send(formData);
    });
  };

  const handleCancel = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setUploading(false);
    setShowPipeline(false);
    setError('Assessment cancelled.');
    showToast('error', 'Cancelled', 'The assessment was cancelled.');
  };

  const handleRunAssessment = async () => {
    if (!selectedPatient) { setError('Please select a patient.'); return; }
    if (inputMode === 'file' && !videoFile) { setError('Please upload a video file.'); return; }
    if (inputMode === 'youtube' && !youtubeUrl.trim()) { setError('Please enter a YouTube URL.'); return; }

    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    setUploading(true);
    setError('');
    setResult(null);
    setShowPipeline(true);
    setCurrentStage(0);
    setUploadProgress(0);

    try {
      let response: Response;

      if (inputMode === 'youtube') {
        response = await fetch(`${ML_BACKEND_URL}/api/predict-youtube`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ youtube_url: youtubeUrl.trim(), fps: 15 }),
          signal: abortController.signal,
        });
      } else {
        const formData = new FormData();
        formData.append('video', videoFile!);
        formData.append('fps', '15');
        response = await uploadViaXHR(`${ML_BACKEND_URL}/api/predict`, formData, abortController.signal);
      }

      if (!response.ok) {
        const errData = await response.json().catch(() => null);
        setError(errData?.error || `Server error: ${response.status}`);
        return;
      }

      const contentType = response.headers.get('content-type') || '';

      if (contentType.includes('text/event-stream')) {
        await readSSEStream(
          response,
          (stage) => setCurrentStage(stage),
          (data) => {
            setResult(data); setCurrentStage(pipelineStages.length); saveAssessmentResult(data);
          },
          (message) => setError(message),
        );
      } else {
        const data: PredictionResult = await response.json();
        if (!data.success) { setError(data.error || 'Assessment failed.'); return; }
        setResult(data); setCurrentStage(pipelineStages.length); saveAssessmentResult(data);
      }
    } catch (err: any) {
      if (err.name === 'AbortError') {
        // cancelled, do nothing extra
      } else {
        setError('Could not connect to the ML backend. Make sure the server is running on port 8000.\n\nRun: cd jasmine-next && pip install -r backend/requirements.txt && uvicorn backend.main:app --reload --port 8000');
      }
    } finally {
      setUploading(false);
      abortControllerRef.current = null;
    }
  };

  const handlePrint = () => {
    window.print();
  };

  const riskColor = (risk: string) => {
    switch (risk) {
      case 'High Risk': return { bg: 'var(--risk-high-bg)', text: 'var(--risk-high)' };
      case 'Moderate Risk': return { bg: 'var(--risk-moderate-bg)', text: 'var(--risk-moderate)' };
      default: return { bg: 'var(--risk-low-bg)', text: 'var(--risk-low)' };
    }
  };

  const isValidUrl = (url: string) => url.match(/^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.be)\/.+/i);

  // Filtered + paginated assessments
  const filteredAssessments = assessments.filter(a =>
    a.patientName?.toLowerCase().includes(searchQuery.toLowerCase())
  );
  const totalPages = Math.ceil(filteredAssessments.length / ITEMS_PER_PAGE);
  const paginatedAssessments = filteredAssessments.slice(
    (currentPage - 1) * ITEMS_PER_PAGE,
    currentPage * ITEMS_PER_PAGE
  );

  return (
    <ErrorBoundary>
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
      <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}
        className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: 'var(--foreground)' }}>Assessments</h1>
          <p style={{ color: 'var(--text-muted)' }}>Run ASD screening on patient video</p>
        </div>
        {result && (
          <button onClick={handlePrint}
            className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all"
            style={{ backgroundColor: 'var(--background-alt)', color: 'var(--foreground)', border: '1px solid var(--border)' }}>
            <Printer className="w-4 h-4" /> Download / Print
          </button>
        )}
      </motion.div>

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
                <option key={patient.id} value={patient.id}>{patient.name}</option>
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

            {/* Upload progress bar */}
            {uploading && inputMode === 'file' && uploadProgress > 0 && uploadProgress < 100 && (
              <div className="w-full">
                <div className="flex justify-between text-xs mb-1" style={{ color: 'var(--text-muted)' }}>
                  <span>Uploading...</span><span>{uploadProgress}%</span>
                </div>
                <div className="w-full h-2 rounded-full" style={{ backgroundColor: 'var(--background-alt)' }}>
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${uploadProgress}%` }}
                    className="h-full rounded-full" style={{ backgroundColor: 'var(--primary)' }}
                  />
                </div>
              </div>
            )}

            {!result && inputMode === 'youtube' && (
              <div className="relative">
                <Link2 className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5" style={{ color: 'var(--text-muted)' }} />
                <input type="url" value={youtubeUrl} onChange={(e) => { setYoutubeUrl(e.target.value); setError(''); setResult(null); }}
                  placeholder="https://youtube.com/watch?v=..."
                  className="w-full pl-12 pr-4 py-3 rounded-xl" style={{ backgroundColor: 'var(--background-alt)', border: '1px solid var(--border)', color: 'var(--foreground)' }} />
              </div>
            )}

            <div className="flex gap-2">
              <button onClick={handleRunAssessment}
                disabled={!selectedPatient || uploading || (inputMode === 'file' && !videoFile) || (inputMode === 'youtube' && (!youtubeUrl.trim() || !isValidUrl(youtubeUrl)))}
                className="flex-1 px-6 py-3 text-white font-medium rounded-xl transition-all disabled:opacity-50 flex items-center justify-center gap-2"
                style={{ backgroundColor: 'var(--primary)' }}>
                {uploading ? <><Loader2 className="w-5 h-5 animate-spin" /> Processing...</> : <>{inputMode === 'youtube' ? <Youtube className="w-5 h-5" /> : <Upload className="w-5 h-5" />} Run Assessment</>}
              </button>
              {uploading && (
                <button onClick={handleCancel}
                  className="px-4 py-3 rounded-xl text-white font-medium flex items-center gap-2"
                  style={{ backgroundColor: 'var(--risk-high)' }}>
                  <StopCircle className="w-5 h-5" />
                </button>
              )}
            </div>
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
              <div className="absolute top-8 left-6 right-6 h-0.5" style={{ backgroundColor: 'var(--border)' }} />
              <div className="flex justify-between relative">
                {pipelineStages.map((stage, i) => {
                  const isActive = currentStage >= i;
                  const isCurrent = currentStage === i;
                  const isComplete = currentStage > i || currentStage === pipelineStages.length;
                  return (
                    <div key={stage.key} className="flex flex-col items-center gap-2 z-10" style={{ width: 120 }}>
                      <motion.div
                        animate={isCurrent ? { scale: [1, 1.15, 1], transition: { repeat: Infinity, duration: 1.5 } } : {}}
                        className="w-14 h-14 rounded-full flex items-center justify-center relative"
                        style={{
                          backgroundColor: isComplete ? 'var(--risk-low)' : isActive ? 'var(--primary)' : 'var(--background-alt)',
                          border: isActive ? 'none' : '2px solid var(--border)',
                        }}>
                        {isComplete ? <CheckCircle className="w-6 h-6 text-white" /> : <stage.icon className="w-6 h-6" style={{ color: isActive ? 'white' : 'var(--text-muted)' }} />}
                        {isCurrent && <motion.span animate={{ scale: [1, 1.3, 1], opacity: [0.5, 0, 0.5] }} transition={{ repeat: Infinity, duration: 1.5 }} className="absolute inset-0 rounded-full" style={{ backgroundColor: 'var(--primary)', opacity: 0.3 }} />}
                      </motion.div>
                      <p className="text-xs font-medium text-center" style={{ color: isActive ? 'var(--foreground)' : 'var(--text-muted)' }}>{stage.label}</p>
                      <p className="text-[10px] text-center leading-tight" style={{ color: 'var(--text-muted)' }}>{stage.desc}</p>
                    </div>
                  );
                })}
              </div>
            </div>
            <motion.p key={currentStage} initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-sm text-center mt-4" style={{ color: 'var(--text-muted)' }}>
              {currentStage < pipelineStages.length && currentStage >= 0
                ? `Running: ${pipelineStages[currentStage].label} - ${pipelineStages[currentStage].desc}`
                : currentStage === pipelineStages.length ? 'Pipeline complete!' : 'Initializing...'}
            </motion.p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error */}
      {error && (
        <div className="p-4 rounded-xl whitespace-pre-line" style={{ backgroundColor: 'var(--risk-high-bg)', border: '1px solid var(--risk-high)', color: 'var(--risk-high)' }}>
          <p className="text-sm">{error}</p>
        </div>
      )}

      {/* Results */}
      {result && (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
          className="p-6 rounded-2xl space-y-6 print:break-inside-avoid" style={{ backgroundColor: 'var(--background)', border: '1px solid var(--border)' }}>
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold" style={{ color: 'var(--foreground)' }}>Assessment Result</h2>
            {result.source === 'youtube' && (
              <a href={result.youtube_url} target="_blank" rel="noopener noreferrer"
                className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs" style={{ backgroundColor: 'var(--background-alt)', color: 'var(--primary)' }}>
                <Youtube className="w-3 h-3" /> Source Video
              </a>
            )}
          </div>

          {/* Score */}
          <motion.div initial={{ scale: 0.8, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ delay: 0.2, type: 'spring' }}
            className="text-center py-4">
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Ensemble ASD Probability</p>
            <motion.p initial={{ scale: 0.5 }} animate={{ scale: 1 }} transition={{ delay: 0.4, type: 'spring', stiffness: 200 }}
              className="text-6xl font-bold my-3" style={{ color: riskColor(result.risk_level).text }}>
              {(result.ensemble_probability * 100).toFixed(1)}%
            </motion.p>
            <motion.span initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: 0.6, type: 'spring' }}
              className="inline-block px-5 py-2 rounded-full text-lg font-semibold"
              style={{ backgroundColor: riskColor(result.risk_level).bg, color: riskColor(result.risk_level).text }}>
              {result.risk_level}
            </motion.span>
            {result.num_frames_processed && (
              <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>{result.num_frames_processed} frames processed</p>
            )}
          </motion.div>

          {/* Pose Visualization */}
          {(() => {
            const vk = result.viz_keypoints;
            if (!vk || vk.length === 0) return null;
            return (
              <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}
                className="p-4 rounded-xl" style={{ backgroundColor: 'var(--background-alt)' }}>
                <div className="flex items-center gap-2 mb-3">
                  <ZoomIn className="w-4 h-4" style={{ color: 'var(--primary)' }} />
                  <p className="text-sm font-medium" style={{ color: 'var(--foreground)' }}>Extracted Pose — Sample Frames</p>
                </div>
                <div className="flex items-center gap-3 overflow-x-auto pb-2">
                  {vk.map((frame: { keypoints: number[][] }, fi: number) => (
                    <motion.button key={fi} whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
                      onClick={() => setSelectedSample(fi)}
                      className={`relative rounded-xl overflow-hidden transition-all ${selectedSample === fi ? 'ring-2 ring-primary' : 'ring-1 ring-gray-200 dark:ring-gray-700'}`}>
                      <PoseViewer keypoints={frame.keypoints} width={160} height={220} />
                      <div className="absolute bottom-0 left-0 right-0 py-1 text-center text-[10px] font-medium" style={{ backgroundColor: 'rgba(0,0,0,0.6)', color: '#fff' }}>
                        Frame {fi === 0 ? 'Start' : fi === vk.length - 1 ? 'End' : `Mid ${fi + 1}`}
                      </div>
                    </motion.button>
                  ))}
                </div>
                <div className="mt-3 flex justify-center">
                  <motion.div key={selectedSample} initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ type: 'spring' }}>
                    {vk[selectedSample] && <PoseViewer keypoints={vk[selectedSample].keypoints} width={200} height={280} />}
                  </motion.div>
                </div>
              </motion.div>
            );
          })()}

          {/* Model Explainability */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }}
            className="p-4 rounded-xl print:break-inside-avoid" style={{ backgroundColor: 'rgba(116, 179, 206, 0.1)', border: '1px solid rgba(116, 179, 206, 0.3)' }}>
            <button onClick={() => setShowExplain(!showExplain)}
              className="w-full flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <Brain className="w-4 h-4" style={{ color: 'var(--primary)' }} />
                <p className="text-sm font-medium" style={{ color: 'var(--foreground)' }}>How Did the Model Decide?</p>
              </div>
              <ChevronDown className={`w-4 h-4 transition-transform ${showExplain ? 'rotate-180' : ''}`} style={{ color: 'var(--text-muted)' }} />
            </button>

            <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>
              Weighted ensemble of 4 models analyzing body movement patterns from video.
            </p>

            {showExplain && (
              <motion.div initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }}
                className="mt-4 space-y-4">

                {/* Feature Importance */}
                {result.feature_importance && Object.keys(result.feature_importance).length > 0 && (() => {
                  const entries = Object.entries(result.feature_importance!);
                  const maxImp = Math.max(...entries.map(([, v]) => v), 0.01);
                  return (
                    <div className="pt-3 border-t" style={{ borderColor: 'rgba(116, 179, 206, 0.2)' }}>
                      <p className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: 'var(--text-muted)' }}>Top Contributing Features</p>
                      <div className="space-y-2">
                        {entries.map(([name, importance]) => (
                          <div key={name} className="flex items-center gap-2">
                            <span className="text-xs truncate flex-1 min-w-0" style={{ color: 'var(--foreground)' }} title={name}>
                              {formatFeatureName(name)}
                            </span>
                            <div className="flex-1 h-2.5 rounded-full" style={{ backgroundColor: 'var(--background-alt)' }}>
                              <motion.div initial={{ width: 0 }} animate={{ width: `${(importance / maxImp) * 100}%` }}
                                className="h-full rounded-full transition-all" style={{ backgroundColor: 'var(--primary)', opacity: 0.75 }} />
                            </div>
                            <span className="text-xs w-10 text-right shrink-0" style={{ color: 'var(--text-muted)' }}>
                              {(importance * 100).toFixed(1)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })()}

                {/* Ensemble Weights */}
                {result.ensemble_weights && (
                  <div className="pt-3 border-t" style={{ borderColor: 'rgba(116, 179, 206, 0.2)' }}>
                    <p className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: 'var(--text-muted)' }}>Ensemble Model Contribution</p>
                    <div className="grid grid-cols-2 gap-2">
                      {Object.entries(result.ensemble_weights).map(([model, weight]) => {
                        const pred = result.model_predictions[model];
                        const conf = result.model_confidence?.[model];
                        return (
                          <div key={model} className="p-2.5 rounded-lg" style={{ backgroundColor: 'var(--background-alt)' }}>
                            <div className="flex justify-between items-center mb-1">
                              <span className="text-xs font-semibold" style={{ color: 'var(--foreground)' }}>{model.toUpperCase()}</span>
                              <span className="text-xs" style={{ color: 'var(--primary)' }}>{(weight * 100).toFixed(0)}% weight</span>
                            </div>
                            <div className="h-2 rounded-full" style={{ backgroundColor: 'var(--background)' }}>
                              <motion.div initial={{ width: 0 }} animate={{ width: `${(pred?.probability || 0) * 100}%` }}
                                className="h-full rounded-full"
                                style={{ backgroundColor: riskColor(pred?.risk_level || 'Low Risk').text }} />
                            </div>
                            <div className="flex justify-between mt-1">
                              <span className="text-xs font-medium" style={{ color: riskColor(pred?.risk_level || 'Low Risk').text }}>
                                {((pred?.probability || 0) * 100).toFixed(0)}% ASD probability
                              </span>
                              {conf !== undefined && (
                                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                                  {(conf * 100).toFixed(0)}% confidence
                                </span>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Per-Model Reasoning */}
                <div className="pt-3 border-t" style={{ borderColor: 'rgba(116, 179, 206, 0.2)' }}>
                  <p className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: 'var(--text-muted)' }}>What Each Model Analyzes</p>
                  <div className="space-y-2">
                    {modelExplanations.map(({ model, label, explanation }) => {
                      const pred = result.model_predictions[model];
                      return (
                        <div key={model} className="p-2.5 rounded-lg" style={{ backgroundColor: 'var(--background-alt)' }}>
                          <div className="flex justify-between items-center mb-0.5">
                            <p className="text-xs font-medium" style={{ color: 'var(--foreground)' }}>{label}</p>
                            {pred && (
                              <span className="text-xs font-medium" style={{ color: riskColor(pred.risk_level).text }}>
                                {pred.risk_level}
                              </span>
                            )}
                          </div>
                          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{explanation}</p>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Pipeline Summary */}
                <div className="pt-3 border-t text-xs leading-relaxed" style={{ borderColor: 'rgba(116, 179, 206, 0.2)', color: 'var(--text-muted)' }}>
                  <strong style={{ color: 'var(--foreground)' }}>Pipeline:</strong> Video → Pose Detection (MediaPipe, 25 keypoints) → Feature Extraction (kinematic + statistical, ~983 features) → 4 Models → Weighted Ensemble → Score
                </div>
              </motion.div>
            )}
          </motion.div>

          {/* Model Predictions */}
          <div className="pt-2">
            <p className="text-sm font-medium mb-3" style={{ color: 'var(--foreground)' }}>Model Breakdown:</p>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {Object.entries(result.model_predictions).map(([model, pred], idx) => (
                <motion.div key={model} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.7 + idx * 0.1 }} whileHover={{ scale: 1.03 }}
                  className="p-3 rounded-lg" style={{ backgroundColor: 'var(--background-alt)' }}>
                  <p className="text-xs uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>{model}</p>
                  <p className="text-lg font-semibold" style={{ color: 'var(--foreground)' }}>{(pred.probability * 100).toFixed(0)}%</p>
                  <p className="text-xs" style={{ color: riskColor(pred.risk_level).text }}>{pred.risk_level}</p>
                </motion.div>
              ))}
            </div>
          </div>

          {/* Review & Share */}
          {lastAssessmentId && (
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 1 }}
              className="flex flex-wrap gap-3 pt-2 border-t print:hidden" style={{ borderColor: 'var(--border)' }}>
              <motion.button whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
                onClick={() => setShowConfirmReview(true)}
                className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-white text-sm font-medium"
                style={{ backgroundColor: 'var(--primary)' }}>
                <Eye className="w-4 h-4" /> Review & Confirm
              </motion.button>
              <motion.button whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
                onClick={() => setShowConfirmShare(true)}
                className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-white text-sm font-medium"
                style={{ backgroundColor: 'var(--primary-dark)' }}>
                <Share2 className="w-4 h-4" /> Share with Parent
              </motion.button>
              <Link href="/professional/messages"
                className="flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-all hover:scale-[1.02]"
                style={{ backgroundColor: 'var(--background-alt)', color: 'var(--primary)', border: '1px solid var(--border)' }}>
                <MessageSquare className="w-4 h-4" /> Discuss
              </Link>
              <p className="text-xs self-center" style={{ color: 'var(--text-muted)' }}>Review to confirm accuracy, then share so the parent can view results.</p>
            </motion.div>
          )}

          {/* Review Confirmation Modal */}
          <AnimatePresence>
            {showConfirmReview && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                onClick={() => setShowConfirmReview(false)}
                className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
                <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.95 }}
                  onClick={(e) => e.stopPropagation()}
                  className="w-full max-w-sm p-6 rounded-2xl text-center" style={{ backgroundColor: 'var(--background)', border: '1px solid var(--border)' }}>
                  <div className="w-16 h-16 mx-auto mb-4 rounded-full flex items-center justify-center" style={{ backgroundColor: 'var(--primary-light)' }}>
                    <Eye className="w-8 h-8" style={{ color: 'var(--primary)' }} />
                  </div>
                  <h3 className="text-lg font-semibold mb-2" style={{ color: 'var(--foreground)' }}>Review Assessment</h3>
                  <p className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>Confirm that you have reviewed this assessment for {selectedPatientName}.</p>
                  <div className="flex gap-3 justify-center">
                    <button onClick={() => setShowConfirmReview(false)}
                      className="px-4 py-2 rounded-xl text-sm font-medium" style={{ backgroundColor: 'var(--background-alt)', color: 'var(--foreground)' }}>Cancel</button>
                    <button onClick={handleReview}
                      className="px-4 py-2 rounded-xl text-sm font-medium text-white" style={{ backgroundColor: 'var(--primary)' }}>Confirm Review</button>
                  </div>
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Share Confirmation Modal */}
          <AnimatePresence>
            {showConfirmShare && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                onClick={() => setShowConfirmShare(false)}
                className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
                <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.95 }}
                  onClick={(e) => e.stopPropagation()}
                  className="w-full max-w-md p-6 rounded-2xl" style={{ backgroundColor: 'var(--background)', border: '1px solid var(--border)' }}>
                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-12 h-12 rounded-full flex items-center justify-center shrink-0" style={{ backgroundColor: 'var(--primary-light)' }}>
                      <Share2 className="w-6 h-6" style={{ color: 'var(--primary-dark)' }} />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold" style={{ color: 'var(--foreground)' }}>Share with Parent</h3>
                      <p className="text-sm" style={{ color: 'var(--text-muted)' }}>The parent will be able to see all assessment details, including scores and model breakdowns.</p>
                    </div>
                  </div>
                  <textarea value={shareNotes} onChange={(e) => setShareNotes(e.target.value)}
                    placeholder="Add optional notes for the parent (e.g., observations, recommendations)..."
                    className="w-full p-3 rounded-xl text-sm mb-4 resize-none" rows={4}
                    style={{ backgroundColor: 'var(--background-alt)', border: '1px solid var(--border)', color: 'var(--foreground)' }} />
                  <div className="flex gap-3 justify-end">
                    <button onClick={() => { setShowConfirmShare(false); setShareNotes(''); }}
                      className="px-4 py-2 rounded-xl text-sm font-medium" style={{ backgroundColor: 'var(--background-alt)', color: 'var(--foreground)' }}>Cancel</button>
                    <button onClick={() => handleShareWithNotes(shareNotes)}
                      className="px-4 py-2 rounded-xl text-sm font-medium text-white" style={{ backgroundColor: 'var(--primary-dark)' }}>
                      Share {shareNotes ? 'with Notes' : 'without Notes'}
                    </button>
                  </div>
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Show Notes if Shared */}
          {result && (result as any).sharedNotes && (
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 1.2 }}
              className="mt-3 p-4 rounded-xl" style={{ backgroundColor: 'var(--background-alt)', border: '1px solid var(--border)' }}>
              <p className="text-sm font-medium mb-1" style={{ color: 'var(--primary)' }}>
                <Info className="w-4 h-4 inline mr-1" /> Your Notes (shared with parent)
              </p>
              <p className="text-sm" style={{ color: 'var(--foreground)' }}>{(result as any).sharedNotes}</p>
            </motion.div>
          )}
        </motion.div>
      )}

      {/* Recent Assessments */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}
        className="p-6 rounded-2xl" style={{ backgroundColor: 'var(--background)', border: '1px solid var(--border)' }}>
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-4">
          <h2 className="text-lg font-semibold" style={{ color: 'var(--foreground)' }}>Recent Assessments</h2>
          {/* Search */}
          <div className="relative max-w-xs w-full">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4" style={{ color: 'var(--text-muted)' }} />
            <input type="text" placeholder="Search by patient name..." value={searchQuery} onChange={e => { setSearchQuery(e.target.value); setCurrentPage(1); }}
              className="w-full pl-10 pr-4 py-2 rounded-lg text-sm"
              style={{ backgroundColor: 'var(--background-alt)', border: '1px solid var(--border)', color: 'var(--foreground)' }} />
          </div>
        </div>

        {assessments.length === 0 ? (
          <div className="py-12 text-center">
            <Video className="w-12 h-12 mx-auto mb-4" style={{ color: 'var(--text-muted)' }} />
            <p className="text-lg font-medium" style={{ color: 'var(--foreground)' }}>No assessments yet</p>
            <p className="text-sm mt-1" style={{ color: 'var(--text-muted)' }}>Upload a video or paste a YouTube link above to begin your first assessment.</p>
            <button onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
              className="mt-4 px-4 py-2 rounded-xl text-sm font-medium text-white" style={{ backgroundColor: 'var(--primary)' }}>
              Run Your First Assessment
            </button>
          </div>
        ) : filteredAssessments.length === 0 ? (
          <div className="py-8 text-center">
            <Search className="w-8 h-8 mx-auto mb-2" style={{ color: 'var(--text-muted)' }} />
            <p style={{ color: 'var(--text-muted)' }}>No assessments match your search.</p>
          </div>
        ) : (
          <>
            <div className="space-y-3">
              {paginatedAssessments.map((a, idx) => (
                <motion.div key={a.id} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: idx * 0.05 }} whileHover={{ x: 4 }}
                  className="flex items-center justify-between p-4 rounded-xl" style={{ backgroundColor: 'var(--background-alt)' }}>
                  <div className="flex items-center gap-4 min-w-0">
                    <div className="w-10 h-10 rounded-full flex items-center justify-center shrink-0" style={{ backgroundColor: 'var(--primary-light)' }}>
                      <Activity className="w-5 h-5" style={{ color: 'var(--primary)' }} />
                    </div>
                    <div className="min-w-0">
                      <p className="font-medium truncate" style={{ color: 'var(--foreground)' }}>{a.patientName}</p>
                      <div className="flex items-center gap-2 text-xs flex-wrap" style={{ color: 'var(--text-muted)' }}>
                        <Calendar className="w-3 h-3 shrink-0" />{a.date}
                        {(a as any).reviewed && <span className="flex items-center gap-1" style={{ color: 'var(--primary)' }}><Eye className="w-3 h-3" /> Reviewed</span>}
                        {(a as any).shared && <span className="flex items-center gap-1" style={{ color: 'var(--primary-dark)' }}><Share2 className="w-3 h-3" /> Shared</span>}
                      </div>
                      {a.sharedNotes && <p className="text-xs mt-1 italic truncate" style={{ color: 'var(--text-muted)' }}>Notes: {a.sharedNotes}</p>}
                    </div>
                  </div>
                  <div className="text-right shrink-0 ml-3">
                    <p className="text-lg font-bold" style={{ color: a.risk_level === 'High Risk' ? 'var(--risk-high)' : a.risk_level === 'Moderate Risk' ? 'var(--risk-moderate)' : 'var(--risk-low)' }}>
                      {(a.ensemble_probability * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{a.risk_level}</p>
                  </div>
                </motion.div>
              ))}
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex items-center justify-center gap-2 mt-6">
                <button onClick={() => setCurrentPage(p => Math.max(1, p - 1))} disabled={currentPage === 1}
                  className="px-3 py-1.5 rounded-lg text-sm disabled:opacity-30"
                  style={{ backgroundColor: 'var(--background-alt)', color: 'var(--foreground)' }}>
                  Previous
                </button>
                <span className="text-sm" style={{ color: 'var(--text-muted)' }}>
                  Page {currentPage} of {totalPages}
                </span>
                <button onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))} disabled={currentPage === totalPages}
                  className="px-3 py-1.5 rounded-lg text-sm disabled:opacity-30"
                  style={{ backgroundColor: 'var(--background-alt)', color: 'var(--foreground)' }}>
                  Next
                </button>
              </div>
            )}
          </>
        )}
      </motion.div>
    </motion.div>
    </ErrorBoundary>
  );
}
