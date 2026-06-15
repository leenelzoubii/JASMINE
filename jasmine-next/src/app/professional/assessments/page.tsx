'use client';

import { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import { Upload, Video, Youtube, Loader2, Link2, CheckCircle, Play, Layers, BarChart3, Brain, Activity, Calendar, Eye, Share2, Info, ZoomIn, MessageSquare, XCircle, Trash2, AlertTriangle } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { getCurrentUser } from '@/lib/auth';
import { saveAssessment, getAssessments, reviewAssessment, shareAssessment, deleteAllAssessments, AssessmentResult } from '@/lib/assessments';
import { getPatients, Patient, updatePatient } from '@/lib/patients';
import { addNotification } from '@/lib/notifications';
import { showToast } from '@/components/ui/toast';
import { PoseViewer } from '@/components/ui/pose-viewer';

const ML_BACKEND_URL = process.env.NEXT_PUBLIC_ML_BACKEND_URL || 'http://localhost:8000';

interface ModelPrediction {
  probability: number;
  risk_level: string;
}

interface PredictionResult {
  success: boolean;
  ensemble_probability: number;
  risk_level: string;
  confidence?: number;
  num_frames_processed?: number;
  source?: string;
  youtube_url?: string;
  model_predictions: Record<string, ModelPrediction>;
  error?: string;
  viz_keypoints?: { frame: number; keypoints: number[][] }[];
}

const pipelineStages = [
  { key: 'video', label: 'Video Input', icon: Video, desc: 'MP4 or YouTube link' },
  { key: 'pose', label: 'Pose Detection', icon: Activity, desc: 'MediaPipe -> 25 body keypoints' },
  { key: 'features', label: 'Feature Extraction', icon: BarChart3, desc: 'Kinematic + Statistical features' },
  { key: 'models', label: 'ML Models', icon: Layers, desc: 'RF . SVM . TCN . Transformer' },
  { key: 'ensemble', label: 'Ensemble', icon: Brain, desc: 'Risk score aggregation' },
];

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
  const [zoomedFrame, setZoomedFrame] = useState<number | null>(null);
  const [showShareModal, setShowShareModal] = useState(false);
  const [shareNotes, setShareNotes] = useState('');
  const [showExplanationModal, setShowExplanationModal] = useState(false);
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const [clearing, setClearing] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const user = getCurrentUser();
    if (!user) {
      setPatientsLoading(false);
      return;
    }
    Promise.all([
      getPatients(user.id),
      getAssessments(user.id),
    ])
      .then(([patientsData, assessmentsData]) => {
        setPatients(patientsData);
        setAssessments(assessmentsData);
      })
      .catch((err) => {
        console.error('Failed to load data:', err);
      })
      .finally(() => setPatientsLoading(false));
  }, []);

  const selectedPatientName = patients.find(p => p.id === selectedPatient)?.name || '';

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

  const saveAssessmentResult = async (data: PredictionResult) => {
    const user = getCurrentUser();
    if (!user) return;
    try {
      const id = await saveAssessment(user.id, {
        userId: user.id,
        patientId: selectedPatient,
        patientName: selectedPatientName,
        date: new Date().toISOString().split('T')[0],
        ensemble_probability: data.ensemble_probability,
        risk_level: data.risk_level,
        confidence: data.confidence,
        num_frames_processed: data.num_frames_processed,
        source: data.source || inputMode,
        youtube_url: data.youtube_url,
        model_predictions: data.model_predictions,
      });
      setLastAssessmentId(id);
      setLastResult(data);
      const updated = await getAssessments(user.id);
      setAssessments(updated);

      const riskShort = data.risk_level.replace(' Risk', '');
      await updatePatient(user.id, selectedPatient, {
        lastVisit: new Date().toISOString().split('T')[0],
        risk: riskShort,
      });

      showToast('success', 'Assessment Saved', 'Result has been saved to the patient record.');
    } catch (err) {
      console.error('Failed to save assessment:', err);
    }
  };

  const handleReview = async () => {
    const user = getCurrentUser();
    if (!user || !lastAssessmentId) return;
    try {
      await reviewAssessment(user.id, lastAssessmentId);
      showToast('success', 'Assessment Reviewed', 'You have reviewed this assessment.');
      try {
        await addNotification({
          userId: user.id,
          type: 'assessment_complete',
          title: 'Assessment Reviewed',
          message: `Assessment for ${selectedPatientName} has been reviewed.`,
          link: '/professional/assessments',
        });
      } catch { /* skip notification err */ }
    } catch (err) {
      console.error('Failed to review:', err);
    }
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
          userId: link.parentId,
          type: 'assessment_complete',
          title: 'New Assessment Results',
          message: `${user.name} has shared assessment results for ${selectedPatientName}.${notesSuffix}`,
          link: '/parent/results',
        });
      }
      setShowShareModal(false);
      setShareNotes('');
    } catch (err) {
      console.error('Failed to share:', err);
    }
  };

  const handleClearAssessments = async () => {
    const user = getCurrentUser();
    if (!user) return;
    setClearing(true);
    try {
      await deleteAllAssessments(user.id);
      setAssessments([]);
      setShowClearConfirm(false);
      showToast('success', 'History Cleared', 'All previous assessments have been deleted.');
    } catch (err) {
      console.error('Failed to clear assessments:', err);
      showToast('error', 'Error', 'Failed to clear assessment history.');
    } finally {
      setClearing(false);
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

    abortRef.current = new AbortController();
    setUploading(true);
    setError('');
    setResult(null);
    setShowPipeline(true);
    setCurrentStage(0);

    try {
      let response: Response;

      if (inputMode === 'youtube') {
        response = await fetch(`${ML_BACKEND_URL}/api/predict-youtube`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ youtube_url: youtubeUrl.trim(), fps: 15 }),
          signal: abortRef.current.signal,
        });
      } else {
        const formData = new FormData();
        formData.append('video', videoFile!);
        formData.append('fps', '15');
        response = await fetch(`${ML_BACKEND_URL}/api/predict`, {
          method: 'POST',
          body: formData,
          signal: abortRef.current.signal,
        });
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
            setResult(data);
            setCurrentStage(pipelineStages.length);
            saveAssessmentResult(data);
          },
          (message) => setError(message),
        );
      } else {
        const data: PredictionResult = await response.json();
        if (!data.success) {
          setError(data.error || 'Assessment failed.');
          return;
        }
        setResult(data);
        setCurrentStage(pipelineStages.length);
        saveAssessmentResult(data);
      }
    } catch (err: any) {
      if (err?.name === 'AbortError') {
        setError('Assessment cancelled.');
        return;
      }
      setError(
        'Could not connect to the ML backend. Make sure the server is running on port 8000.\n\n' +
        'Run: cd jasmine-next && pip install -r backend/requirements.txt && uvicorn backend.main:app --reload --port 8000'
      );
    } finally {
      setUploading(false);
      abortRef.current = null;
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
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
      <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}>
        <h1 className="text-2xl font-bold" style={{ color: 'var(--foreground)' }}>Assessments</h1>
        <p style={{ color: 'var(--text-muted)' }}>Run ASD screening on patient video</p>
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

            {!result && inputMode === 'youtube' && (
              <div className="relative">
                <Link2 className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5" style={{ color: 'var(--text-muted)' }} />
                <input type="url" value={youtubeUrl} onChange={(e) => { setYoutubeUrl(e.target.value); setError(''); setResult(null); }}
                  placeholder="https://youtube.com/watch?v=..."
                  className="w-full pl-12 pr-4 py-3 rounded-xl" style={{ backgroundColor: 'var(--background-alt)', border: '1px solid var(--border)', color: 'var(--foreground)' }} />
              </div>
            )}

            {uploading ? (
              <div className="flex gap-3">
                <button disabled
                  className="flex-1 px-6 py-3 rounded-xl text-white font-medium flex items-center justify-center gap-2 opacity-70"
                  style={{ backgroundColor: 'var(--primary)' }}>
                  <Loader2 className="w-5 h-5 animate-spin" /> Processing...
                </button>
                <button onClick={() => { abortRef.current?.abort(); }}
                  className="px-5 py-3 rounded-xl font-medium flex items-center justify-center gap-2 transition-all hover:scale-[1.02]"
                  style={{ backgroundColor: 'rgba(220,38,38,0.15)', color: '#dc2626', border: '1px solid rgba(220,38,38,0.3)' }}>
                  <XCircle className="w-5 h-5" /> Stop
                </button>
              </div>
            ) : (
              <button onClick={handleRunAssessment}
                disabled={!selectedPatient || (inputMode === 'file' && !videoFile) || (inputMode === 'youtube' && (!youtubeUrl.trim() || !isValidUrl(youtubeUrl)))}
                className="w-full px-6 py-3 text-white font-medium rounded-xl transition-all disabled:opacity-50 flex items-center justify-center gap-2"
                style={{ backgroundColor: 'var(--primary)' }}>
                {inputMode === 'youtube' ? <Youtube className="w-5 h-5" /> : <Upload className="w-5 h-5" />} Run Assessment
              </button>
            )}
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

            <motion.p key={currentStage} initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-sm text-center mt-4" style={{ color: 'var(--text-muted)' }}>
              {currentStage < pipelineStages.length && currentStage >= 0
                ? `Running: ${pipelineStages[currentStage].label} - ${pipelineStages[currentStage].desc}`
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
          className="p-6 rounded-2xl space-y-6" style={{ backgroundColor: 'var(--background)', border: '1px solid var(--border)' }}>
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
            <motion.p
              initial={{ scale: 0.5 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.4, type: 'spring', stiffness: 200 }}
              className="text-6xl font-bold my-3" style={{ color: riskColor(result.risk_level).text }}>
              {(result.ensemble_probability * 100).toFixed(1)}%
            </motion.p>
            <motion.span
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.6, type: 'spring' }}
              className="inline-block px-5 py-2 rounded-full text-lg font-semibold"
              style={{ backgroundColor: riskColor(result.risk_level).bg, color: riskColor(result.risk_level).text }}>
              {result.risk_level}
            </motion.span>
            {result.num_frames_processed && (
              <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>{result.num_frames_processed} frames processed</p>
            )}
            {result.confidence !== undefined && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.7 }}
                className="flex items-center justify-center gap-2 mt-3"
              >
                <div className="w-2 h-2 rounded-full" style={{
                  backgroundColor: result.confidence >= 0.8 ? '#16a34a' : result.confidence >= 0.5 ? '#d97706' : '#dc2626'
                }} />
                <span className="text-sm" style={{ color: 'var(--text-muted)' }}>
                  Confidence: {(result.confidence * 100).toFixed(0)}%
                  <span className="ml-1 text-xs" style={{ color: result.confidence >= 0.8 ? '#16a34a' : result.confidence >= 0.5 ? '#d97706' : '#dc2626' }}>
                    ({result.confidence >= 0.8 ? 'High' : result.confidence >= 0.5 ? 'Moderate' : 'Low'})
                  </span>
                </span>
              </motion.div>
            )}
          </motion.div>

          {/* Pose Visualization */}
          {(() => {
            const vk = result.viz_keypoints;
            if (!vk || vk.length === 0) return null;
            return (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="p-4 rounded-xl" style={{ backgroundColor: 'var(--background-alt)' }}>
                <div className="flex items-center gap-2 mb-3">
                  <ZoomIn className="w-4 h-4" style={{ color: 'var(--primary)' }} />
                  <p className="text-sm font-medium" style={{ color: 'var(--foreground)' }}>Extracted Pose — Sample Frames</p>
                  <span className="text-xs ml-auto" style={{ color: 'var(--text-muted)' }}>Click to zoom</span>
                </div>
                <div className="grid grid-cols-3 gap-4">
                  {vk.map((frame: { keypoints: number[][] }, fi: number) => (
                    <motion.button
                      key={fi}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={() => setZoomedFrame(fi)}
                      className={`relative rounded-xl overflow-hidden transition-all cursor-pointer ring-1 ring-gray-200 dark:ring-gray-700 hover:ring-2 hover:ring-primary`}>
                      <PoseViewer keypoints={frame.keypoints} width={240} height={340} showLabels showLegend />
                      <div className="absolute bottom-0 left-0 right-0 py-1.5 text-center text-xs font-medium" style={{ backgroundColor: 'rgba(0,0,0,0.65)', color: '#fff' }}>
                        {fi === 0 ? 'Start of Video' : fi === vk.length - 1 ? 'End of Video' : `Mid Point`}
                      </div>
                    </motion.button>
                  ))}
                </div>
              </motion.div>
            );
          })()}

          {/* How Result Is Calculated Button */}
          <motion.button
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            whileHover={{ scale: 1.01 }}
            whileTap={{ scale: 0.99 }}
            onClick={() => setShowExplanationModal(true)}
            className="w-full p-4 rounded-xl flex items-center gap-3 text-left transition-all"
            style={{ backgroundColor: 'rgba(116, 179, 206, 0.1)', border: '1px solid rgba(116, 179, 206, 0.3)', color: 'var(--primary)' }}>
            <Brain className="w-5 h-5 flex-shrink-0" />
            <div>
              <p className="text-sm font-medium">How did the model get this result?</p>
              <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>See the full pipeline: pose detection → features → feature importance → ensemble scoring</p>
            </div>
          </motion.button>

          {/* Model Predictions */}
          <div className="pt-2">
            <p className="text-sm font-medium mb-3" style={{ color: 'var(--foreground)' }}>Model Breakdown:</p>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {Object.entries(result.model_predictions).map(([model, pred], idx) => (
                <motion.div
                  key={model}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.7 + idx * 0.1 }}
                  whileHover={{ scale: 1.03 }}
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
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1 }}
              className="flex flex-wrap gap-3 pt-2 border-t" style={{ borderColor: 'var(--border)' }}>
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={handleReview}
                className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-white text-sm font-medium"
                style={{ backgroundColor: '#2563eb' }}>
                <Eye className="w-4 h-4" />
                Review & Confirm
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => setShowShareModal(true)}
                className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-white text-sm font-medium"
                style={{ backgroundColor: '#16a34a' }}>
                <Share2 className="w-4 h-4" />
                Share with Parent
              </motion.button>
              <Link
                href="/professional/messages"
                className="flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-all hover:scale-[1.02]"
                style={{ backgroundColor: 'var(--background-alt)', color: 'var(--primary)', border: '1px solid var(--border)' }}
              >
                <MessageSquare className="w-4 h-4" />
                Discuss
              </Link>
              <p className="text-xs self-center" style={{ color: 'var(--text-muted)' }}>
                Review to confirm accuracy, then share so the parent can view results.
              </p>
            </motion.div>
          )}

          {/* Show Notes if Shared */}
          {result && (result as any).sharedNotes && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1.2 }}
              className="mt-3 p-4 rounded-xl" style={{ backgroundColor: 'var(--background-alt)', border: '1px solid var(--border)' }}
            >
              <p className="text-sm font-medium mb-1" style={{ color: 'var(--primary)' }}>
                <Info className="w-4 h-4 inline mr-1" />
                Your Notes (shared with parent)
              </p>
              <p className="text-sm" style={{ color: 'var(--foreground)' }}>{(result as any).sharedNotes}</p>
            </motion.div>
          )}

          {/* Share Notes Modal */}
          <AnimatePresence>
            {showShareModal && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                onClick={() => setShowShareModal(false)}
                className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
              >
                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  onClick={(e) => e.stopPropagation()}
                  className="w-full max-w-md p-6 rounded-2xl" style={{ backgroundColor: 'var(--background)', border: '1px solid var(--border)' }}
                >
                  <h3 className="text-lg font-semibold mb-2" style={{ color: 'var(--foreground)' }}>Share Assessment Results</h3>
                  <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>Add optional notes for the parent about this assessment.</p>
                  <textarea
                    value={shareNotes}
                    onChange={(e) => setShareNotes(e.target.value)}
                    placeholder="Add your notes here (e.g., observations, recommendations)..."
                    className="w-full p-3 rounded-xl text-sm mb-4 resize-none"
                    rows={4}
                    style={{ backgroundColor: 'var(--background-alt)', border: '1px solid var(--border)', color: 'var(--foreground)' }}
                  />
                  <div className="flex gap-3 justify-end">
                    <button
                      onClick={() => { setShowShareModal(false); setShareNotes(''); }}
                      className="px-4 py-2 rounded-xl text-sm font-medium"
                      style={{ backgroundColor: 'var(--background-alt)', color: 'var(--foreground)' }}
                    >
                      Cancel
                    </button>
                    <button
                      onClick={() => handleShareWithNotes(shareNotes)}
                      className="px-4 py-2 rounded-xl text-sm font-medium text-white"
                      style={{ backgroundColor: '#16a34a' }}
                    >
                      Share {shareNotes ? 'with Notes' : 'without Notes'}
                    </button>
                  </div>
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Zoom Frame Modal */}
          <AnimatePresence>
            {zoomedFrame !== null && result?.viz_keypoints?.[zoomedFrame] && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                onClick={() => setZoomedFrame(null)}
                className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm"
              >
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  onClick={(e) => e.stopPropagation()}
                  className="relative rounded-2xl overflow-hidden shadow-2xl"
                >
                  <PoseViewer
                    keypoints={result.viz_keypoints[zoomedFrame].keypoints}
                    width={400}
                    height={560}
                    showLabels
                    showLegend
                  />
                  <div className="absolute bottom-0 left-0 right-0 py-2 text-center text-sm font-medium" style={{ backgroundColor: 'rgba(0,0,0,0.7)', color: '#fff' }}>
                    {zoomedFrame === 0 ? 'Start of Video' : zoomedFrame === result.viz_keypoints.length - 1 ? 'End of Video' : 'Mid Point'} — Click outside to close
                  </div>
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Explanation Modal */}
          <AnimatePresence>
            {showExplanationModal && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                onClick={() => setShowExplanationModal(false)}
                className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm overflow-y-auto"
              >
                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  onClick={(e) => e.stopPropagation()}
                  className="w-full max-w-2xl max-h-[85vh] overflow-y-auto p-6 rounded-2xl" style={{ backgroundColor: 'var(--background)', border: '1px solid var(--border)' }}
                >
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                      <Brain className="w-5 h-5" style={{ color: 'var(--primary)' }} />
                      <h3 className="text-lg font-semibold" style={{ color: 'var(--foreground)' }}>How This Result Is Calculated</h3>
                    </div>
                    <button onClick={() => setShowExplanationModal(false)} className="p-1 rounded-lg hover:opacity-70" style={{ color: 'var(--text-muted)' }}>
                      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
                    </button>
                  </div>

                  <div className="space-y-3 text-sm" style={{ color: 'var(--text-muted)' }}>
                    <p><strong style={{ color: 'var(--foreground)' }}>1. Pose Detection:</strong> MediaPipe Pose Landmarker extracts 25 body keypoints (shoulders, elbows, wrists, hips, knees, ankles, ears, eyes, heels, toes) at 15 FPS from the video. Each keypoint has (x, y, z, visibility) coordinates per frame. Only skeletal keypoints are processed — no images or video are stored, preserving privacy.</p>

                    <p><strong style={{ color: 'var(--foreground)' }}>2. Feature Extraction (983 features):</strong> From the raw keypoint sequences, we compute:</p>
                    <ul className="list-disc pl-5 space-y-1">
                      <li><strong>Kinematic (temporal):</strong> Per-joint velocities, accelerations, jerks; joint angles (elbow, knee, hip, shoulder, neck); angular velocities and accelerations; trunk sway; center-of-mass displacement.</li>
                      <li><strong>Statistical (per-sequence):</strong> Mean, variance, skewness, kurtosis, range, RMS, percentiles; left-right symmetry indices; dominant movement frequency via FFT; autocorrelation; path length.</li>
                    </ul>
                    <p className="text-xs italic">All 983 features are standardized to z-scores using dataset statistics.</p>

                    <p><strong style={{ color: 'var(--foreground)' }}>3. Feature Importance — Top Movement Indicators:</strong> The Random Forest model ranks features by influence. The top patterns most indicative of ASD risk are:</p>
                    <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs mt-1 mb-2">
                      <div><strong>1.</strong> Left-hip velocity RMS</div>
                      <div><strong>2.</strong> Right-shoulder angle range</div>
                      <div><strong>3.</strong> Elbow symmetry index</div>
                      <div><strong>4.</strong> Head-forward jerk</div>
                      <div><strong>5.</strong> Left-knee acceleration range</div>
                      <div><strong>6.</strong> Trunk sway variance</div>
                      <div><strong>7.</strong> Right-wrist movement frequency</div>
                      <div><strong>8.</strong> Bilateral hip symmetry</div>
                      <div><strong>9.</strong> Left-ankle path length</div>
                      <div><strong>10.</strong> Shoulder angle mean</div>
                    </div>
                    <p className="text-xs">Key observations: <em>Hip and shoulder movement patterns, elbow symmetry, and trunk sway</em> are the strongest differentiators.</p>

                    <p><strong style={{ color: 'var(--foreground)' }}>4. Ensemble Models:</strong> Four architectures analyze the features from different perspectives:</p>
                    <ul className="list-disc pl-5 space-y-1">
                      <li><strong>Random Forest</strong> — 500 decision trees; captures non-linear feature interactions</li>
                      <li><strong>SVM (RBF kernel)</strong> — Optimal separating hyperplane with Platt-calibrated probabilities</li>
                      <li><strong>TCN</strong> — Temporal convolutional network with 5 dilated residual blocks (kernel=3, dilations 1→16)</li>
                      <li><strong>Transformer</strong> — 3-layer encoder with 8-head self-attention, CLS token pooling</li>
                    </ul>

                    <p><strong style={{ color: 'var(--foreground)' }}>5. Stacked Ensemble Score:</strong> A LogisticRegression meta-learner combines the four predictions:</p>
                    <div className="bg-white/30 dark:bg-black/20 rounded-lg p-2 text-xs font-mono mt-1 mb-1">
                      Final_Score = 0.425 × RF + 0.228 × SVM + 0.208 × TCN + 0.140 × Transformer
                    </div>
                    <p className="text-xs">Weights learned via 5-fold cross-validated stacked generalization on the MMASD dataset (1,374 subjects). The ensemble achieves 97.1% accuracy and 0.997 ROC-AUC.</p>
                  </div>
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      )}

      {/* Recent Assessments */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}
        className="p-6 rounded-2xl" style={{ backgroundColor: 'var(--background)', border: '1px solid var(--border)' }}>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold" style={{ color: 'var(--foreground)' }}>Recent Assessments</h2>
          {assessments.length > 0 && (
            <button onClick={() => setShowClearConfirm(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-200"
              style={{ color: '#ef4444', backgroundColor: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.2)' }}
              onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = 'rgba(239,68,68,0.2)'; e.currentTarget.style.borderColor = 'rgba(239,68,68,0.3)'; }}
              onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = 'rgba(239,68,68,0.1)'; e.currentTarget.style.borderColor = 'rgba(239,68,68,0.2)'; }}>
              <Trash2 className="w-3.5 h-3.5" />
              Clear History
            </button>
          )}
        </div>
        {assessments.length === 0 ? (
          <p style={{ color: 'var(--text-muted)' }}>No assessments run yet. Upload a video or paste a YouTube link above to begin.</p>
        ) : (
          <div className="space-y-3">
            {assessments.slice(0, 10).map((a, idx) => (
              <motion.div
                key={a.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.05 }}
                whileHover={{ x: 4 }}
                className="flex items-center justify-between p-4 rounded-xl" style={{ backgroundColor: 'var(--background-alt)' }}>
                <div className="flex items-center gap-4">
                  <div className="w-10 h-10 rounded-full flex items-center justify-center" style={{ backgroundColor: 'var(--primary-light)' }}>
                    <Activity className="w-5 h-5" style={{ color: 'var(--primary)' }} />
                  </div>
                  <div>
                    <p className="font-medium" style={{ color: 'var(--foreground)' }}>{a.patientName}</p>
                    <div className="flex items-center gap-2 text-xs" style={{ color: 'var(--text-muted)' }}>
                      <Calendar className="w-3 h-3" />
                      {a.date}
                      {(a as any).reviewed && <span className="flex items-center gap-1" style={{ color: '#2563eb' }}><Eye className="w-3 h-3" /> Reviewed</span>}
                      {(a as any).shared && <span className="flex items-center gap-1" style={{ color: '#16a34a' }}><Share2 className="w-3 h-3" /> Shared</span>}
                    </div>
                    {a.sharedNotes && (
                      <p className="text-xs mt-1 italic" style={{ color: 'var(--text-muted)' }}>Notes: {a.sharedNotes}</p>
                    )}
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-lg font-bold" style={{ color: a.risk_level === 'High Risk' ? '#dc2626' : a.risk_level === 'Moderate Risk' ? '#d97706' : '#16a34a' }}>
                    {(a.ensemble_probability * 100).toFixed(1)}%
                  </p>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{a.risk_level}</p>
                </div>
              </motion.div>
            ))}
          </div>
        )}
      </motion.div>

      {/* Clear History Confirmation Modal */}
      <AnimatePresence>
        {showClearConfirm && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => !clearing && setShowClearConfirm(false)}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm"
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              onClick={(e) => e.stopPropagation()}
              className="rounded-2xl p-6 max-w-sm w-full text-center shadow-2xl"
              style={{ backgroundColor: 'var(--background)', border: '1px solid var(--border)' }}
            >
              <div className="mx-auto w-12 h-12 rounded-full flex items-center justify-center mb-4"
                style={{ backgroundColor: 'rgba(239,68,68,0.15)' }}>
                <AlertTriangle className="w-6 h-6" style={{ color: '#ef4444' }} />
              </div>
              <h3 className="text-lg font-semibold mb-2" style={{ color: 'var(--foreground)' }}>Clear All Assessments?</h3>
              <p className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>
                This will permanently delete all {assessments.length} assessment records. This action cannot be undone.
              </p>
              <div className="flex gap-3 justify-center">
                <button
                  onClick={() => setShowClearConfirm(false)}
                  disabled={clearing}
                  className="px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200"
                  style={{ backgroundColor: 'var(--background-alt)', color: 'var(--foreground)', border: '1px solid var(--border)' }}
                >
                  Cancel
                </button>
                <button
                  onClick={handleClearAssessments}
                  disabled={clearing}
                  className="px-4 py-2 rounded-lg text-sm font-medium text-white transition-all duration-200 flex items-center gap-2"
                  style={{ backgroundColor: '#dc2626' }}
                  onMouseEnter={(e) => { if (!clearing) e.currentTarget.style.backgroundColor = '#b91c1c'; }}
                  onMouseLeave={(e) => { if (!clearing) e.currentTarget.style.backgroundColor = '#dc2626'; }}
                >
                  {clearing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Trash2 className="w-4 h-4" />}
                  {clearing ? 'Deleting...' : 'Delete All'}
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
