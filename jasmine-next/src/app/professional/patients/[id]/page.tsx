'use client';

import { useState, useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { ArrowLeft, Mail, Phone, Calendar, User, Heart, Edit2, Save, X, CheckCircle, AlertTriangle, FileText, Info } from 'lucide-react';
import { getPatient, updatePatient, Patient } from '@/lib/patients';
import { getCurrentUser } from '@/lib/auth';
import { getAssessmentsByPatient, AssessmentResult } from '@/lib/assessments';
import { showToast } from '@/components/ui/toast';

function calculateAge(dob: string): number {
  const birth = new Date(dob);
  const today = new Date();
  let age = today.getFullYear() - birth.getFullYear();
  const m = today.getMonth() - birth.getMonth();
  if (m < 0 || (m === 0 && today.getDate() < birth.getDate())) age--;
  return age;
}

const riskColors: Record<string, string> = {
  High: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400',
  Moderate: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400',
  Low: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
  Unknown: 'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400',
};

const riskColorValue: Record<string, string> = {
  'High Risk': '#dc2626',
  'Moderate Risk': '#d97706',
  'Low Risk': '#16a34a',
  Unknown: '#6b7280',
};

const container = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.08 } },
};

const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: [0.16, 1, 0.3, 1] } },
};

export default function PatientProfilePage() {
  const params = useParams();
  const router = useRouter();
  const patientId = params.id as string;

  const [patient, setPatient] = useState<Patient | null>(null);
  const [assessments, setAssessments] = useState<AssessmentResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [mounted, setMounted] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editData, setEditData] = useState<Partial<Patient>>({});
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [showCancelConfirm, setShowCancelConfirm] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    setMounted(true);
    const user = getCurrentUser();
    if (user && patientId) {
      Promise.all([
        getPatient(user.id, patientId),
        getAssessmentsByPatient(user.id, patientId),
      ])
        .then(([p, assessmentsData]) => {
          if (p) {
            setPatient(p);
            setEditData(p);
          }
          assessmentsData.sort((a, b) => {
            const tA = (a.createdAt as any)?.toMillis?.() || 0;
            const tB = (b.createdAt as any)?.toMillis?.() || 0;
            return tB - tA;
          });
          setAssessments(assessmentsData);
        })
        .catch(console.error)
        .finally(() => setLoading(false));
    } else {
      setLoading(false);
    }
  }, [patientId]);

  const handleSave = async () => {
    if (!editData.name || !editData.dob || !editData.parentName || !editData.email) {
      setError('All fields except phone are required.');
      return;
    }
    if (!editData.dob.match(/^\d{4}-\d{2}-\d{2}$/)) {
      setError('Please enter a valid date (YYYY-MM-DD).');
      return;
    }
    if (new Date(editData.dob) > new Date()) {
      setError('Date of birth cannot be in the future.');
      return;
    }
    if (editData.phone && !/^[+]?[\d\s\-()]*$/.test(editData.phone)) {
      setError('Please enter a valid phone number');
      return;
    }

    const user = getCurrentUser();
    if (!user) return;

    setSaving(true);
    setError('');
    try {
      await updatePatient(user.id, patientId, editData);
      setPatient({ ...patient, ...editData } as Patient);
      setSaved(true);
      showToast('success', 'Profile Updated', `${editData.name}'s profile has been updated.`);
      setTimeout(() => {
        setIsEditing(false);
        setSaved(false);
      }, 2000);
    } catch (err) {
      setError('Failed to update patient. Please try again.');
      console.error(err);
    } finally {
      setSaving(false);
    }
  };

  const handleCancel = () => {
    if (JSON.stringify(editData) !== JSON.stringify(patient)) {
      setShowCancelConfirm(true);
    } else {
      setIsEditing(false);
    }
  };

  const handlePhoneClick = (phone: string) => {
    if (!phone) {
      alert('Phone number not provided.');
      return;
    }
    window.location.href = `tel:${phone}`;
  };

  const handleMailClick = (email: string) => {
    window.location.href = `mailto:${email}`;
  };

  if (loading) {
    return (
      <div className="space-y-4">
        <div className="h-8 w-48 skeleton" />
        <div className="h-64 skeleton" />
        <div className="h-96 skeleton" />
      </div>
    );
  }

  if (!patient) {
    return (
      <motion.div variants={fadeUp} initial="hidden" animate="show" className="text-center py-12">
        <h1 className="text-2xl font-bold mb-2" style={{ color: 'var(--foreground)' }}>Patient Not Found</h1>
        <p style={{ color: 'var(--text-muted)' }} className="mb-6">The patient profile you're looking for doesn't exist.</p>
        <motion.button whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
          onClick={() => router.back()} className="premium-btn premium-btn-primary">
          <ArrowLeft className="w-4 h-4" /> Go Back
        </motion.button>
      </motion.div>
    );
  }

  const age = calculateAge(patient.dob);

  return (
    <motion.div variants={container} initial="hidden" animate={mounted ? 'show' : 'hidden'} className="space-y-6">
      {/* Header */}
      <motion.div variants={fadeUp} className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
            onClick={() => router.back()} className="p-2 rounded-lg transition-colors hover:bg-[var(--background-alt)]">
            <ArrowLeft className="w-5 h-5" />
          </motion.button>
          <div>
            <h1 className="text-3xl font-bold" style={{ color: 'var(--foreground)' }}>{patient.name}</h1>
            <p style={{ color: 'var(--text-muted)' }}>Patient Profile</p>
          </div>
        </div>
        {!isEditing && (
          <motion.button whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
            onClick={() => setIsEditing(true)} className="premium-btn premium-btn-primary">
            <Edit2 className="w-4 h-4" /> Edit
          </motion.button>
        )}
      </motion.div>

      {/* Main Content */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* Patient Information */}
        <motion.div variants={fadeUp} className="space-y-4">
          <div className="premium-card p-6 space-y-4">
            <h2 className="text-xl font-semibold mb-6" style={{ color: 'var(--foreground)' }}>Patient Information</h2>

            <div className="flex items-center gap-3 mb-6">
              <div className="w-16 h-16 rounded-full flex items-center justify-center text-white font-semibold text-2xl" style={{ background: 'var(--gradient-primary)' }}>
                {patient.name.charAt(0)}
              </div>
              <div>
                <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Age</p>
                <p className="text-2xl font-bold" style={{ color: 'var(--foreground)' }}>{age} years old</p>
              </div>
            </div>

            <div className="space-y-4">
              {/* Name */}
              <div>
                <label className="block text-sm font-medium mb-1.5" style={{ color: 'var(--foreground)' }}>
                  <User className="w-4 h-4 inline mr-2" />
                  Name
                </label>
                {isEditing ? (
                  <input type="text" value={editData.name || ''} onChange={(e) => setEditData(prev => ({ ...prev, name: e.target.value }))} className="premium-input" />
                ) : (
                  <p style={{ color: 'var(--text-secondary)' }}>{patient.name}</p>
                )}
              </div>

              {/* Date of Birth */}
              <div>
                <label className="block text-sm font-medium mb-1.5" style={{ color: 'var(--foreground)' }}>
                  <Calendar className="w-4 h-4 inline mr-2" />
                  Date of Birth
                </label>
                {isEditing ? (
                  <input type="date" value={editData.dob || ''} onChange={(e) => setEditData(prev => ({ ...prev, dob: e.target.value }))} className="premium-input" />
                ) : (
                  <p style={{ color: 'var(--text-secondary)' }}>{patient.dob}</p>
                )}
              </div>

              {/* Risk Level */}
              <div>
                <label className="block text-sm font-medium mb-1.5" style={{ color: 'var(--foreground)' }}>
                  <Heart className="w-4 h-4 inline mr-2" />
                  Risk Level
                </label>
                {isEditing ? (
                  <select value={editData.risk || 'Unknown'} onChange={(e) => setEditData(prev => ({ ...prev, risk: e.target.value }))} className="premium-select">
                    <option value="Unknown">Unknown</option>
                    <option value="Low">Low</option>
                    <option value="Moderate">Moderate</option>
                    <option value="High">High</option>
                  </select>
                ) : (
                  <span className={`premium-badge ${riskColors[patient.risk] || ''}`}>{patient.risk}</span>
                )}
              </div>

              {/* Last Visit */}
              <div>
                <label className="block text-sm font-medium mb-1.5" style={{ color: 'var(--foreground)' }}>
                  <Calendar className="w-4 h-4 inline mr-2" />
                  Last Visit
                </label>
                {isEditing ? (
                  <input type="date" value={editData.lastVisit || ''} onChange={(e) => setEditData(prev => ({ ...prev, lastVisit: e.target.value }))} className="premium-input" />
                ) : (
                  <p style={{ color: 'var(--text-secondary)' }}>{patient.lastVisit || 'N/A'}</p>
                )}
              </div>
            </div>
          </div>
        </motion.div>

        {/* Parent/Guardian Information */}
        <motion.div variants={fadeUp} className="space-y-4">
          <div className="premium-card p-6 space-y-4">
            <h2 className="text-xl font-semibold mb-6" style={{ color: 'var(--foreground)' }}>Parent / Guardian Information</h2>

            <div className="space-y-4">
              {/* Parent Name */}
              <div>
                <label className="block text-sm font-medium mb-1.5" style={{ color: 'var(--foreground)' }}>
                  <User className="w-4 h-4 inline mr-2" />
                  Name
                </label>
                {isEditing ? (
                  <input type="text" value={editData.parentName || ''} onChange={(e) => setEditData(prev => ({ ...prev, parentName: e.target.value }))} className="premium-input" />
                ) : (
                  <p style={{ color: 'var(--text-secondary)' }}>{patient.parentName}</p>
                )}
              </div>

              {/* Email */}
              <div>
                <label className="block text-sm font-medium mb-1.5" style={{ color: 'var(--foreground)' }}>
                  <Mail className="w-4 h-4 inline mr-2" />
                  Email
                </label>
                {isEditing ? (
                  <input type="email" value={editData.email || ''} onChange={(e) => setEditData(prev => ({ ...prev, email: e.target.value }))} className="premium-input" />
                ) : (
                  <div className="flex items-center justify-between p-3 rounded-lg" style={{ backgroundColor: 'var(--background-alt)' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>{patient.email}</span>
                    <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={() => handleMailClick(patient.email)} className="p-2 rounded-lg text-white" style={{ backgroundColor: 'var(--primary)' }}>
                      <Mail className="w-4 h-4" />
                    </motion.button>
                  </div>
                )}
              </div>

              {/* Phone */}
              <div>
                <label className="block text-sm font-medium mb-1.5" style={{ color: 'var(--foreground)' }}>
                  <Phone className="w-4 h-4 inline mr-2" />
                  Phone (optional)
                </label>
                {isEditing ? (
                  <input type="tel" value={editData.phone || ''} onChange={(e) => setEditData(prev => ({ ...prev, phone: e.target.value }))} className="premium-input" />
                ) : (
                  <div className="flex items-center justify-between p-3 rounded-lg" style={{ backgroundColor: 'var(--background-alt)' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>{patient.phone || 'Not provided'}</span>
                    {patient.phone && (
                      <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={() => handlePhoneClick(patient.phone)} className="p-2 rounded-lg text-white" style={{ backgroundColor: 'var(--risk-low)' }}>
                        <Phone className="w-4 h-4" />
                      </motion.button>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Edit Controls */}
      {isEditing && (
        <motion.div variants={fadeUp} className="sticky bottom-0 premium-card p-6 border-t" style={{ borderColor: 'var(--border-light)' }}>
          {error && (
            <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} className="p-3 rounded-xl text-sm mb-4" style={{ backgroundColor: 'var(--risk-high-bg)', color: 'var(--risk-high)' }}>
              {error}
            </motion.div>
          )}

          {showCancelConfirm && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mb-4 p-4 rounded-xl" style={{ backgroundColor: 'var(--background-alt)' }}>
              <div className="flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 flex-shrink-0 mt-0.5" style={{ color: 'var(--risk-moderate)' }} />
                <div className="flex-1">
                  <p className="font-medium mb-3" style={{ color: 'var(--foreground)' }}>Discard changes?</p>
                  <div className="flex gap-2">
                    <button onClick={() => {
                      setEditData(patient || {});
                      setShowCancelConfirm(false);
                      setIsEditing(false);
                      setError('');
                    }} className="px-4 py-2 rounded-lg text-sm font-medium" style={{ backgroundColor: 'var(--risk-high)', color: 'white' }}>
                      Yes, discard
                    </button>
                    <button onClick={() => setShowCancelConfirm(false)} className="px-4 py-2 rounded-lg text-sm font-medium" style={{ backgroundColor: 'var(--background-alt)', color: 'var(--foreground)' }}>
                      No, keep editing
                    </button>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {saved && (
            <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="mb-4 p-4 rounded-xl flex items-center gap-3" style={{ background: 'var(--gradient-primary)' }}>
              <CheckCircle className="w-5 h-5 text-white flex-shrink-0" />
              <span className="text-white font-medium">Profile updated successfully!</span>
            </motion.div>
          )}

          <div className="flex gap-3">
            <button onClick={handleCancel} disabled={saving || saved} className="premium-btn premium-btn-ghost flex-1">
              <X className="w-4 h-4" /> Cancel
            </button>
            <motion.button whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
              onClick={handleSave} disabled={saving || saved} className="premium-btn premium-btn-primary flex-1">
              {saving ? (
                <><span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" /> Saving...</>
              ) : saved ? (
                <><CheckCircle className="w-4 h-4" /> Saved!</>
              ) : (
                <><Save className="w-4 h-4" /> Save Changes</>
              )}
            </motion.button>
          </div>
        </motion.div>
      )}

      {/* Assessment History */}
      <motion.div variants={fadeUp} className="premium-card p-6">
        <h2 className="text-xl font-semibold mb-6" style={{ color: 'var(--foreground)' }}>
          <FileText className="w-5 h-5 inline mr-2" />
          Assessment History
        </h2>
        {assessments.length === 0 ? (
          <p style={{ color: 'var(--text-muted)' }}>No assessments recorded yet.</p>
        ) : (
          <div className="space-y-3">
            {assessments.map((a) => (
              <motion.div
                key={a.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="p-4 rounded-xl" style={{ backgroundColor: 'var(--background-alt)' }}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-3">
                    <FileText className="w-5 h-5" style={{ color: 'var(--primary)' }} />
                    <div>
                      <p className="font-medium" style={{ color: 'var(--foreground)' }}>
                        {(a.ensemble_probability * 100).toFixed(1)}% — {a.risk_level}
                        {a.confidence !== undefined && (
                          <span className="ml-2 text-xs font-normal" style={{ color: 'var(--text-muted)' }}>
                            ({(a.confidence * 100).toFixed(0)}% confidence)
                          </span>
                        )}
                      </p>
                      <div className="flex items-center gap-2 text-xs" style={{ color: 'var(--text-muted)' }}>
                        <Calendar className="w-3 h-3" />
                        {a.date}
                        {a.source && <span>· {a.source}</span>}
                        {a.reviewed && <span>· <CheckCircle className="w-3 h-3 inline" /> Reviewed</span>}
                        {a.shared && <span>· Shared with parent</span>}
                      </div>
                    </div>
                  </div>
                  {a.model_predictions && Object.keys(a.model_predictions).length > 0 && (
                    <div className="hidden sm:flex gap-1">
                      {Object.entries(a.model_predictions).map(([model, pred]) => (
                        <span key={model} className="px-2 py-0.5 rounded text-xs font-medium" style={{ backgroundColor: 'var(--primary-light)', color: 'var(--primary)' }}>
                          {model}: {(pred.probability * 100).toFixed(0)}%
                        </span>
                      ))}
                    </div>
                  )}
                </div>
                {a.sharedNotes && (
                  <div className="mt-2 p-3 rounded-lg" style={{ backgroundColor: 'var(--background)', border: '1px solid var(--border)' }}>
                    <p className="text-xs font-medium mb-1" style={{ color: 'var(--primary)' }}>
                      <Info className="w-3 h-3 inline mr-1" />
                      Doctor&apos;s Notes
                    </p>
                    <p className="text-sm" style={{ color: 'var(--foreground)' }}>{a.sharedNotes}</p>
                  </div>
                )}
              </motion.div>
            ))}
          </div>
        )}
      </motion.div>
    </motion.div>
  );
}
