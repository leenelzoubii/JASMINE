'use client';

import { UserPlus, Search, Phone, Mail, X, Send, CheckCircle, Trash2, AlertTriangle } from 'lucide-react';
import Link from 'next/link';
import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { getPatients, addPatient, deletePatient, Patient } from '@/lib/patients';
import { getCurrentUser } from '@/lib/auth';
import { createPatientAccess } from '@/lib/patient-access';
import { sendParentRequest } from '@/lib/parent-requests';
import { addNotification } from '@/lib/notifications';
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

export default function ProfessionalPatientsPage() {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [loading, setLoading] = useState(true);
  const [mounted, setMounted] = useState(false);
  const [search, setSearch] = useState('');
  const [showAddModal, setShowAddModal] = useState(false);
  const [formData, setFormData] = useState({ name: '', dob: '', parentName: '', email: '', phone: '' });
  const [sendCredentials, setSendCredentials] = useState(true);
  const [formError, setFormError] = useState('');
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [savedMessage, setSavedMessage] = useState('');
  const [shake, setShake] = useState(false);
  const [showCancelConfirm, setShowCancelConfirm] = useState(false);
  const [sortField, setSortField] = useState('lastVisit');
  const [sortAsc, setSortAsc] = useState(false);
  const [visitFilter, setVisitFilter] = useState('all');
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [patientToDelete, setPatientToDelete] = useState<Patient | null>(null);

  useEffect(() => {
    setMounted(true);
    const user = getCurrentUser();
    if (user) {
      getPatients(user.id).then(setPatients).catch(console.error).finally(() => setLoading(false));
    } else { setLoading(false); }
  }, []);

  const handleAddPatient = async () => {
    if (!formData.name || !formData.dob || !formData.parentName || !formData.email) {
      setFormError('All fields except phone are required.'); return;
    }
    if (!formData.dob.match(/^\d{4}-\d{2}-\d{2}$/)) { setFormError('Please enter a valid date (YYYY-MM-DD).'); return; }
    if (new Date(formData.dob) > new Date()) { setFormError('Date of birth cannot be in the future.'); return; }
    if (formData.phone && !/^[+]?[\d\s\-()]*$/.test(formData.phone)) { setFormError('Please enter a valid phone number'); return; }

    const user = getCurrentUser();
    if (!user) return;
    const isDemo = user.id === 'demo-doctor' || user.id === 'demo-parent';
    setSaving(true);
    try {
      const newPatient = await addPatient(user.id, {
        name: formData.name, dob: formData.dob, parentName: formData.parentName,
        email: formData.email, phone: formData.phone || '',
        lastVisit: new Date().toISOString().split('T')[0], risk: 'Unknown',
      });
      setPatients(prev => [newPatient, ...prev]);
      if (!isDemo) {
        try { await sendParentRequest({ professionalId: user.id, professionalName: user.name, patientId: newPatient.id, patientName: newPatient.name, parentEmail: formData.email, parentName: formData.parentName }); } catch {}
      }
      try { await addNotification({ userId: user.id, type: 'patient_added', title: 'Patient Added', message: `${newPatient.name} has been added successfully.`, link: '/professional/patients' }); } catch {}
      try {
        if (sendCredentials) {
          const accessResult = await createPatientAccess({ patientId: newPatient.id, patientName: newPatient.name, professionalId: user.id, professionalName: user.name, parentName: formData.parentName, parentEmail: formData.email });
          if (accessResult.success && accessResult.parentTempPassword) setSavedMessage('Patient added! Account credentials sent to parent.');
          else if (accessResult.success) setSavedMessage('Patient added! Parent already has access.');
          else setSavedMessage('Patient added, but failed to send credentials.');
        } else { setSavedMessage('Patient added successfully.'); }
      } catch { setSavedMessage('Patient added, but failed to create access.'); }
      setSaved(true);
      showToast('success', 'Patient Added', `${newPatient.name} has been added successfully.`);
      setFormData({ name: '', dob: '', parentName: '', email: '', phone: '' });
      setFormError('');
      setTimeout(() => { setShowAddModal(false); setSaved(false); setSavedMessage(''); }, 2000);
    } catch (err) {
      console.error('[AddPatient] Error:', err);
      const msg = err instanceof Error ? err.message : 'Failed to add patient. Please try again.';
      setFormError(msg);
    }
    finally { setSaving(false); }
  };

  const hasFormData = formData.name || formData.dob || formData.parentName || formData.email || formData.phone;
  const closeModal = () => { setShowAddModal(false); setFormError(''); setFormData({ name: '', dob: '', parentName: '', email: '', phone: '' }); setShowCancelConfirm(false); setSaved(false); setSavedMessage(''); };
  const handleCancelClick = () => { if (hasFormData) setShowCancelConfirm(true); else closeModal(); };
  const handlePhoneClick = (phone: string) => { if (!phone) { alert('Phone number not provided.'); return; } window.location.href = `tel:${phone}`; };
  const handleMailClick = (email: string) => { window.location.href = `mailto:${email}`; };
  const getAge = (patient: Patient): number => patient.dob ? calculateAge(patient.dob) : 0;

  const displayedPatients = mounted ? patients : [];
  const filteredPatients = displayedPatients.filter(p => {
    const matchesSearch = p.name.toLowerCase().includes(search.toLowerCase()) || p.parentName?.toLowerCase().includes(search.toLowerCase());
    if (visitFilter === 'all') return matchesSearch;
    const lastVisit = p.lastVisit ? new Date(p.lastVisit) : null;
    if (!lastVisit) return matchesSearch;
    const now = new Date();
    const daysDiff = Math.floor((now.getTime() - lastVisit.getTime()) / (1000 * 60 * 60 * 24));
    switch (visitFilter) {
      case '7days': return matchesSearch && daysDiff <= 7;
      case '30days': return matchesSearch && daysDiff <= 30;
      case '90days': return matchesSearch && daysDiff <= 90;
      case 'year': return matchesSearch && daysDiff <= 365;
      default: return matchesSearch;
    }
  });

  const riskOrder: Record<string, number> = { 'High': 3, 'Moderate': 2, 'Low': 1, 'Unknown': 0 };
  const getCreatedAt = (patient: Patient): number => {
    if (!patient.createdAt) return 0;
    if (typeof patient.createdAt === 'number') return patient.createdAt;
    if (typeof patient.createdAt === 'object' && patient.createdAt && 'seconds' in patient.createdAt) return (patient.createdAt as { seconds: number }).seconds;
    return 0;
  };
  const getSortValue = (patient: Patient): number | string => {
    switch (sortField) {
      case 'name': return patient.name.toLowerCase();
      case 'age': return getAge(patient);
      case 'createdAt': return getCreatedAt(patient);
      case 'lastVisit': return patient.lastVisit || '';
      case 'risk': return riskOrder[patient.risk || 'Unknown'];
      default: return getCreatedAt(patient);
    }
  };
  const sortedPatients = [...filteredPatients].sort((a, b) => {
    const aVal = getSortValue(a), bVal = getSortValue(b);
    let result = 0;
    if (typeof aVal === 'string' && typeof bVal === 'string') result = aVal.localeCompare(bVal);
    else result = (aVal as number) - (bVal as number);
    return sortAsc ? result : -result;
  });

  if (loading) {
    return (
      <div className="space-y-4">
        <div className="h-8 w-48 skeleton" />
        <div className="h-10 w-72 skeleton" />
        <div className="h-64 skeleton" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: 'var(--foreground)' }}>Patients</h1>
          <p style={{ color: 'var(--text-muted)' }}>Manage your patients</p>
        </div>
        <motion.button whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
          onClick={() => setShowAddModal(true)} className="premium-btn premium-btn-primary">
          <UserPlus className="w-5 h-5" /> Add Patient
        </motion.button>
      </div>

      <div className="flex flex-wrap items-center gap-4">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4" style={{ color: 'var(--text-dim)' }} />
          <input type="text" placeholder="Search patients..." value={search} onChange={(e) => setSearch(e.target.value)} className="premium-input pl-11" />
        </div>
        <div className="flex items-center gap-2 text-sm" style={{ color: 'var(--text-muted)' }}>
          <span>Sort:</span>
          <select value={sortField} onChange={(e) => setSortField(e.target.value)} className="premium-select py-2 w-auto">
            <option value="lastVisit">Last Visit</option><option value="createdAt">Date Added</option>
            <option value="name">Name</option><option value="age">Age</option><option value="risk">Risk</option>
          </select>
        </div>
        <label className="flex items-center gap-2 text-sm cursor-pointer" style={{ color: 'var(--text-muted)' }}>
          <input type="checkbox" checked={sortAsc} onChange={(e) => setSortAsc(e.target.checked)} className="w-4 h-4 rounded" style={{ accentColor: 'var(--primary)' }} />
          Ascending
        </label>
        <div className="flex items-center gap-2 text-sm" style={{ color: 'var(--text-muted)' }}>
          <span>Visit:</span>
          <select value={visitFilter} onChange={(e) => setVisitFilter(e.target.value)} className="premium-select py-2 w-auto">
            <option value="all">All time</option><option value="7days">7 days</option>
            <option value="30days">30 days</option><option value="90days">3 months</option><option value="year">Year</option>
          </select>
        </div>
      </div>

      <div className="premium-card overflow-hidden">
        <div className="overflow-x-auto">
          {sortedPatients.length === 0 ? (
            <div className="py-16 text-center" style={{ color: 'var(--text-muted)' }}>
              {patients.length === 0 ? 'No patients yet. Add your first patient.' : 'No patients match your search.'}
            </div>
          ) : (
            <table className="w-full">
              <thead>
                <tr style={{ backgroundColor: 'var(--background-alt)' }}>
                  {['Name', 'Age', 'Parent', 'Contact', 'Last Visit', 'Risk', ''].map(h => (
                    <th key={h} className="px-6 py-4 text-left text-sm font-semibold" style={{ color: 'var(--foreground)' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y" style={{ borderColor: 'var(--border-light)' }}>
                {sortedPatients.map((patient) => (
                  <motion.tr key={patient.id} initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                    className="transition-colors hover:bg-[var(--background-alt)]">
                    <td className="px-6 py-4">
                      <Link href={`/professional/patients/${patient.id}`}>
                        <div className="flex items-center gap-3 cursor-pointer group">
                          <div className="w-10 h-10 rounded-full flex items-center justify-center text-white font-semibold text-sm" style={{ background: 'var(--gradient-primary)' }}>
                            {patient.name.charAt(0)}
                          </div>
                          <span className="font-medium group-hover:opacity-70 transition-opacity" style={{ color: 'var(--foreground)' }}>{patient.name}</span>
                        </div>
                      </Link>
                    </td>
                    <td className="px-6 py-4" style={{ color: 'var(--text-secondary)' }}>{getAge(patient)}</td>
                    <td className="px-6 py-4" style={{ color: 'var(--text-secondary)' }}>{patient.parentName}</td>
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-1">
                        <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={() => handlePhoneClick(patient.phone)} className="p-2 rounded-lg text-white" style={{ backgroundColor: 'var(--risk-low)' }} title={patient.phone || 'N/A'}>
                          <Phone className="w-4 h-4" />
                        </motion.button>
                        <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={() => handleMailClick(patient.email)} className="p-2 rounded-lg text-white" style={{ backgroundColor: 'var(--primary)' }}>
                          <Mail className="w-4 h-4" />
                        </motion.button>
                      </div>
                    </td>
                    <td className="px-6 py-4" style={{ color: 'var(--text-secondary)' }}>{patient.lastVisit}</td>
                    <td className="px-6 py-4">
                      <span className={`premium-badge ${riskColors[patient.risk] || ''}`}>{patient.risk}</span>
                    </td>
                    <td className="px-6 py-4">
                      <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
                        onClick={() => { setPatientToDelete(patient); setShowDeleteConfirm(true); }}
                        className="p-2 rounded-lg transition-colors" style={{ backgroundColor: 'var(--risk-high-bg)', color: 'var(--risk-high)' }}>
                        <Trash2 className="w-4 h-4" />
                      </motion.button>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {/* Add Patient Modal */}
      {showAddModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm p-4" onClick={() => setShake(true)}>
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }} animate={shake ? { x: [0, -10, 10, -10, 10, 0] } : { opacity: 1, scale: 1 }}
            onAnimationComplete={() => setShake(false)} onClick={(e) => e.stopPropagation()}
            className="w-full max-w-md max-h-[90vh] premium-card flex flex-col relative overflow-hidden">
            {showCancelConfirm && (
              <div className="absolute inset-0 z-50 flex items-center justify-center p-6" style={{ backgroundColor: 'var(--surface)' }}>
                <div className="text-center">
                  <AlertTriangle className="w-14 h-14 mx-auto mb-4" style={{ color: 'var(--risk-moderate)' }} />
                  <p className="text-lg font-medium mb-2" style={{ color: 'var(--foreground)' }}>Discard changes?</p>
                  <p className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>Your data will be lost.</p>
                  <div className="flex gap-3 justify-center">
                    <button onClick={closeModal} className="premium-btn" style={{ backgroundColor: 'var(--risk-high)', color: 'white' }}>Yes, discard</button>
                    <button onClick={() => setShowCancelConfirm(false)} className="premium-btn premium-btn-ghost">No, keep</button>
                  </div>
                </div>
              </div>
            )}
            {saved && (
              <div className="absolute inset-0 z-50 flex items-center justify-center" style={{ background: 'var(--gradient-primary)' }}>
                <div className="text-center p-6">
                  <div className="w-20 h-20 mx-auto mb-4 rounded-full flex items-center justify-center" style={{ backgroundColor: 'rgba(255,255,255,0.2)' }}>
                    <CheckCircle className="w-10 h-10 text-white" />
                  </div>
                  <p className="text-2xl font-bold text-white mb-2">Saved!</p>
                  {savedMessage && <p className="text-white/80 text-sm max-w-xs mx-auto">{savedMessage}</p>}
                </div>
              </div>
            )}
            <div className="flex items-center justify-between p-6 border-b shrink-0" style={{ borderColor: 'var(--border-light)' }}>
              <h2 className="text-xl font-semibold" style={{ color: 'var(--foreground)' }}>Add Patient</h2>
              <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={handleCancelClick}
                className="p-2 rounded-lg" style={{ backgroundColor: 'var(--background-alt)' }}>
                <X className="w-5 h-5" />
              </motion.button>
            </div>
            <div className="p-6 space-y-4 overflow-y-auto">
              {formError && <div className="p-3 rounded-xl text-sm" style={{ backgroundColor: 'var(--risk-high-bg)', color: 'var(--risk-high)' }}>{formError}</div>}
              {[{ label: "Child's Name", placeholder: 'e.g. Tala Al Zoubi', key: 'name', type: 'text', required: true },
                { label: 'Date of Birth', placeholder: '', key: 'dob', type: 'date', required: true },
                { label: 'Parent / Guardian Name', placeholder: 'e.g. John Thompson', key: 'parentName', type: 'text', required: true },
                { label: 'Email', placeholder: 'e.g. john@email.com', key: 'email', type: 'email', required: true },
                { label: 'Phone (optional)', placeholder: 'e.g. +1 555-0123', key: 'phone', type: 'tel', required: false },
              ].map(f => (
                <div key={f.key}>
                  <label className="block text-sm font-medium mb-1.5" style={{ color: 'var(--foreground)' }}>
                    {f.label}
                    {f.required && <span className="ml-1 text-red-500">*</span>}
                  </label>
                  <input type={f.type} placeholder={f.placeholder} value={(formData as any)[f.key]}
                    onChange={(e) => setFormData(prev => ({ ...prev, [f.key]: e.target.value }))} className="premium-input" />
                </div>
              ))}
              <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}><span className="text-red-500">*</span> Required fields</p>
              <div className="pt-2 border-t" style={{ borderColor: 'var(--border-light)' }}>
                <label className="flex items-center gap-3 cursor-pointer">
                  <input type="checkbox" checked={sendCredentials} onChange={(e) => setSendCredentials(e.target.checked)}
                    className="w-5 h-5 rounded" style={{ accentColor: 'var(--primary)' }} />
                  <span className="text-sm font-medium" style={{ color: 'var(--foreground)' }}>
                    <Send className="w-4 h-4 inline mr-1.5" style={{ color: 'var(--primary)' }} />
                    Send account details to parent
                  </span>
                </label>
              </div>
            </div>
            <div className="flex gap-3 p-6 border-t shrink-0" style={{ borderColor: 'var(--border-light)' }}>
              <button onClick={handleCancelClick} className="premium-btn premium-btn-ghost flex-1">Cancel</button>
              <motion.button whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
                onClick={handleAddPatient} disabled={saving || saved || !formData.name || !formData.dob || !formData.parentName || !formData.email}
                className="premium-btn premium-btn-primary flex-1 disabled:opacity-50 disabled:cursor-not-allowed">
                {saving ? <><span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" /> Saving...</>
                  : saved ? <><CheckCircle className="w-4 h-4" /> Saved!</>
                  : <><Send className="w-4 h-4" /> {sendCredentials ? 'Save & Send' : 'Save Patient'}</>}
              </motion.button>
            </div>
          </motion.div>
        </div>
      )}

      {showDeleteConfirm && patientToDelete && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm p-4">
          <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}
            className="w-full max-w-sm premium-card p-6 text-center">
            <div className="w-16 h-16 mx-auto mb-4 rounded-full flex items-center justify-center" style={{ backgroundColor: 'var(--risk-high-bg)' }}>
              <AlertTriangle className="w-8 h-8" style={{ color: 'var(--risk-high)' }} />
            </div>
            <h2 className="text-xl font-bold mb-2" style={{ color: 'var(--foreground)' }}>Delete Patient</h2>
            <p className="mb-6 text-sm" style={{ color: 'var(--text-muted)' }}>
              Delete <strong>{patientToDelete.name}</strong>? This cannot be undone.
            </p>
            <div className="flex gap-3">
              <button onClick={async () => {
                const user = getCurrentUser();
                if (user) {
                  try {
                    await deletePatient(user.id, patientToDelete.id);
                    setPatients(prev => prev.filter(p => p.id !== patientToDelete.id));
                    showToast('success', 'Patient Removed', `${patientToDelete.name} has been removed.`);
                  } catch (err) { console.error(err); }
                }
                setShowDeleteConfirm(false); setPatientToDelete(null);
              }} className="premium-btn flex-1" style={{ backgroundColor: 'var(--risk-high)', color: 'white' }}>
                <Trash2 className="w-4 h-4" /> Delete
              </button>
              <button onClick={() => { setShowDeleteConfirm(false); setPatientToDelete(null); }} className="premium-btn premium-btn-ghost flex-1">Cancel</button>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  );
}
