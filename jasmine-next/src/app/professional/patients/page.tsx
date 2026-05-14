'use client';

import { UserPlus, Search, MoreVertical, Phone, Mail, X, Send, CheckCircle } from 'lucide-react';
import { useState, useEffect } from 'react';
import { motion, useAnimation } from 'framer-motion';
import { getPatients, addPatient, Patient } from '@/lib/patients';
import { getCurrentUser } from '@/lib/auth';
import { createPatientAccess } from '@/lib/patient-access';
import { sendParentRequest } from '@/lib/parent-requests';
import { addNotification } from '@/lib/notifications';

function calculateAge(dob: string): number {
  const birth = new Date(dob);
  const today = new Date();
  let age = today.getFullYear() - birth.getFullYear();
  const m = today.getMonth() - birth.getMonth();
  if (m < 0 || (m === 0 && today.getDate() < birth.getDate())) {
    age--;
  }
  return age;
}

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

  useEffect(() => {
    setMounted(true);
    const user = getCurrentUser();
    if (user) {
      getPatients(user.id)
        .then(setPatients)
        .catch(console.error)
        .finally(() => setLoading(false));
    } else {
      setLoading(false);
    }
  }, []);

  const handleAddPatient = async () => {
    if (!formData.name || !formData.dob || !formData.parentName || !formData.email) {
      setFormError('All fields except phone are required.');
      return;
    }
    if (!formData.dob.match(/^\d{4}-\d{2}-\d{2}$/)) {
      setFormError('Please enter a valid date (YYYY-MM-DD).');
      return;
    }
    if (new Date(formData.dob) > new Date()) {
      setFormError('Date of birth cannot be in the future.');
      return;
    }
    if (formData.phone && !/^[+]?[\d\s\-()]*$/.test(formData.phone)) {
      setFormError('Please enter a valid phone number');
      return;
    }

    const user = getCurrentUser();
    if (!user) return;

    setSaving(true);
    try {
      const newPatient = await addPatient(user.id, {
        name: formData.name,
        dob: formData.dob,
        parentName: formData.parentName,
        email: formData.email,
        phone: formData.phone || '',
        lastVisit: new Date().toISOString().split('T')[0],
        risk: 'Unknown',
      });
      setPatients(prev => [newPatient, ...prev]);

      // Send parent request (friend request to parent email)
      try {
        await sendParentRequest({
          professionalId: user.id,
          professionalName: user.name,
          patientId: newPatient.id,
          patientName: newPatient.name,
          parentEmail: formData.email,
          parentName: formData.parentName,
        });
      } catch (err) {
        console.warn('Failed to send parent request:', err);
      }

      // Notification for professional
      try {
        await addNotification({
          userId: user.id,
          type: 'patient_added',
          title: 'Patient Added',
          message: `${newPatient.name} has been added successfully.`,
          link: '/professional/patients',
        });
      } catch (err) {
        console.warn('Failed to create notification:', err);
      }

      if (sendCredentials) {
        const accessResult = await createPatientAccess({
          patientId: newPatient.id,
          patientName: newPatient.name,
          professionalId: user.id,
          parentName: formData.parentName,
          parentEmail: formData.email,
        });

        if (accessResult.success && accessResult.parentTempPassword) {
          setSavedMessage('Patient added! Account credentials sent to parent.');
        } else if (accessResult.success) {
          setSavedMessage('Patient added! Parent already has access.');
        } else {
          setSavedMessage('Patient added, but failed to send credentials.');
        }
      } else {
        setSavedMessage('Patient added successfully.');
      }

      setSaved(true);
      setFormData({ name: '', dob: '', parentName: '', email: '', phone: '' });
      setFormError('');
      setTimeout(() => {
        setShowAddModal(false);
        setSaved(false);
        setSavedMessage('');
      }, 2000);
    } catch (err) {
      setFormError('Failed to add patient. Please try again.');
      console.error(err);
    } finally {
      setSaving(false);
    }
  };

  const hasFormData = formData.name || formData.dob || formData.parentName || formData.email || formData.phone;

  const closeModal = () => {
    setShowAddModal(false);
    setFormError('');
    setFormData({ name: '', dob: '', parentName: '', email: '', phone: '' });
    setShowCancelConfirm(false);
    setSaved(false);
    setSavedMessage('');
  };

  const handleCancelClick = () => {
    if (hasFormData) {
      setShowCancelConfirm(true);
    } else {
      closeModal();
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

  const getAge = (patient: Patient): number => {
    if (patient.dob) return calculateAge(patient.dob);
    return 0;
  };

  const displayedPatients = mounted ? patients : [];
  
  const filteredPatients = displayedPatients.filter(p => {
    const matchesSearch = p.name.toLowerCase().includes(search.toLowerCase()) ||
      p.parentName?.toLowerCase().includes(search.toLowerCase());
    
    if (visitFilter === 'all') return matchesSearch;
    
    const lastVisit = p.lastVisit ? new Date(p.lastVisit) : null;
    if (!lastVisit) return matchesSearch;
    
    const now = new Date();
    const daysDiff = Math.floor((now.getTime() - lastVisit.getTime()) / (1000 * 60 * 60 * 24));
    
    switch (visitFilter) {
      case '7days':
        return matchesSearch && daysDiff <= 7;
      case '30days':
        return matchesSearch && daysDiff <= 30;
      case '90days':
        return matchesSearch && daysDiff <= 90;
      case 'year':
        return matchesSearch && daysDiff <= 365;
      default:
        return matchesSearch;
    }
  });

  const riskOrder: Record<string, number> = { 'High': 3, 'Moderate': 2, 'Low': 1, 'Unknown': 0 };
  
  const getCreatedAt = (patient: Patient): number => {
    if (!patient.createdAt) return 0;
    if (typeof patient.createdAt === 'number') return patient.createdAt;
    if (typeof patient.createdAt === 'object' && patient.createdAt && 'seconds' in patient.createdAt) {
      return (patient.createdAt as { seconds: number }).seconds;
    }
    return 0;
  };
  
  const getSortValue = (patient: Patient): number | string => {
    switch (sortField) {
      case 'name':
        return patient.name.toLowerCase();
      case 'age':
        return getAge(patient);
      case 'createdAt':
        return getCreatedAt(patient);
      case 'lastVisit':
        return patient.lastVisit || '';
      case 'risk':
        return riskOrder[patient.risk || 'Unknown'];
      default:
        return getCreatedAt(patient);
    }
  };
  
  const sortedPatients = [...filteredPatients].sort((a, b) => {
    const aVal = getSortValue(a);
    const bVal = getSortValue(b);
    
    let result = 0;
    if (typeof aVal === 'string' && typeof bVal === 'string') {
      result = aVal.localeCompare(bVal);
    } else {
      result = (aVal as number) - (bVal as number);
    }
    
    return sortAsc ? result : -result;
  });

  const riskColors: Record<string, string> = {
    High: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400',
    Moderate: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400',
    Low: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
    Unknown: 'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400',
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="w-8 h-8 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Patients</h1>
          <p className="text-gray-500 dark:text-gray-400">Manage your patients</p>
        </div>
        <button
          onClick={() => setShowAddModal(true)}
          className="inline-flex items-center gap-2 px-4 py-2.5 bg-primary hover:bg-primary-dark text-white font-medium rounded-xl transition-all"
        >
          <UserPlus className="w-5 h-5" />
          Add Patient
        </button>
      </div>

      {/* Search */}
      <div className="relative max-w-md">
        <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
        <input
          type="text"
          placeholder="Search patients..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full pl-12 pr-4 py-3 bg-white dark:bg-dark-surface border border-gray-200 dark:border-dark-deep rounded-xl focus:outline-none focus:ring-2 focus:ring-primary"
        />
      </div>

      {/* Sort and Filter */}
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-500 dark:text-gray-400">Sort by:</span>
          <select
            value={sortField}
            onChange={(e) => setSortField(e.target.value)}
            className="px-3 py-2 bg-white dark:bg-dark-surface border border-gray-200 dark:border-dark-deep rounded-lg text-sm text-gray-700 dark:text-gray-300 focus:outline-none focus:ring-2 focus:ring-primary"
          >
            <option value="lastVisit">Last Visit</option>
            <option value="createdAt">Date Added</option>
            <option value="name">Name</option>
            <option value="age">Age</option>
            <option value="risk">Risk</option>
          </select>
        </div>
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={sortAsc}
            onChange={(e) => setSortAsc(e.target.checked)}
            className="w-4 h-4 rounded border-gray-300 text-primary focus:ring-primary"
          />
          <span className="text-sm text-gray-500 dark:text-gray-400">Ascending</span>
        </label>
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-500 dark:text-gray-400">Last visit:</span>
          <select
            value={visitFilter}
            onChange={(e) => setVisitFilter(e.target.value)}
            className="px-3 py-2 bg-white dark:bg-dark-surface border border-gray-200 dark:border-dark-deep rounded-lg text-sm text-gray-700 dark:text-gray-300 focus:outline-none focus:ring-2 focus:ring-primary"
          >
            <option value="all">All time</option>
            <option value="7days">Last 7 days</option>
            <option value="30days">Last 30 days</option>
            <option value="90days">Last 3 months</option>
            <option value="year">Last year</option>
          </select>
        </div>
      </div>

      {/* Patients Table */}
      <div className="bg-white dark:bg-dark-surface rounded-2xl border border-gray-200 dark:border-dark-deep overflow-hidden">
        <div className="overflow-x-auto">
          {sortedPatients.length === 0 ? (
            <div className="py-16 text-center">
              <p className="text-gray-500 dark:text-gray-400">
                {patients.length === 0 
                  ? 'No patients yet. Add your first patient to get started.' 
                  : 'No patients match your search.'}
              </p>
            </div>
          ) : (
            <table className="w-full">
              <thead className="bg-gray-50 dark:bg-dark-deep">
                <tr>
                  <th className="px-6 py-4 text-left text-sm font-semibold text-gray-900 dark:text-white">Name</th>
                  <th className="px-6 py-4 text-left text-sm font-semibold text-gray-900 dark:text-white">Age</th>
                  <th className="px-6 py-4 text-left text-sm font-semibold text-gray-900 dark:text-white">Parent</th>
                  <th className="px-6 py-4 text-left text-sm font-semibold text-gray-900 dark:text-white">Contact</th>
                  <th className="px-6 py-4 text-left text-sm font-semibold text-gray-900 dark:text-white">Last Visit</th>
                  <th className="px-6 py-4 text-left text-sm font-semibold text-gray-900 dark:text-white">Risk</th>
                  <th className="px-6 py-4"></th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 dark:divide-dark-deep">
                {sortedPatients.map((patient) => (
                  <motion.tr
                    key={patient.id}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="hover:bg-gray-50 dark:hover:bg-dark-deep transition-colors"
                  >
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-full bg-primary-light flex items-center justify-center text-primary font-semibold">
                          {patient.name.charAt(0)}
                        </div>
                        <span className="font-medium text-gray-900 dark:text-white">{patient.name}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-gray-600 dark:text-gray-300">{getAge(patient)}</td>
                    <td className="px-6 py-4 text-gray-600 dark:text-gray-300">{patient.parentName}</td>
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => handlePhoneClick(patient.phone)}
                          className="p-2 bg-green-700 text-white hover:bg-green-800 rounded-lg"
                          title={patient.phone || 'Phone number not provided'}
                        >
                          <Phone className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => handleMailClick(patient.email)}
                          className="p-2 bg-blue-800 text-white hover:bg-blue-900 rounded-lg"
                        >
                          <Mail className="w-4 h-4" />
                        </button>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-gray-600 dark:text-gray-300">{patient.lastVisit}</td>
                    <td className="px-6 py-4">
                      <span className={`px-3 py-1 rounded-full text-sm font-medium ${riskColors[patient.risk] || ''}`}>
                        {patient.risk}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <button className="p-2 bg-primary/10 text-primary hover:bg-primary/20 rounded-lg transition-colors">
                        <MoreVertical className="w-4 h-4" />
                      </button>
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
        <div 
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4"
          onClick={() => setShake(true)}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={shake ? { x: [0, -10, 10, -10, 10, 0] } : { opacity: 1, scale: 1 }}
            onAnimationComplete={() => setShake(false)}
            onClick={(e) => e.stopPropagation()}
            className="w-full max-w-md max-h-[90vh] bg-white dark:bg-dark-surface rounded-2xl shadow-xl flex flex-col relative"
          >
            {/* Cancel Confirmation Overlay */}
            {showCancelConfirm && (
              <div className="absolute inset-0 z-50 bg-white/95 dark:bg-dark-surface/95 rounded-2xl flex items-center justify-center p-6">
                <div className="text-center">
                  <div className="w-16 h-16 mx-auto mb-4 bg-yellow-100 dark:bg-yellow-900/30 rounded-full flex items-center justify-center">
                    <span className="text-3xl">⚠️</span>
                  </div>
                  <p className="text-lg font-medium text-gray-900 dark:text-white mb-2">Are you sure?</p>
                  <p className="text-sm text-gray-500 dark:text-gray-400 mb-6">Your data will be lost.</p>
                  <div className="flex gap-3 justify-center">
                    <button
                      onClick={closeModal}
                      className="px-6 py-2.5 bg-red-500 hover:bg-red-600 text-white font-medium rounded-xl"
                    >
                      Yes, discard
                    </button>
                    <button
                      onClick={() => setShowCancelConfirm(false)}
                      className="px-6 py-2.5 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300 font-medium rounded-xl"
                    >
                      No, keep
                    </button>
                  </div>
                </div>
              </div>
            )}

              {/* Saved Overlay */}
            {saved && (
              <div className="absolute inset-0 z-50 bg-green-500 rounded-2xl flex items-center justify-center">
                <div className="text-center p-6">
                  <div className="w-24 h-24 mx-auto mb-4 bg-white rounded-full flex items-center justify-center shadow-lg">
                    <CheckCircle className="w-12 h-12 text-green-500" />
                  </div>
                  <p className="text-2xl font-bold text-white drop-shadow-md mb-2">Saved!</p>
                  {savedMessage && (
                    <p className="text-white/90 text-sm max-w-xs mx-auto">{savedMessage}</p>
                  )}
                </div>
              </div>
            )}

            <div className="flex items-center justify-between p-6 border-b border-gray-100 dark:border-dark-deep shrink-0">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white">Add Patient</h2>
              <button
                onClick={handleCancelClick}
                className="p-2 bg-primary text-white hover:bg-primary-dark rounded-lg transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="p-6 space-y-4 overflow-y-auto">
              {formError && (
                <div className="p-3 rounded-xl bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 text-sm">
                  {formError}
                </div>
              )}

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">Child's Name</label>
                <input
                  type="text"
                  placeholder="e.g. Emma Thompson"
                  value={formData.name}
                  onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
                  className="w-full px-4 py-2.5 bg-gray-50 dark:bg-dark-deep border border-gray-200 dark:border-dark-deep rounded-xl focus:outline-none focus:ring-2 focus:ring-primary"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">Date of Birth</label>
                <input
                  type="date"
                  value={formData.dob}
                  onChange={(e) => setFormData(prev => ({ ...prev, dob: e.target.value }))}
                  className="w-full px-4 py-2.5 bg-gray-50 dark:bg-dark-deep border border-gray-200 dark:border-dark-deep rounded-xl focus:outline-none focus:ring-2 focus:ring-primary"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">Parent / Guardian Name</label>
                <input
                  type="text"
                  placeholder="e.g. John Thompson"
                  value={formData.parentName}
                  onChange={(e) => setFormData(prev => ({ ...prev, parentName: e.target.value }))}
                  className="w-full px-4 py-2.5 bg-gray-50 dark:bg-dark-deep border border-gray-200 dark:border-dark-deep rounded-xl focus:outline-none focus:ring-2 focus:ring-primary"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">Email</label>
                <input
                  type="email"
                  placeholder="e.g. john@email.com"
                  value={formData.email}
                  onChange={(e) => setFormData(prev => ({ ...prev, email: e.target.value }))}
                  className="w-full px-4 py-2.5 bg-gray-50 dark:bg-dark-deep border border-gray-200 dark:border-dark-deep rounded-xl focus:outline-none focus:ring-2 focus:ring-primary"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">
                  Phone <span className="text-gray-400 font-normal">(optional)</span>
                </label>
                <input
                  type="tel"
                  placeholder="e.g. +1 555-0123"
                  value={formData.phone}
                  onChange={(e) => setFormData(prev => ({ ...prev, phone: e.target.value }))}
                  className="w-full px-4 py-2.5 bg-gray-50 dark:bg-dark-deep border border-gray-200 dark:border-dark-deep rounded-xl focus:outline-none focus:ring-2 focus:ring-primary"
                />
              </div>

              <div className="pt-2 border-t border-gray-100 dark:border-dark-deep">
                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={sendCredentials}
                    onChange={(e) => setSendCredentials(e.target.checked)}
                    className="w-5 h-5 rounded border-gray-300 text-primary focus:ring-primary"
                  />
                  <div className="flex items-center gap-2">
                    <Send className="w-4 h-4 text-primary" />
                    <span className="text-sm text-gray-700 dark:text-gray-300">
                      Send account details to parent
                    </span>
                  </div>
                </label>
                <p className="mt-1 text-xs text-gray-500 dark:text-gray-400 ml-8">
                  An email with login credentials will be sent to the parent.
                </p>
              </div>
            </div>

            <div className="flex gap-3 p-6 border-t border-gray-100 dark:border-dark-deep shrink-0">
              <button
                onClick={handleCancelClick}
                className="flex-1 px-4 py-2.5 bg-primary hover:bg-primary-dark text-white font-medium rounded-xl transition-all flex items-center justify-center gap-2"
              >
                Cancel
              </button>
              <button
                onClick={handleAddPatient}
                disabled={saving || saved}
                className="flex-1 px-4 py-2.5 bg-primary hover:bg-primary-dark text-white font-medium rounded-xl transition-all flex items-center justify-center gap-2 disabled:opacity-50"
              >
                {saving ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Saving...
                  </>
                ) : saved ? (
                  <>
                    <CheckCircle className="w-4 h-4" />
                    Saved!
                  </>
                ) : (
                  <>
                    {sendCredentials ? <Send className="w-4 h-4" /> : <UserPlus className="w-4 h-4" />}
                    {sendCredentials ? 'Save & Send Credentials' : 'Save Patient'}
                  </>
                )}
              </button>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  );
}