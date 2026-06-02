'use client';

import { useState, useEffect } from 'react';
import { Link2, Mail, CheckCircle, XCircle, RefreshCw, Users, Search, X } from 'lucide-react';
import { motion } from 'framer-motion';
import { getPatientLinksByProfessional } from '@/lib/patient-access';
import { getPatients } from '@/lib/patients';
import { getCurrentUser } from '@/lib/auth';
import { resendParentCredentials, deactivateParentAccount, reactivateParentAccount } from '@/lib/parent-accounts';
import { PatientAccessLink } from '@/lib/patient-access';
import { Patient } from '@/lib/patients';

function formatDate(date: unknown): string {
  if (!date) return 'N/A';
  if (date instanceof Date) return date.toLocaleDateString();
  if (typeof date === 'object' && date && 'seconds' in date) {
    return new Date((date as { seconds: number }).seconds * 1000).toLocaleDateString();
  }
  return 'N/A';
}

export default function PatientAccessPage() {
  const [links, setLinks] = useState<PatientAccessLink[]>([]);
  const [patients, setPatients] = useState<Patient[]>([]);
  const [loading, setLoading] = useState(true);
  const [mounted, setMounted] = useState(false);
  const [search, setSearch] = useState('');
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [toast, setToast] = useState<{ message: string; type: 'success' | 'error' } | null>(null);
  const [showRevokeModal, setShowRevokeModal] = useState<PatientAccessLink | null>(null);

  useEffect(() => {
    setMounted(true);
    loadData();
  }, []);

  const loadData = async () => {
    const user = getCurrentUser();
    if (!user) {
      setLoading(false);
      return;
    }

    try {
      const [linksData, patientsData] = await Promise.all([
        getPatientLinksByProfessional(user.id),
        getPatients(user.id),
      ]);
      setLinks(linksData);
      setPatients(patientsData);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const showToast = (message: string, type: 'success' | 'error') => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 3000);
  };

  const getPatientName = (patientId: string): string => {
    const patient = patients.find(p => p.id === patientId);
    return patient?.name || 'Unknown Patient';
  };

  const handleResendCredentials = async (link: PatientAccessLink) => {
    setActionLoading(link.id);
    try {
      const result = await resendParentCredentials(link.parentId, link.patientName);
      if (result.success) {
        showToast(`Credentials sent to ${link.parentEmail}`, 'success');
      } else {
        showToast(result.error || 'Failed to send credentials', 'error');
      }
    } catch {
      showToast('Failed to send credentials', 'error');
    } finally {
      setActionLoading(null);
    }
  };

  const handleToggleAccess = async (link: PatientAccessLink, action: 'revoke' | 'grant') => {
    setActionLoading(link.id);
    try {
      let result;
      if (action === 'revoke') {
        result = await deactivateParentAccount(link.parentId);
      } else {
        result = await reactivateParentAccount(link.parentId);
      }

      if (result.success) {
        await loadData();
        showToast(
          action === 'revoke' ? `Access revoked for ${link.parentEmail}` : `Access restored for ${link.parentEmail}`,
          'success'
        );
      } else {
        showToast(result.error || `Failed to ${action} access`, 'error');
      }
    } catch {
      showToast('Failed to update access', 'error');
    } finally {
      setActionLoading(null);
      setShowRevokeModal(null);
    }
  };

  const filteredLinks = mounted ? links.filter(link => {
    const patientName = getPatientName(link.patientId).toLowerCase();
    const parentName = link.parentName.toLowerCase();
    const parentEmail = link.parentEmail.toLowerCase();
    const searchLower = search.toLowerCase();
    return patientName.includes(searchLower) || parentName.includes(searchLower) || parentEmail.includes(searchLower);
  }) : [];

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="w-8 h-8 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {toast && (
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
          className={`fixed top-4 right-4 z-50 px-4 py-3 rounded-xl flex items-center gap-2 shadow-lg ${
            toast.type === 'success' ? 'bg-green-500 text-white' : 'bg-red-500 text-white'
          }`}
        >
          {toast.type === 'success' ? <CheckCircle className="w-5 h-5" /> : <XCircle className="w-5 h-5" />}
          {toast.message}
        </motion.div>
      )}

      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Patient Access</h1>
          <p className="text-gray-500 dark:text-gray-400">Manage parent access to patient data</p>
        </div>
      </div>

      <div className="relative max-w-md">
        <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
        <input
          type="text"
          placeholder="Search by patient, parent name, or email..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full pl-12 pr-4 py-3 bg-white dark:bg-dark-surface border border-gray-200 dark:border-dark-deep rounded-xl focus:outline-none focus:ring-2 focus:ring-primary"
        />
      </div>

      <div className="bg-white dark:bg-dark-surface rounded-2xl border border-gray-200 dark:border-dark-deep overflow-hidden">
        <div className="overflow-x-auto">
          {filteredLinks.length === 0 ? (
            <div className="py-16 text-center">
              <Users className="w-12 h-12 mx-auto text-gray-400 mb-4" />
              <p className="text-gray-500 dark:text-gray-400">
                {links.length === 0
                  ? 'No parent access links yet. Add a patient with parent email to get started.'
                  : 'No access links match your search.'}
              </p>
            </div>
          ) : (
            <table className="w-full">
              <thead className="bg-gray-50 dark:bg-dark-deep">
                <tr>
                  <th className="px-6 py-4 text-left text-sm font-semibold text-gray-900 dark:text-white">Patient</th>
                  <th className="px-6 py-4 text-left text-sm font-semibold text-gray-900 dark:text-white">Parent</th>
                  <th className="px-6 py-4 text-left text-sm font-semibold text-gray-900 dark:text-white">Status</th>
                  <th className="px-6 py-4 text-left text-sm font-semibold text-gray-900 dark:text-white">Created</th>
                  <th className="px-6 py-4 text-left text-sm font-semibold text-gray-900 dark:text-white">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 dark:divide-dark-deep">
                {filteredLinks.map((link) => (
                  <motion.tr
                    key={link.id}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="hover:bg-gray-50 dark:hover:bg-dark-deep transition-colors"
                  >
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-full bg-primary-light dark:bg-primary-dark/40 flex items-center justify-center text-primary dark:text-primary-light font-semibold">
                          {getPatientName(link.patientId).charAt(0)}
                        </div>
                        <span className="font-medium text-gray-900 dark:text-white">
                          {getPatientName(link.patientId)}
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <div className="text-gray-900 dark:text-white">{link.parentName}</div>
                      <div className="text-sm text-gray-500 dark:text-gray-400">{link.parentEmail}</div>
                    </td>
                    <td className="px-6 py-4">
                      <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                        link.accessGranted
                          ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                          : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                      }`}>
                        {link.accessGranted ? 'Active' : 'Revoked'}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-gray-600 dark:text-gray-300">
                      {formatDate(link.createdAt)}
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => handleResendCredentials(link)}
                          disabled={actionLoading === link.id}
                          className="p-2 bg-primary/10 text-primary dark:text-primary-light hover:bg-primary/20 rounded-lg disabled:opacity-50 transition-colors"
                          title="Resend credentials"
                        >
                          <RefreshCw className={`w-4 h-4 ${actionLoading === link.id ? 'animate-spin' : ''}`} />
                        </button>
                        <button
                          onClick={() => window.location.href = `mailto:${link.parentEmail}`}
                          className="p-2 bg-blue-800 text-white hover:bg-blue-900 rounded-lg transition-colors"
                          title="Email parent"
                        >
                          <Mail className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => setShowRevokeModal(link)}
                          className={`p-2 rounded-lg disabled:opacity-50 transition-colors ${
                            link.accessGranted
                              ? 'bg-red-100 text-red-600 hover:bg-red-200 dark:bg-red-900/30 dark:text-red-400 dark:hover:bg-red-900/50'
                              : 'bg-green-100 text-green-600 hover:bg-green-200 dark:bg-green-900/30 dark:text-green-400 dark:hover:bg-green-900/50'
                          }`}
                          title={link.accessGranted ? 'Revoke access' : 'Restore access'}
                        >
                          {link.accessGranted ? <XCircle className="w-4 h-4" /> : <CheckCircle className="w-4 h-4" />}
                        </button>
                      </div>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {showRevokeModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="w-full max-w-md bg-white dark:bg-dark-surface rounded-2xl shadow-xl p-6"
          >
            <div className="text-center">
              <div className={`w-16 h-16 mx-auto mb-4 rounded-full flex items-center justify-center ${
                showRevokeModal.accessGranted
                  ? 'bg-red-100 dark:bg-red-900/30'
                  : 'bg-green-100 dark:bg-green-900/30'
              }`}>
                {showRevokeModal.accessGranted ? (
                  <XCircle className="w-8 h-8 text-red-500" />
                ) : (
                  <CheckCircle className="w-8 h-8 text-green-500" />
                )}
              </div>
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                {showRevokeModal.accessGranted ? 'Revoke Access?' : 'Restore Access?'}
              </h3>
              <p className="text-gray-500 dark:text-gray-400 mb-6">
                {showRevokeModal.accessGranted ? (
                  <>This will prevent {showRevokeModal.parentEmail} from accessing {getPatientName(showRevokeModal.patientId)}&apos;s data.</>
                ) : (
                  <>This will restore {showRevokeModal.parentEmail}&apos;s access to {getPatientName(showRevokeModal.patientId)}&apos;s data.</>
                )}
              </p>
              <div className="flex gap-3 justify-center">
                <button
                  onClick={() => setShowRevokeModal(null)}
                  className="px-6 py-2.5 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300 font-medium rounded-xl"
                >
                  Cancel
                </button>
                <button
                  onClick={() => handleToggleAccess(showRevokeModal, showRevokeModal.accessGranted ? 'revoke' : 'grant')}
                  className={`px-6 py-2.5 font-medium rounded-xl ${
                    showRevokeModal.accessGranted
                      ? 'bg-red-500 hover:bg-red-600 text-white'
                      : 'bg-green-500 hover:bg-green-600 text-white'
                  }`}
                >
                  {showRevokeModal.accessGranted ? 'Revoke Access' : 'Restore Access'}
                </button>
              </div>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  );
}
