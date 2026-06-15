'use client';

import { useState, useEffect } from 'react';
import { UserPlus, CheckCircle, XCircle, Clock, Mail, Send, X } from 'lucide-react';
import { motion } from 'framer-motion';
import { getCurrentUser } from '@/lib/auth';
import { getProfessionalRequests, sendParentRequest, ParentRequest } from '@/lib/parent-requests';
import { addPatient } from '@/lib/patients';
import { createPatientAccess } from '@/lib/patient-access';
import { showToast } from '@/components/ui/toast';

export default function ProfessionalRequestsPage() {
  const [requests, setRequests] = useState<ParentRequest[]>([]);
  const [loading, setLoading] = useState(true);
  const [showSendModal, setShowSendModal] = useState(false);
  const [formData, setFormData] = useState({ parentName: '', parentEmail: '', childName: '' });
  const [formError, setFormError] = useState('');
  const [sending, setSending] = useState(false);

  const loadRequests = () => {
    const user = getCurrentUser();
    if (user) {
      getProfessionalRequests(user.id)
        .then(setRequests)
        .catch(console.error)
        .finally(() => setLoading(false));
    }
  };

  useEffect(() => { loadRequests(); }, []);

  const handleSendRequest = async () => {
    const user = getCurrentUser();
    if (!user) return;
    if (!formData.parentName || !formData.parentEmail || !formData.childName) {
      setFormError('All fields are required.');
      return;
    }
    setSending(true);
    setFormError('');
    try {
      console.log('[SendRequest] Step 1: addPatient');
      const newPatient = await addPatient(user.id, {
        name: formData.childName,
        dob: '',
        parentName: formData.parentName,
        email: formData.parentEmail,
        phone: '',
        lastVisit: '',
        risk: 'Unknown',
      });
      console.log('[SendRequest] Step 1 OK, patient:', newPatient?.id);

      console.log('[SendRequest] Step 2: sendParentRequest');
      await sendParentRequest({
        professionalId: user.id,
        professionalName: user.name,
        patientId: newPatient.id,
        patientName: newPatient.name,
        parentEmail: formData.parentEmail,
        parentName: formData.parentName,
      });
      console.log('[SendRequest] Step 2 OK');

      console.log('[SendRequest] Step 3: createPatientAccess');
      await createPatientAccess({
        patientId: newPatient.id,
        patientName: newPatient.name,
        professionalId: user.id,
        professionalName: user.name,
        parentName: formData.parentName,
        parentEmail: formData.parentEmail,
      });
      console.log('[SendRequest] Step 3 OK');

      showToast('success', 'Request Sent', `Connection request sent to ${formData.parentName}.`);
      setShowSendModal(false);
      setFormData({ parentName: '', parentEmail: '', childName: '' });
      loadRequests();
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error('[SendRequest] Error:', msg, err);
      setFormError(`Failed to send request: ${msg}`);
    } finally {
      setSending(false);
    }
  };

  const statusConfig = {
    pending: { icon: Clock, label: 'Pending', bg: 'rgba(217, 119, 6, 0.1)', color: '#d97706' },
    accepted: { icon: CheckCircle, label: 'Accepted', bg: 'rgba(22, 163, 74, 0.1)', color: '#16a34a' },
    declined: { icon: XCircle, label: 'Declined', bg: 'rgba(220, 38, 38, 0.1)', color: '#dc2626' },
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
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: 'var(--foreground)' }}>Parent Requests</h1>
          <p style={{ color: 'var(--text-muted)' }}>Invite parents to connect and view their child&apos;s results</p>
        </div>
        <motion.button whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
          onClick={() => { setShowSendModal(true); setFormError(''); }}
          className="premium-btn premium-btn-primary">
          <UserPlus className="w-4 h-4" /> Send Request
        </motion.button>
      </div>

      {requests.length === 0 ? (
        <div className="py-16 text-center">
          <UserPlus className="w-12 h-12 mx-auto mb-4" style={{ color: 'var(--text-muted)' }} />
          <p className="text-lg font-medium" style={{ color: 'var(--foreground)' }}>No requests yet</p>
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
            When you add a patient with a parent email, a request will appear here.
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {requests.map((req) => {
            const config = statusConfig[req.status] || statusConfig.pending;
            return (
              <motion.div
                key={req.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="p-5 rounded-2xl" style={{ backgroundColor: 'var(--background)', border: '1px solid var(--border)' }}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className="w-11 h-11 rounded-full flex items-center justify-center text-white font-semibold" style={{ backgroundColor: 'var(--primary)' }}>
                      {req.parentName.charAt(0)}
                    </div>
                    <div>
                      <h3 className="font-semibold" style={{ color: 'var(--foreground)' }}>{req.parentName}</h3>
                      <div className="flex items-center gap-2 text-sm" style={{ color: 'var(--text-muted)' }}>
                        <Mail className="w-3 h-3" />
                        {req.parentEmail}
                      </div>
                      <p className="text-sm" style={{ color: 'var(--primary)' }}>
                        Patient: {req.patientName}
                      </p>
                    </div>
                  </div>
                  <span className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium" style={{ backgroundColor: config.bg, color: config.color }}>
                    <config.icon className="w-4 h-4" />
                    {config.label}
                  </span>
                </div>
              </motion.div>
            );
          })}
        </div>
      )}

      {/* Send Request Modal */}
      {showSendModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm p-4"
          onClick={() => !sending && setShowSendModal(false)}>
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            onClick={(e) => e.stopPropagation()}
            className="w-full max-w-md premium-card p-6"
          >
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold" style={{ color: 'var(--foreground)' }}>Invite a Parent</h2>
              <button onClick={() => !sending && setShowSendModal(false)}
                className="p-1 rounded-lg hover:bg-black/5 transition-colors">
                <X className="w-5 h-5" />
              </button>
            </div>
            <p className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>
              Send a connection request so the parent can view their child&apos;s assessment results.
            </p>
            {formError && (
              <div className="p-3 rounded-xl text-sm mb-4" style={{ backgroundColor: 'var(--risk-high-bg)', color: 'var(--risk-high)' }}>
                {formError}
              </div>
            )}
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1" style={{ color: 'var(--foreground)' }}>
                  Child Name <span className="text-red-500">*</span>
                </label>
                <input type="text" placeholder="e.g. Alex Johnson"
                  value={formData.childName}
                  onChange={(e) => setFormData(prev => ({ ...prev, childName: e.target.value }))}
                  className="premium-input w-full" />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1" style={{ color: 'var(--foreground)' }}>
                  Parent Name <span className="text-red-500">*</span>
                </label>
                <input type="text" placeholder="e.g. Sarah Johnson"
                  value={formData.parentName}
                  onChange={(e) => setFormData(prev => ({ ...prev, parentName: e.target.value }))}
                  className="premium-input w-full" />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1" style={{ color: 'var(--foreground)' }}>
                  Parent Email <span className="text-red-500">*</span>
                </label>
                <input type="email" placeholder="sarah@example.com"
                  value={formData.parentEmail}
                  onChange={(e) => setFormData(prev => ({ ...prev, parentEmail: e.target.value }))}
                  className="premium-input w-full" />
              </div>
            </div>
            <div className="flex gap-3 mt-6">
              <button onClick={() => { setShowSendModal(false); setFormError(''); }}
                disabled={sending}
                className="premium-btn premium-btn-ghost flex-1 disabled:opacity-50">Cancel</button>
              <motion.button whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
                onClick={handleSendRequest} disabled={sending}
                className="premium-btn premium-btn-primary flex-1 disabled:opacity-50">
                {sending ? 'Sending...' : <><Send className="w-4 h-4" /> Send Request</>}
              </motion.button>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  );
}
