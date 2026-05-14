'use client';

import { useState, useEffect } from 'react';
import { CheckCircle, XCircle, UserPlus, Stethoscope } from 'lucide-react';
import { motion } from 'framer-motion';
import { getCurrentUser } from '@/lib/auth';
import { getParentRequestsByEmail, acceptParentRequest, declineParentRequest, ParentRequest } from '@/lib/parent-requests';
import { addNotification } from '@/lib/notifications';
import { showToast } from '@/components/ui/toast';

export default function ParentRequestsPage() {
  const [requests, setRequests] = useState<ParentRequest[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const user = getCurrentUser();
    if (user) {
      getParentRequestsByEmail(user.email)
        .then(setRequests)
        .catch(console.error)
        .finally(() => setLoading(false));
    }
  }, []);

  const handleAccept = async (req: ParentRequest) => {
    const user = getCurrentUser();
    if (!user) return;

    try {
      await acceptParentRequest(req.id, user.id);

      showToast('success', 'Connection Accepted', `You are now connected with ${req.professionalName} for ${req.patientName}.`);

      // Notify professional
      await addNotification({
        userId: req.professionalId,
        type: 'request_accepted',
        title: 'Request Accepted',
        message: `${user.name} accepted your connection request for ${req.patientName}.`,
        link: '/professional/messages',
      });

      setRequests(prev => prev.filter(r => r.id !== req.id));
    } catch (err) {
      console.error('Failed to accept:', err);
      showToast('error', 'Failed', 'Could not accept the request. Please try again.');
    }

  const handleDecline = async (reqId: string) => {
    try {
      await declineParentRequest(reqId);
      setRequests(prev => prev.filter(r => r.id !== reqId));
      showToast('success', 'Request Declined', 'The connection request has been declined.');
    } catch (err) {
      console.error('Failed to decline:', err);
    }
  };

  const handleDecline = async (reqId: string) => {
    try {
      await declineParentRequest(reqId);
      setRequests(prev => prev.filter(r => r.id !== reqId));
    } catch (err) {
      console.error('Failed to decline:', err);
    }
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
      <div>
        <h1 className="text-2xl font-bold" style={{ color: 'var(--foreground)' }}>Connection Requests</h1>
        <p style={{ color: 'var(--text-muted)' }}>Connect with healthcare professionals</p>
      </div>

      {requests.length === 0 ? (
        <div className="py-16 text-center">
          <UserPlus className="w-12 h-12 mx-auto mb-4" style={{ color: 'var(--text-muted)' }} />
          <p className="text-lg font-medium" style={{ color: 'var(--foreground)' }}>No pending requests</p>
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
            When a professional adds you to their patient list, you&apos;ll see the request here.
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {requests.map((req) => (
            <motion.div
              key={req.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="p-6 rounded-2xl" style={{ backgroundColor: 'var(--background)', border: '1px solid var(--border)' }}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-4">
                  <div className="w-14 h-14 rounded-full flex items-center justify-center text-white" style={{ backgroundColor: 'var(--primary)' }}>
                    <Stethoscope className="w-7 h-7" />
                  </div>
                  <div>
                    <h3 className="font-semibold" style={{ color: 'var(--foreground)' }}>{req.professionalName}</h3>
                    <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                      Has requested access to your child&apos;s records
                    </p>
                    <p className="text-sm font-medium mt-1" style={{ color: 'var(--primary)' }}>
                      Patient: {req.patientName}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={() => handleAccept(req)}
                    className="flex items-center gap-2 px-4 py-2 rounded-lg text-white text-sm font-medium"
                    style={{ backgroundColor: '#16a34a' }}
                  >
                    <CheckCircle className="w-4 h-4" />
                    Accept
                  </button>
                  <button
                    onClick={() => handleDecline(req.id)}
                    className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium"
                    style={{ backgroundColor: 'var(--background-alt)', color: '#dc2626', border: '1px solid #dc2626' }}
                  >
                    <XCircle className="w-4 h-4" />
                    Decline
                  </button>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
}
