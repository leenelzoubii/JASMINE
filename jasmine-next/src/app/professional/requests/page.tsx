'use client';

import { useState, useEffect } from 'react';
import { CheckCircle, XCircle, Clock, Mail, UserPlus } from 'lucide-react';
import { motion } from 'framer-motion';
import { getCurrentUser } from '@/lib/auth';
import { getProfessionalRequests, acceptParentRequest, declineParentRequest, ParentRequest } from '@/lib/parent-requests';
import { addNotification } from '@/lib/notifications';

export default function ProfessionalRequestsPage() {
  const [requests, setRequests] = useState<ParentRequest[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const user = getCurrentUser();
    if (user) {
      getProfessionalRequests(user.id)
        .then(setRequests)
        .catch(console.error)
        .finally(() => setLoading(false));
    }
  }, []);

  const handleAccept = async (req: ParentRequest) => {
    // Need to find parentId by email - in a real app you'd look up the parent
    // For demo, we auto-accept
    try {
      await acceptParentRequest(req.id, req.parentId || 'unknown');
      setRequests(prev => prev.map(r => r.id === req.id ? { ...r, status: 'accepted' as const } : r));
    } catch (err) {
      console.error('Failed to accept:', err);
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
        <h1 className="text-2xl font-bold" style={{ color: 'var(--foreground)' }}>Parent Requests</h1>
        <p style={{ color: 'var(--text-muted)' }}>Manage parent connection requests</p>
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
                  <div className="w-12 h-12 rounded-full flex items-center justify-center text-white" style={{ backgroundColor: 'var(--primary)' }}>
                    {req.parentName.charAt(0)}
                  </div>
                  <div>
                    <h3 className="font-semibold" style={{ color: 'var(--foreground)' }}>{req.parentName}</h3>
                    <div className="flex items-center gap-2 text-sm" style={{ color: 'var(--text-muted)' }}>
                      <Mail className="w-3 h-3" />
                      {req.parentEmail}
                    </div>
                    <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                      Patient: {req.patientName}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  {req.status === 'pending' && (
                    <>
                      <button
                        onClick={() => handleAccept(req)}
                        className="p-2 rounded-lg" style={{ backgroundColor: 'rgba(22, 163, 74, 0.1)' }}
                      >
                        <CheckCircle className="w-5 h-5" style={{ color: '#16a34a' }} />
                      </button>
                      <button
                        onClick={() => handleDecline(req.id)}
                        className="p-2 rounded-lg" style={{ backgroundColor: 'rgba(220, 38, 38, 0.1)' }}
                      >
                        <XCircle className="w-5 h-5" style={{ color: '#dc2626' }} />
                      </button>
                    </>
                  )}
                  {req.status === 'accepted' && (
                    <span className="px-3 py-1 rounded-full text-sm font-medium" style={{ backgroundColor: 'rgba(22, 163, 74, 0.1)', color: '#16a34a' }}>
                      Accepted
                    </span>
                  )}
                  {req.status === 'declined' && (
                    <span className="px-3 py-1 rounded-full text-sm font-medium" style={{ backgroundColor: 'rgba(220, 38, 38, 0.1)', color: '#dc2626' }}>
                      Declined
                    </span>
                  )}
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
}
