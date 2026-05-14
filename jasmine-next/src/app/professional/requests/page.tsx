'use client';

import { useState, useEffect } from 'react';
import { UserPlus, CheckCircle, XCircle, Clock, Mail } from 'lucide-react';
import { motion } from 'framer-motion';
import { getCurrentUser } from '@/lib/auth';
import { getProfessionalRequests, ParentRequest } from '@/lib/parent-requests';

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
      <div>
        <h1 className="text-2xl font-bold" style={{ color: 'var(--foreground)' }}>Parent Requests</h1>
        <p style={{ color: 'var(--text-muted)' }}>History of parent connection requests</p>
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
    </div>
  );
}
