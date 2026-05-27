'use client';

import { useParams } from 'next/navigation';
import { useState, useEffect } from 'react';
import { Baby, Calendar, FileText, ArrowLeft, Loader2, Phone, Mail, Send, Stethoscope, MessageSquare, Info, CalendarDays } from 'lucide-react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import { getCurrentUser } from '@/lib/auth';
import { getPatientLinksByParent, PatientAccessLink } from '@/lib/patient-access';
import { getAssessmentsByPatient, AssessmentResult } from '@/lib/assessments';
import { isDemoUser, getDemoLinksByParent, getDemoAssessmentsByPatient, DEMO_CHILD_ID } from '@/lib/demo-data';

const riskColorValue: Record<string, string> = {
  'High Risk': '#dc2626',
  'Moderate Risk': '#d97706',
  'Low Risk': '#16a34a',
  Unknown: '#6b7280',
};

export default function ChildDetailPage() {
  const params = useParams();
  const childId = params.id as string;

  const [link, setLink] = useState<PatientAccessLink | null>(null);
  const [assessments, setAssessments] = useState<AssessmentResult[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const user = getCurrentUser();
    if (!user || !childId) {
      setLoading(false);
      return;
    }

    const loadData = async () => {
      if (isDemoUser(user.id)) {
        const demoLinks = getDemoLinksByParent() as any;
        const found = demoLinks.find((l: any) => l.patientId === childId);
        if (!found) {
          setLoading(false);
          return;
        }
        setLink(found);
        const all = getDemoAssessmentsByPatient();
        setAssessments(all.filter(a => a.shared));
      } else {
        const links = await getPatientLinksByParent(user.id);
        const found = links.find((l) => l.patientId === childId);
        if (!found) {
          setLoading(false);
          return;
        }
        setLink(found);

        const childAssessments = await getAssessmentsByPatient(found.professionalId, childId);
        const sharedAssessments = childAssessments.filter(a => a.shared);
        sharedAssessments.sort((a, b) => {
          const tA = (a.createdAt as any)?.toMillis?.() || 0;
          const tB = (b.createdAt as any)?.toMillis?.() || 0;
          return tB - tA;
        });
        setAssessments(sharedAssessments);
      }
    };
    loadData().catch(console.error).finally(() => setLoading(false));
  }, [childId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="w-8 h-8 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
      </div>
    );
  }

  if (!link) {
    return (
      <div className="text-center py-16">
        <p className="text-lg font-medium" style={{ color: 'var(--foreground)' }}>Child not found</p>
        <Link href="/parent/children" className="text-primary hover:underline mt-2 inline-block">
          Back to children
        </Link>
      </div>
    );
  }

  const latestAssessment = assessments[0];

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      {/* Back Link */}
      <Link href="/parent/children" className="inline-flex items-center gap-2 text-sm" style={{ color: 'var(--primary)' }}>
        <ArrowLeft className="w-4 h-4" />
        Back to Children
      </Link>

      {/* Header */}
      <div className="p-6 rounded-2xl" style={{ backgroundColor: 'var(--background)', border: '1px solid var(--border)' }}>
        <div className="flex items-center gap-4 mb-4">
          <div className="w-16 h-16 rounded-full flex items-center justify-center text-white text-2xl font-bold" style={{ backgroundColor: 'var(--primary)' }}>
            {link.patientName.charAt(0)}
          </div>
          <div>
            <h1 className="text-2xl font-bold" style={{ color: 'var(--foreground)' }}>{link.patientName}</h1>
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Patient Profile</p>
          </div>
        </div>

        {/* Child Info */}
        <div className="grid grid-cols-2 gap-4 mb-4">
          {(link as any).age !== undefined && (
            <div className="flex items-center gap-2 p-3 rounded-xl" style={{ backgroundColor: 'var(--background-alt)' }}>
              <CalendarDays className="w-5 h-5" style={{ color: 'var(--primary)' }} />
              <div>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Age</p>
                <p className="font-semibold" style={{ color: 'var(--foreground)' }}>{(link as any).age} years</p>
              </div>
            </div>
          )}
          {(link as any).dob && (
            <div className="flex items-center gap-2 p-3 rounded-xl" style={{ backgroundColor: 'var(--background-alt)' }}>
              <Calendar className="w-5 h-5" style={{ color: 'var(--primary)' }} />
              <div>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Date of Birth</p>
                <p className="font-semibold" style={{ color: 'var(--foreground)' }}>{(link as any).dob}</p>
              </div>
            </div>
          )}
        </div>

        {(link as any).about && (
          <div className="p-4 rounded-xl mb-4" style={{ backgroundColor: 'var(--background-alt)' }}>
            <p className="text-xs font-medium mb-1" style={{ color: 'var(--text-muted)' }}>About {link.patientName}</p>
            <p className="text-sm" style={{ color: 'var(--foreground)' }}>{(link as any).about}</p>
          </div>
        )}

        {latestAssessment && (
          <div className="flex items-center gap-4 p-4 rounded-xl" style={{ backgroundColor: 'var(--background-alt)' }}>
            <div className="text-center">
              <p className="text-3xl font-bold" style={{ color: riskColorValue[latestAssessment.risk_level] || '#6b7280' }}>
                {(latestAssessment.ensemble_probability * 100).toFixed(0)}%
              </p>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Latest Score</p>
            </div>
            <div className="text-sm" style={{ color: 'var(--text-muted)' }}>
              <p>Risk: <strong style={{ color: riskColorValue[latestAssessment.risk_level] }}>{latestAssessment.risk_level}</strong></p>
              <p>Date: {latestAssessment.date}</p>
            </div>
          </div>
        )}
      </div>

      {/* Professional Info */}
      <div className="p-6 rounded-2xl" style={{ backgroundColor: 'var(--background)', border: '1px solid var(--border)' }}>
        <h2 className="text-lg font-semibold mb-4" style={{ color: 'var(--foreground)' }}>Connected Professional</h2>
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-full flex items-center justify-center text-white" style={{ backgroundColor: 'var(--primary)' }}>
            <Stethoscope className="w-6 h-6" />
          </div>
          <div>
            <p className="font-medium" style={{ color: 'var(--foreground)' }}>{link.professionalName || 'Specialist'}</p>
            <Link href="/parent/messages" className="text-sm" style={{ color: 'var(--primary)' }}>
              <Send className="w-3 h-3 inline mr-1" />
              Send Message
            </Link>
          </div>
        </div>
      </div>

      {/* Assessment History */}
      <div className="p-6 rounded-2xl" style={{ backgroundColor: 'var(--background)', border: '1px solid var(--border)' }}>
        <h2 className="text-lg font-semibold mb-4" style={{ color: 'var(--foreground)' }}>Assessment History</h2>
        {assessments.length === 0 ? (
          <p style={{ color: 'var(--text-muted)' }}>No assessments yet.</p>
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
                      <p className="font-medium" style={{ color: 'var(--foreground)' }}>{(a.ensemble_probability * 100).toFixed(1)}% — {a.risk_level}</p>
                      <div className="flex items-center gap-2 text-xs" style={{ color: 'var(--text-muted)' }}>
                        <Calendar className="w-3 h-3" />
                        {a.date}
                        {a.source && <span>· {a.source}</span>}
                      </div>
                    </div>
                  </div>
                  <div className="flex gap-1">
                    {Object.entries(a.model_predictions || {}).map(([model, pred]) => (
                      <span key={model} className="px-2 py-0.5 rounded text-xs font-medium" style={{ backgroundColor: 'var(--primary-light)', color: 'var(--primary)' }}>
                        {model}: {(pred.probability * 100).toFixed(0)}%
                      </span>
                    ))}
                  </div>
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
                <div className="mt-2 flex justify-end">
                  <Link
                    href="/parent/messages"
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all hover:scale-[1.02]"
                    style={{ backgroundColor: 'var(--primary)', color: 'white' }}
                  >
                    <MessageSquare className="w-3.5 h-3.5" />
                    Discuss with Doctor
                  </Link>
                </div>
              </motion.div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
