'use client';

import { Baby, FileText, MessageSquare, Calendar, AlertCircle, Send, Heart, Activity } from 'lucide-react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { useState, useEffect } from 'react';
import { getCurrentUser } from '@/lib/auth';
import { getPatientLinksByParent, PatientAccessLink } from '@/lib/patient-access';
import { getAssessmentsByPatient, AssessmentResult } from '@/lib/assessments';

const container = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.08 } },
};

const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: [0.16, 1, 0.3, 1] } },
};

const riskColors: Record<string, string> = {
  'High Risk': 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400',
  'Moderate Risk': 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400',
  'Low Risk': 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
  Unknown: 'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400',
};

export default function ParentDashboard() {
  const [links, setLinks] = useState<PatientAccessLink[]>([]);
  const [assessments, setAssessments] = useState<AssessmentResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [showContact, setShowContact] = useState(false);

  useEffect(() => {
    const user = getCurrentUser();
    if (!user) { setLoading(false); return; }

    const loadData = async () => {
      const linksData = await getPatientLinksByParent(user.id);
      setLinks(linksData);
      const all: AssessmentResult[] = [];
      for (const link of linksData) {
        try {
          const childAssessments = await getAssessmentsByPatient(link.professionalId, link.patientId);
          all.push(...childAssessments.filter(a => a.shared));
        } catch { /* skip */ }
      }
      all.sort((a, b) => {
        const tA = (a.createdAt as any)?.toMillis?.() || 0;
        const tB = (b.createdAt as any)?.toMillis?.() || 0;
        return tB - tA;
      });
      setAssessments(all);
    };
    loadData().catch(console.error).finally(() => setLoading(false));
  }, []);

  const latest = assessments[0];
  const latestScore = latest ? latest.ensemble_probability : 0;
  const latestRisk = latest ? latest.risk_level : 'Unknown';
  const latestDate = latest ? latest.date : 'N/A';
  const childName = links[0]?.patientName || 'your child';

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="h-8 w-48 skeleton" />
        <div className="grid grid-cols-3 gap-4">
          {[1, 2, 3].map(i => <div key={i} className="h-28 skeleton" />)}
        </div>
        <div className="h-48 skeleton" />
      </div>
    );
  }

  return (
    <motion.div variants={container} initial="hidden" animate="show" className="max-w-2xl mx-auto space-y-6">
      <motion.div variants={fadeUp}>
        <h1 className="text-2xl font-bold" style={{ color: 'var(--foreground)' }}>Welcome Back</h1>
        <p style={{ color: 'var(--text-muted)' }}>Here&apos;s an overview of your child&apos;s progress</p>
      </motion.div>

      <motion.div variants={fadeUp} className="grid grid-cols-3 gap-4">
        {[
          { icon: Baby, value: links.length, label: 'Children', delay: 0 },
          { icon: Activity, value: `${(latestScore * 100).toFixed(0)}%`, label: 'Latest Score', delay: 0.1 },
          { icon: Calendar, value: latestDate, label: 'Last Check', delay: 0.2 },
        ].map((item, i) => (
          <motion.div
            key={i}
            whileHover={{ y: -2 }}
            className="premium-card p-4 text-center"
            style={{ animationDelay: `${item.delay}s` }}
          >
            <div
              className="w-10 h-10 rounded-xl flex items-center justify-center mx-auto mb-2"
              style={{ background: 'var(--gradient-primary-subtle)' }}
            >
              <item.icon className="w-5 h-5" style={{ color: 'var(--primary)' }} />
            </div>
            <p className="text-2xl font-bold" style={{ color: 'var(--foreground)' }}>{item.value}</p>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{item.label}</p>
          </motion.div>
        ))}
      </motion.div>

      <motion.div variants={fadeUp}>
        <div className="premium-card p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold" style={{ color: 'var(--foreground)' }}>
              <Heart className="w-5 h-5 inline mr-2" style={{ color: 'var(--primary)' }} />
              My Child
            </h2>
            <span className={`px-3 py-1 rounded-full text-sm font-medium ${riskColors[latestRisk] || riskColors.Unknown}`}>
              {latestRisk}
            </span>
          </div>

          <div className="flex items-center gap-4 mb-4">
            <div
              className="w-16 h-16 rounded-full flex items-center justify-center text-white text-2xl font-bold"
              style={{ background: 'var(--gradient-primary)' }}
            >
              {childName.charAt(0)}
            </div>
            <div>
              <h3 className="text-xl font-bold" style={{ color: 'var(--foreground)' }}>{childName}</h3>
              <p style={{ color: 'var(--text-dim)' }}>{assessments.length} assessment{assessments.length !== 1 ? 's' : ''}</p>
            </div>
          </div>

          {links[0] && (
            <div className="pt-4 border-t" style={{ borderColor: 'var(--border-light)' }}>
              <p className="text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>Assigned Specialist</p>
              <div className="flex items-center justify-between">
                <p style={{ color: 'var(--foreground)' }}>{links[0].professionalName || 'Specialist'}</p>
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => setShowContact(!showContact)}
                  className="px-4 py-2 rounded-xl text-sm font-medium text-white"
                  style={{ background: 'var(--gradient-primary)' }}
                >
                  Contact
                </motion.button>
              </div>
            </div>
          )}

          {showContact && links[0] && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              className="mt-4 pt-4 border-t space-y-2" style={{ borderColor: 'var(--border-light)' }}>
              <Link
                href="/parent/messages"
                className="flex items-center gap-3 p-3 rounded-xl transition-colors"
                style={{ backgroundColor: 'var(--background-alt)' }}
              >
                <Send className="w-5 h-5" style={{ color: 'var(--primary)' }} />
                <span className="text-sm font-medium" style={{ color: 'var(--foreground)' }}>
                  Send Message to {links[0].professionalName || 'Specialist'}
                </span>
              </Link>
            </motion.div>
          )}
        </div>
      </motion.div>

      <motion.div
        variants={fadeUp}
        className="p-4 rounded-xl flex items-start gap-3"
        style={{
          backgroundColor: 'rgba(184, 134, 11, 0.08)',
          border: '1px solid rgba(184, 134, 11, 0.2)',
        }}
      >
        <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" style={{ color: 'var(--risk-moderate)' }} />
        <p className="text-sm" style={{ color: 'var(--risk-moderate)' }}>
          <strong>Important:</strong> This is a research demo and NOT a diagnostic tool.
          Consult a qualified healthcare professional for medical advice.
        </p>
      </motion.div>

      <motion.div variants={fadeUp} className="grid grid-cols-2 gap-4">
        {[
          { href: '/parent/results', icon: FileText, label: 'View Results' },
          { href: '/parent/messages', icon: MessageSquare, label: 'Message Specialist' },
        ].map((action, i) => (
          <Link key={i} href={action.href}>
            <motion.div
              whileHover={{ y: -2 }}
              className="premium-card p-5 flex items-center gap-4 transition-all cursor-pointer"
            >
              <div
                className="w-12 h-12 rounded-xl flex items-center justify-center"
                style={{ background: 'var(--gradient-primary-subtle)' }}
              >
                <action.icon className="w-6 h-6" style={{ color: 'var(--primary)' }} />
              </div>
              <span className="font-medium text-sm" style={{ color: 'var(--foreground)' }}>{action.label}</span>
            </motion.div>
          </Link>
        ))}
      </motion.div>
    </motion.div>
  );
}
