'use client';

import { FileText, Calendar, TrendingUp, AlertCircle, Loader2, MessageSquare, Send } from 'lucide-react';
import { motion } from 'framer-motion';
import { useState, useEffect } from 'react';
import Link from 'next/link';
import { getCurrentUser } from '@/lib/auth';
import { getPatientLinksByParent, PatientAccessLink } from '@/lib/patient-access';
import { getAssessmentsByPatient, AssessmentResult } from '@/lib/assessments';
import { isDemoUser, getDemoLinksByParent, getDemoAssessmentsByPatient } from '@/lib/demo-data';

const riskColors: Record<string, string> = {
  'High Risk': 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400',
  'Moderate Risk': 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400',
  'Low Risk': 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
  Unknown: 'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400',
};

const riskColorValue: Record<string, string> = {
  'High Risk': '#dc2626',
  'Moderate Risk': '#d97706',
  'Low Risk': '#16a34a',
};

export default function ParentResultsPage() {
  const [links, setLinks] = useState<PatientAccessLink[]>([]);
  const [assessments, setAssessments] = useState<AssessmentResult[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const user = getCurrentUser();
    if (!user) {
      setLoading(false);
      return;
    }

    const loadData = async () => {
      if (isDemoUser(user.id)) {
        setLinks(getDemoLinksByParent() as any);
        setAssessments(getDemoAssessmentsByPatient());
      } else {
        const linksData = await getPatientLinksByParent(user.id);
        setLinks(linksData);

        const allAssessments: AssessmentResult[] = [];
        for (const link of linksData) {
          try {
            const childAssessments = await getAssessmentsByPatient(link.professionalId, link.patientId);
            allAssessments.push(...childAssessments.filter(a => a.shared));
          } catch (err) {
            console.warn(`Failed to fetch assessments for ${link.patientName}:`, err);
          }
        }

        allAssessments.sort((a, b) => {
          const tA = (a.createdAt as any)?.toMillis?.() || 0;
          const tB = (b.createdAt as any)?.toMillis?.() || 0;
          return tB - tA;
        });

        setAssessments(allAssessments);
      }
    };
    loadData().catch(console.error).finally(() => setLoading(false));
  }, []);

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
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Assessment Results</h1>
        <p className="text-gray-500 dark:text-gray-400">View your children&apos;s screening results</p>
      </div>

      {/* Disclaimer */}
      <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-xl flex items-start gap-3">
        <AlertCircle className="w-5 h-5 text-yellow-600 dark:text-yellow-400 flex-shrink-0 mt-0.5" />
        <div>
          <p className="font-medium text-yellow-800 dark:text-yellow-200">Important Notice</p>
          <p className="text-sm text-yellow-700 dark:text-yellow-300">
            This is a research demo and NOT a diagnostic tool. Results should not be used for clinical decision-making.
            Consult a qualified healthcare professional for diagnosis.
          </p>
        </div>
      </div>

      {assessments.length === 0 ? (
        <div className="py-16 text-center">
          <FileText className="w-12 h-12 mx-auto mb-4" style={{ color: 'var(--text-muted)' }} />
          <p className="text-lg font-medium" style={{ color: 'var(--foreground)' }}>No results yet</p>
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
            When a professional completes an assessment, results will appear here.
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {assessments.map((result, index) => (
            <motion.div
              key={result.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="p-6 bg-white dark:bg-dark-surface rounded-2xl border border-gray-200 dark:border-dark-deep"
            >
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-4">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 rounded-full bg-primary-light dark:bg-primary-dark/40 flex items-center justify-center">
                    <FileText className="w-6 h-6 text-primary dark:text-primary-light" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white">{result.patientName}</h3>
                    <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400">
                      <Calendar className="w-4 h-4" />
                      {result.date}
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-4">
                  <div className="text-right">
                    <p className="text-3xl font-bold text-gray-900 dark:text-white">{(result.ensemble_probability * 100).toFixed(0)}%</p>
                    <p className="text-sm text-gray-500 dark:text-gray-400">ASD Probability</p>
                  </div>
                  <span className={`px-4 py-2 rounded-full text-lg font-semibold ${riskColors[result.risk_level] || riskColors.Unknown}`}>
                    {result.risk_level}
                  </span>
                </div>
              </div>

              {/* Model Breakdown */}
              {result.model_predictions && Object.keys(result.model_predictions).length > 0 && (
                <div className="pt-4 border-t border-gray-100 dark:border-dark-deep">
                  <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Model Predictions:</p>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    {Object.entries(result.model_predictions).map(([model, pred]) => (
                      <div key={model} className="p-3 bg-gray-50 dark:bg-dark-bg rounded-lg">
                        <p className="text-xs text-gray-500 dark:text-gray-400 uppercase">{model}</p>
                        <p className="text-lg font-semibold text-gray-900 dark:text-white">{(pred.probability * 100).toFixed(0)}%</p>
                        <p className="text-xs" style={{ color: riskColorValue[pred.risk_level] || '#6b7280' }}>{pred.risk_level}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Doctor's Notes */}
              {result.sharedNotes && (
                <div className="pt-4 border-t border-gray-100 dark:border-dark-deep">
                  <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Doctor&apos;s Notes:</p>
                  <div className="p-3 rounded-xl text-sm" style={{ backgroundColor: 'var(--background-alt)' }}>
                    <p style={{ color: 'var(--foreground)' }}>{result.sharedNotes}</p>
                  </div>
                </div>
              )}

              {/* Discuss in Chat */}
              <div className="pt-4 border-t border-gray-100 dark:border-dark-deep">
                <Link
                  href="/parent/messages"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all hover:scale-[1.02]"
                  style={{ backgroundColor: 'var(--primary)', color: 'white' }}
                >
                  <MessageSquare className="w-4 h-4" />
                  Discuss Results with Doctor
                </Link>
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
}
