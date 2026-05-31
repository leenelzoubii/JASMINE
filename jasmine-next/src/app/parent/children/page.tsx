'use client';

import { Baby, Calendar, Phone, FileText, Search, Loader2 } from 'lucide-react';
import { useState, useEffect } from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { getCurrentUser } from '@/lib/auth';
import { getPatientLinksByParent } from '@/lib/patient-access';
import { PatientAccessLink } from '@/lib/patient-access';
import { isDemoUser, getDemoLinksByParent } from '@/lib/demo-data';

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

export default function ParentChildrenPage() {
  const [links, setLinks] = useState<PatientAccessLink[]>([]);
  const [loading, setLoading] = useState(true);
  const [mounted, setMounted] = useState(false);
  const [search, setSearch] = useState('');

  useEffect(() => {
    setMounted(true);
    loadLinks();
  }, []);

  const loadLinks = async () => {
    const user = getCurrentUser();
    if (!user) {
      setLoading(false);
      return;
    }

    try {
      if (isDemoUser(user.id)) {
        setLinks(getDemoLinksByParent() as any);
      } else {
        const linksData = await getPatientLinksByParent(user.id);
        setLinks(linksData);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const filteredLinks = mounted ? links.filter(link =>
    link.patientName.toLowerCase().includes(search.toLowerCase())
  ) : [];

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="w-8 h-8 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">My Children</h1>
          <p className="text-gray-500 dark:text-gray-400">View your children&apos;s profiles and assessments</p>
        </div>
      </div>

      {links.length > 1 && (
        <div className="relative max-w-md">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
          <input
            type="text"
            placeholder="Search children..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-12 pr-4 py-3 bg-white dark:bg-dark-surface border border-gray-200 dark:border-dark-deep rounded-xl focus:outline-none focus:ring-2 focus:ring-primary"
          />
        </div>
      )}

      {filteredLinks.length === 0 ? (
        <div className="py-16 text-center">
          <Baby className="w-12 h-12 mx-auto text-gray-400 mb-4" />
          <p className="text-gray-500 dark:text-gray-400">
            {links.length === 0
              ? 'No children linked to your account yet.'
              : 'No children match your search.'}
          </p>
        </div>
      ) : (
        <div className="grid md:grid-cols-2 gap-6">
          {filteredLinks.map((link, index) => (
            <motion.div
              key={link.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="p-6 bg-white dark:bg-dark-surface rounded-2xl border border-gray-200 dark:border-dark-deep"
            >
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-4">
                  <div className="w-16 h-16 rounded-full bg-primary-light dark:bg-primary-dark/40 flex items-center justify-center">
                    <Baby className="w-8 h-8 text-primary dark:text-primary-light" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white">{link.patientName}</h3>
                    <p className="text-gray-500 dark:text-gray-400">Patient</p>
                  </div>
                </div>
              </div>

              <div className="space-y-3 text-sm">
                <div className="flex items-center gap-3 text-gray-600 dark:text-gray-300">
                  <FileText className="w-4 h-4 text-gray-400" />
                  <span>{link.sharedAssessments.length} shared assessment{link.sharedAssessments.length !== 1 ? 's' : ''}</span>
                </div>
              </div>

              <div className="mt-4 pt-4 border-t border-gray-100 dark:border-dark-deep flex gap-2">
                <Link
                  href={`/parent/children/${link.patientId}`}
                  className="flex-1 px-4 py-2 bg-primary rounded-lg font-medium hover:bg-primary-dark transition-colors text-center"
                  style={{ color: '#fff' }}
                >
                  View Profile
                </Link>
                <Link
                  href={`/parent/results`}
                  className="flex-1 px-4 py-2 bg-primary-light/50 dark:bg-primary-dark/20 text-primary dark:text-primary-light rounded-lg font-medium hover:bg-primary-light transition-colors text-center"
                >
                  View Results
                </Link>
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
}
