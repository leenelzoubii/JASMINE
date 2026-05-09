'use client';

import { Users, FileText, MessageSquare, Clock, TrendingUp, ArrowRight } from 'lucide-react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { useState, useEffect } from 'react';
import { getCurrentUser } from '@/lib/auth';
import { getPatients } from '@/lib/patients';
import { Patient } from '@/lib/patients';

const container = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.1 } },
};

const item = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0 },
};

const riskColors: Record<string, string> = {
  High: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400',
  Moderate: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400',
  Low: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
  Unknown: 'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400',
};

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

export default function ProfessionalDashboard() {
  const [mounted, setMounted] = useState(false);
  const [patients, setPatients] = useState<Patient[]>([]);
  const [userName, setUserName] = useState('');

  useEffect(() => {
    setMounted(true);
    const user = getCurrentUser();
    if (user) {
      setUserName(user.name);
      getPatients(user.id)
        .then(setPatients)
        .catch(console.error);
    }
  }, []);

  const stats = [
    { label: 'Total Patients', value: patients.length, icon: Users, change: null, color: 'bg-blue-500' },
    { label: 'Pending Assessments', value: 0, icon: FileText, change: null, color: 'bg-orange-500' },
    { label: 'Unread Messages', value: 0, icon: MessageSquare, change: null, color: 'bg-purple-500' },
    { label: 'This Month', value: 0, icon: TrendingUp, change: null, color: 'bg-green-500' },
  ];

  const recentPatients = patients.slice(0, 3);

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Dashboard</h1>
        <p className="text-gray-500 dark:text-gray-400">
          {userName ? `Welcome back, ${userName}` : 'Welcome back'}
        </p>
      </div>

      {/* Stats Grid */}
      <motion.div
        variants={container}
        initial="hidden"
        animate={mounted ? 'show' : 'hidden'}
        className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4"
      >
        {stats.map((stat, index) => (
          <motion.div
            key={index}
            variants={item}
            className="p-6 bg-white dark:bg-dark-surface rounded-2xl border border-primary-light/30 dark:border-dark-deep hover:shadow-lg transition-shadow"
          >
            <div className="flex items-center justify-between mb-4">
              <div className={`w-12 h-12 rounded-xl ${stat.color} flex items-center justify-center`}>
                <stat.icon className="w-6 h-6 text-white" />
              </div>
            </div>
            <p className="text-3xl font-bold text-gray-900 dark:text-white">{stat.value}</p>
            <p className="text-sm text-gray-500 dark:text-gray-400">{stat.label}</p>
          </motion.div>
        ))}
      </motion.div>

      {/* Recent Patients & Activity */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* Recent Patients */}
        <div className="p-6 bg-white dark:bg-dark-surface rounded-2xl border border-primary-light/30 dark:border-dark-deep">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Recent Patients</h2>
            <Link href="/professional/patients" className="text-sm text-primary hover:text-primary-dark flex items-center gap-1">
              View all <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
          {recentPatients.length === 0 ? (
            <p className="text-gray-500 dark:text-gray-400 text-sm">No patients yet. Add your first patient to get started.</p>
          ) : (
            <div className="space-y-4">
              {recentPatients.map((patient) => (
                <div key={patient.id} className="flex items-center justify-between p-4 bg-gray-50 dark:bg-dark-bg rounded-xl">
                  <div>
                    <p className="font-medium text-gray-900 dark:text-white">{patient.name}</p>
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      Age: {patient.dob ? calculateAge(patient.dob) : 'N/A'} &bull; Last: {patient.lastVisit}
                    </p>
                  </div>
                  <span className={`px-3 py-1 rounded-full text-sm font-medium ${riskColors[patient.risk] || ''}`}>
                    {patient.risk}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Quick Actions */}
        <div className="p-6 bg-white dark:bg-dark-surface rounded-2xl border border-primary-light/30 dark:border-dark-deep">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">Quick Actions</h2>
          <div className="space-y-3">
            <Link href="/professional/assessments" className="flex items-center gap-4 p-4 bg-primary-light/30 dark:bg-primary-dark/20 rounded-xl hover:bg-primary-light/50 transition-colors group">
              <div className="w-10 h-10 rounded-xl bg-primary flex items-center justify-center">
                <FileText className="w-5 h-5 text-white" />
              </div>
              <div className="flex-1">
                <p className="font-medium text-gray-900 dark:text-white">New Assessment</p>
                <p className="text-sm text-gray-500 dark:text-gray-400">Run screening for a patient</p>
              </div>
              <ArrowRight className="w-5 h-5 text-gray-400 group-hover:translate-x-1 transition-transform" />
            </Link>
            <Link href="/professional/patients" className="flex items-center gap-4 p-4 bg-primary-light/30 dark:bg-primary-dark/20 rounded-xl hover:bg-primary-light/50 transition-colors group">
              <div className="w-10 h-10 rounded-xl bg-primary flex items-center justify-center">
                <Users className="w-5 h-5 text-white" />
              </div>
              <div className="flex-1">
                <p className="font-medium text-gray-900 dark:text-white">Add Patient</p>
                <p className="text-sm text-gray-500 dark:text-gray-400">Register a new patient</p>
              </div>
              <ArrowRight className="w-5 h-5 text-gray-400 group-hover:translate-x-1 transition-transform" />
            </Link>
            <Link href="/professional/messages" className="flex items-center gap-4 p-4 bg-primary-light/30 dark:bg-primary-dark/20 rounded-xl hover:bg-primary-light/50 transition-colors group">
              <div className="w-10 h-10 rounded-xl bg-primary flex items-center justify-center">
                <MessageSquare className="w-5 h-5 text-white" />
              </div>
              <div className="flex-1">
                <p className="font-medium text-gray-900 dark:text-white">Message Parent</p>
                <p className="text-sm text-gray-500 dark:text-gray-400">Send update to guardians</p>
              </div>
              <ArrowRight className="w-5 h-5 text-gray-400 group-hover:translate-x-1 transition-transform" />
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}