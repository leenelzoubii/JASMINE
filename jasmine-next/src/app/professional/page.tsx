'use client';

import { Users, FileText, MessageSquare, TrendingUp, ArrowRight, Activity, Calendar } from 'lucide-react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { useState, useEffect } from 'react';
import { getCurrentUser } from '@/lib/auth';
import { getPatients } from '@/lib/patients';
import { Patient } from '@/lib/patients';

const container = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.08 } },
};

const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: [0.16, 1, 0.3, 1] } },
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
  if (m < 0 || (m === 0 && today.getDate() < birth.getDate())) age--;
  return age;
}

export default function ProfessionalDashboard() {
  const [mounted, setMounted] = useState(false);
  const [patients, setPatients] = useState<Patient[]>([]);
  const [userName, setUserName] = useState('');

  const demoPatients: Patient[] = [
    { id: 'demo-1', name: 'Emma Thompson', dob: '2019-03-15', parentName: 'John Thompson', email: 'john@email.com', phone: '+1 555-0123', lastVisit: '2026-05-01', risk: 'High' },
    { id: 'demo-2', name: 'Liam Johnson', dob: '2020-07-22', parentName: 'Sarah Johnson', email: 'sarah@email.com', phone: '+1 555-0124', lastVisit: '2026-04-25', risk: 'Moderate' },
    { id: 'demo-3', name: 'Sophie Williams', dob: '2018-11-05', parentName: 'Mike Williams', email: 'mike@email.com', phone: '+1 555-0125', lastVisit: '2026-04-20', risk: 'Low' },
    { id: 'demo-4', name: 'James Brown', dob: '2021-02-14', parentName: 'Lisa Brown', email: 'lisa@email.com', phone: '+1 555-0126', lastVisit: '2026-04-15', risk: 'Low' },
    { id: 'demo-5', name: 'Olivia Davis', dob: '2019-09-30', parentName: 'Tom Davis', email: 'tom@email.com', phone: '+1 555-0127', lastVisit: '2026-04-10', risk: 'Moderate' },
  ];

  useEffect(() => {
    setMounted(true);
    const user = getCurrentUser();
    if (user) {
      setUserName(user.name);
      getPatients(user.id)
        .then((realPatients) => setPatients(realPatients.length === 0 ? demoPatients : realPatients))
        .catch(() => setPatients(demoPatients));
    }
  }, []);

  const stats = [
    { label: 'Total Patients', value: patients.length, icon: Users, gradient: 'from-blue-500 to-cyan-500' },
    { label: 'Pending Assessments', value: Math.floor(patients.length * 0.3), icon: FileText, gradient: 'from-amber-500 to-orange-500' },
    { label: 'Unread Messages', value: Math.floor(patients.length * 0.4), icon: MessageSquare, gradient: 'from-purple-500 to-pink-500' },
    { label: 'This Month', value: Math.floor(patients.length * 0.2), icon: TrendingUp, gradient: 'from-green-500 to-emerald-500' },
  ];

  const recentPatients = patients.slice(0, 3);

  return (
    <motion.div variants={container} initial="hidden" animate={mounted ? 'show' : 'hidden'} className="space-y-8">
      <motion.div variants={fadeUp}>
        <h1 className="text-3xl font-bold" style={{ color: 'var(--foreground)' }}>Dashboard</h1>
        <p style={{ color: 'var(--text-muted)' }}>
          {userName ? `Welcome back, ${userName}` : 'Welcome back'}
        </p>
      </motion.div>

      <motion.div variants={fadeUp} className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((stat, index) => (
          <motion.div
            key={index}
            whileHover={{ y: -2 }}
            className="premium-card p-5 relative overflow-hidden"
          >
            <div className="absolute top-0 right-0 w-24 h-24 opacity-5 rounded-full -translate-y-1/2 translate-x-1/2" style={{ background: `linear-gradient(135deg, var(--primary), var(--primary-muted))` }} />
            <div
              className="w-11 h-11 rounded-xl flex items-center justify-center mb-3"
              style={{ background: `linear-gradient(135deg, var(--primary), var(--primary-muted))` }}
            >
              <stat.icon className="w-5 h-5 text-white" />
            </div>
            <p className="text-3xl font-bold" style={{ color: 'var(--foreground)' }}>{stat.value}</p>
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>{stat.label}</p>
          </motion.div>
        ))}
      </motion.div>

      <div className="grid lg:grid-cols-2 gap-6">
        <motion.div variants={fadeUp}>
          <div className="premium-card p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-semibold" style={{ color: 'var(--foreground)' }}>Recent Patients</h2>
              <Link
                href="/professional/patients"
                className="text-sm font-medium flex items-center gap-1 transition-colors"
                style={{ color: 'var(--primary)' }}
              >
                View all <ArrowRight className="w-4 h-4" />
              </Link>
            </div>
            {recentPatients.length === 0 ? (
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>No patients yet. Add your first patient to get started.</p>
            ) : (
              <div className="space-y-3">
                {recentPatients.map((patient) => (
                  <motion.div
                    key={patient.id}
                    whileHover={{ x: 3 }}
                    className="flex items-center justify-between p-4 rounded-xl"
                    style={{ backgroundColor: 'var(--background-alt)' }}
                  >
                    <div className="flex items-center gap-3">
                      <div
                        className="w-10 h-10 rounded-full flex items-center justify-center text-white font-bold text-sm"
                        style={{ background: 'var(--gradient-primary)' }}
                      >
                        {patient.name.charAt(0)}
                      </div>
                      <div>
                        <p className="font-medium text-sm" style={{ color: 'var(--foreground)' }}>{patient.name}</p>
                        <div className="flex items-center gap-2 text-xs" style={{ color: 'var(--text-dim)' }}>
                          <Calendar className="w-3 h-3" />
                          Age: {patient.dob ? calculateAge(patient.dob) : 'N/A'}
                          <Activity className="w-3 h-3 ml-1" />
                          Last: {patient.lastVisit}
                        </div>
                      </div>
                    </div>
                    <span className={`px-3 py-1 rounded-full text-xs font-medium ${riskColors[patient.risk] || ''}`}>
                      {patient.risk}
                    </span>
                  </motion.div>
                ))}
              </div>
            )}
          </div>
        </motion.div>

        <motion.div variants={fadeUp}>
          <div className="premium-card p-6">
            <h2 className="text-lg font-semibold mb-6" style={{ color: 'var(--foreground)' }}>Quick Actions</h2>
            <div className="space-y-3">
              {[
                { href: '/professional/assessments', icon: FileText, title: 'New Assessment', desc: 'Run screening for a patient', gradient: true },
                { href: '/professional/patients', icon: Users, title: 'Add Patient', desc: 'Register a new patient' },
                { href: '/professional/messages', icon: MessageSquare, title: 'Message Parent', desc: 'Send update to guardians' },
              ].map((action, i) => (
                <Link key={i} href={action.href}>
                  <motion.div
                    whileHover={{ x: 4 }}
                    className="flex items-center gap-4 p-4 rounded-xl transition-all cursor-pointer group"
                    style={{
                      backgroundColor: action.gradient ? 'var(--gradient-primary-subtle)' : 'var(--background-alt)',
                    }}
                  >
                    <div className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0"
                      style={action.gradient ? { background: 'var(--gradient-primary)' } : { background: 'var(--primary-light)' }}>
                      <action.icon className="w-5 h-5" style={{ color: action.gradient ? 'white' : 'var(--primary)' }} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="font-medium text-sm" style={{ color: 'var(--foreground)' }}>{action.title}</p>
                      <p className="text-xs" style={{ color: 'var(--text-dim)' }}>{action.desc}</p>
                    </div>
                    <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-1" style={{ color: 'var(--text-dim)' }} />
                  </motion.div>
                </Link>
              ))}
            </div>
          </div>
        </motion.div>
      </div>
    </motion.div>
  );
}
