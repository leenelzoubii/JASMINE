'use client';

import { Baby, FileText, MessageSquare, Phone, Mail, Calendar, AlertCircle, Send } from 'lucide-react';
import { motion } from 'framer-motion';
import { useState } from 'react';

const myChild = {
  name: 'Emma',
  age: 6,
  specialist: 'Dr. Jasmine',
  specialistPhone: '+1 555-0100',
  specialistEmail: 'dr.jasmine@jasmine.com',
  lastAssessment: '2026-05-01',
  score: 0.82,
  risk: 'Moderate'
};

const riskColors: Record<string, string> = {
  High: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400',
  Moderate: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400',
  Low: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
  Unknown: 'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400',
};

export default function ParentDashboard() {
  const [showContact, setShowContact] = useState(false);

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Welcome Back</h1>
        <p className="text-gray-500 dark:text-gray-400">Here&apos;s an overview of your child&apos;s progress</p>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-3 gap-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-4 bg-white dark:bg-dark-surface rounded-2xl border border-gray-200 dark:border-dark-deep"
        >
          <Baby className="w-8 h-8 text-primary mb-2" />
          <p className="text-2xl font-bold text-gray-900 dark:text-white">{myChild.age}</p>
          <p className="text-sm text-gray-500">Age</p>
        </motion.div>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="p-4 bg-white dark:bg-dark-surface rounded-2xl border border-gray-200 dark:border-dark-deep"
        >
          <FileText className="w-8 h-8 text-primary mb-2" />
          <p className="text-2xl font-bold text-gray-900 dark:text-white">{(myChild.score * 100).toFixed(0)}%</p>
          <p className="text-sm text-gray-500">Score</p>
        </motion.div>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="p-4 bg-white dark:bg-dark-surface rounded-2xl border border-gray-200 dark:border-dark-deep"
        >
          <Calendar className="w-8 h-8 text-primary mb-2" />
          <p className="text-lg font-bold text-gray-900 dark:text-white">{myChild.lastAssessment}</p>
          <p className="text-sm text-gray-500">Last Check</p>
        </motion.div>
      </div>

      {/* My Child Card */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="p-6 bg-white dark:bg-dark-surface rounded-2xl border border-gray-200 dark:border-dark-deep"
      >
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">My Child</h2>
          <span className={`px-3 py-1 rounded-full text-sm font-medium ${riskColors[myChild.risk]}`}>
            {myChild.risk} Risk
          </span>
        </div>
        
        <div className="flex items-center gap-4 mb-4">
          <div className="w-16 h-16 rounded-full bg-primary flex items-center justify-center text-white text-2xl font-bold">
            {myChild.name.charAt(0)}
          </div>
          <div>
            <h3 className="text-xl font-bold text-gray-900 dark:text-white">{myChild.name}</h3>
            <p className="text-gray-500 dark:text-gray-400">Age: {myChild.age} years</p>
          </div>
        </div>

        <div className="border-t border-gray-100 dark:border-dark-deep pt-4">
          <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Assigned Specialist</p>
          <div className="flex items-center justify-between">
            <p className="text-gray-900 dark:text-white">{myChild.specialist}</p>
            <div className="flex gap-2">
              <button 
                onClick={() => setShowContact(!showContact)}
                className="px-3 py-1.5 bg-primary hover:bg-primary-dark text-white text-sm rounded-lg"
              >
                Contact
              </button>
            </div>
          </div>
        </div>

        {/* Contact Options */}
        {showContact && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            className="mt-4 pt-4 border-t border-gray-100 dark:border-dark-deep space-y-2"
          >
            <a href={`tel:${myChild.specialistPhone}`} className="flex items-center gap-3 p-3 bg-gray-50 dark:bg-dark-bg rounded-lg text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-dark-deep">
              <Phone className="w-5 h-5 text-primary" />
              {myChild.specialistPhone}
            </a>
            <a href={`mailto:${myChild.specialistEmail}`} className="flex items-center gap-3 p-3 bg-gray-50 dark:bg-dark-bg rounded-lg text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-dark-deep">
              <Mail className="w-5 h-5 text-primary" />
              {myChild.specialistEmail}
            </a>
            <a href="/parent/messages" className="flex items-center gap-3 p-3 bg-gray-50 dark:bg-dark-bg rounded-lg text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-dark-deep">
              <Send className="w-5 h-5 text-primary" />
              Send Message
            </a>
          </motion.div>
        )}
      </motion.div>

      {/* Disclaimer */}
      <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-xl flex items-start gap-3">
        <AlertCircle className="w-5 h-5 text-yellow-600 dark:text-yellow-400 flex-shrink-0 mt-0.5" />
        <p className="text-sm text-yellow-800 dark:text-yellow-200">
          <strong>Important:</strong> This is a research demo and NOT a diagnostic tool. Results should not be used for clinical decision-making. 
          Consult {myChild.specialist} for professional medical advice.
        </p>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-2 gap-4">
        <a href="/parent/results" className="flex items-center gap-3 p-4 bg-white dark:bg-dark-surface rounded-xl border border-gray-200 dark:border-dark-deep hover:border-primary transition-colors">
          <FileText className="w-6 h-6 text-primary" />
          <span className="font-medium text-gray-900 dark:text-white">View Results</span>
        </a>
        <a href="/parent/messages" className="flex items-center gap-3 p-4 bg-white dark:bg-dark-surface rounded-xl border border-gray-200 dark:border-dark-deep hover:border-primary transition-colors">
          <MessageSquare className="w-6 h-6 text-primary" />
          <span className="font-medium text-gray-900 dark:text-white">Message Specialist</span>
        </a>
      </div>
    </div>
  );
}