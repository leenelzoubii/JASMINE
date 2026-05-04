'use client';

import { useState } from 'react';
import { Upload, FileText, Clock, CheckCircle, AlertCircle, RefreshCw, Download } from 'lucide-react';
import { motion } from 'framer-motion';

const assessments = [
  { id: 1, patient: 'Emma Thompson', date: '2026-05-01', status: 'Completed', score: 0.82, risk: 'High' },
  { id: 2, patient: 'Liam Johnson', date: '2026-04-28', status: 'Pending', score: null, risk: null },
  { id: 3, patient: 'Sophie Williams', date: '2026-04-25', status: 'Completed', score: 0.23, risk: 'Low' },
  { id: 4, patient: 'James Brown', date: '2026-04-20', status: 'Completed', score: 0.45, risk: 'Moderate' },
];

const statusColors = {
  Completed: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
  Pending: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400',
};

const riskColors = {
  High: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400',
  Moderate: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400',
  Low: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
};

export default function ProfessionalAssessmentsPage() {
  const [selectedPatient, setSelectedPatient] = useState('');
  const [uploading, setUploading] = useState(false);

  const handleUpload = () => {
    setUploading(true);
    setTimeout(() => setUploading(false), 2000);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Assessments</h1>
        <p className="text-gray-500 dark:text-gray-400">Run and manage screenings</p>
      </div>

      {/* New Assessment Card */}
      <div className="p-6 bg-white dark:bg-dark-surface rounded-2xl border-2 border-dashed border-gray-200 dark:border-dark-deep">
        <div className="flex flex-col items-center text-center">
          <div className="w-16 h-16 rounded-full bg-primary-light/50 dark:bg-primary-dark/20 flex items-center justify-center mb-4">
            <Upload className="w-8 h-8 text-primary" />
          </div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">Run New Assessment</h3>
          <p className="text-gray-500 dark:text-gray-400 mb-4">Select a patient and upload pose data (CSV/JSON)</p>
          
          <div className="flex gap-3 w-full max-w-md">
            <select 
              value={selectedPatient}
              onChange={(e) => setSelectedPatient(e.target.value)}
              className="flex-1 px-4 py-3 bg-gray-50 dark:bg-dark-bg border border-gray-200 dark:border-dark-deep rounded-xl focus:outline-none focus:ring-2 focus:ring-primary"
            >
              <option value="">Select patient...</option>
              <option value="1">Emma Thompson</option>
              <option value="2">Liam Johnson</option>
              <option value="3">Sophie Williams</option>
            </select>
            <button 
              onClick={handleUpload}
              disabled={!selectedPatient || uploading}
              className="px-6 py-3 bg-primary hover:bg-primary-dark text-white font-medium rounded-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {uploading ? <RefreshCw className="w-5 h-5 animate-spin" /> : <Upload className="w-5 h-5" />}
              {uploading ? 'Processing...' : 'Run'}
            </button>
          </div>
        </div>
      </div>

      {/* Recent Assessments */}
      <div className="bg-white dark:bg-dark-surface rounded-2xl border border-gray-200 dark:border-dark-deep overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-100 dark:border-dark-deep">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Recent Assessments</h2>
        </div>
        <div className="divide-y divide-gray-100 dark:divide-dark-deep">
          {assessments.map((assessment, index) => (
            <motion.div
              key={assessment.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="px-6 py-4 flex items-center justify-between hover:bg-gray-50 dark:hover:bg-dark-deep transition-colors"
            >
              <div className="flex items-center gap-4">
                <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                  assessment.status === 'Completed' ? 'bg-green-100 dark:bg-green-900/30' : 'bg-yellow-100 dark:bg-yellow-900/30'
                }`}>
                  {assessment.status === 'Completed' ? (
                    <CheckCircle className="w-5 h-5 text-green-600" />
                  ) : (
                    <Clock className="w-5 h-5 text-yellow-600" />
                  )}
                </div>
                <div>
                  <p className="font-medium text-gray-900 dark:text-white">{assessment.patient}</p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">{assessment.date}</p>
                </div>
              </div>
              <div className="flex items-center gap-4">
                {assessment.score !== null && (
                  <div className="text-right">
                    <p className="text-lg font-semibold text-gray-900 dark:text-white">{(assessment.score * 100).toFixed(0)}%</p>
                    <p className={`text-sm font-medium ${riskColors[assessment.risk as keyof typeof riskColors]}`}>
                      {assessment.risk} Risk
                    </p>
                  </div>
                )}
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${statusColors[assessment.status as keyof typeof statusColors]}`}>
                  {assessment.status}
                </span>
                {assessment.status === 'Completed' && (
                  <button className="p-2 text-gray-400 hover:text-primary rounded-lg hover:bg-gray-100 dark:hover:bg-dark-deep">
                    <Download className="w-5 h-5" />
                  </button>
                )}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}