'use client';

import { FileText, Calendar, TrendingUp, AlertCircle } from 'lucide-react';
import { motion } from 'framer-motion';

const results = [
  { id: 1, child: 'Emma', date: '2026-05-01', score: 0.82, risk: 'High', models: { rf: 0.78, svm: 0.85, lstm: 0.80, transformer: 0.84 } },
  { id: 2, child: 'Emma', date: '2026-03-15', score: 0.65, risk: 'Moderate', models: { rf: 0.62, svm: 0.68, lstm: 0.63, transformer: 0.67 } },
  { id: 3, child: 'Liam', date: '2026-04-01', score: 0.23, risk: 'Low', models: { rf: 0.21, svm: 0.25, lstm: 0.22, transformer: 0.24 } },
];

const riskColors = {
  High: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400',
  Moderate: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400',
  Low: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
};

export default function ParentResultsPage() {
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

      {/* Results Cards */}
      <div className="space-y-4">
        {results.map((result, index) => (
          <motion.div
            key={result.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="p-6 bg-white dark:bg-dark-surface rounded-2xl border border-gray-200 dark:border-dark-deep"
          >
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-4">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-full bg-primary-light flex items-center justify-center">
                  <FileText className="w-6 h-6 text-primary" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white">{result.child}</h3>
                  <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400">
                    <Calendar className="w-4 h-4" />
                    {result.date}
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-4">
                <div className="text-right">
                  <p className="text-3xl font-bold text-gray-900 dark:text-white">{(result.score * 100).toFixed(0)}%</p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">ASD Probability</p>
                </div>
                <span className={`px-4 py-2 rounded-full text-lg font-semibold ${riskColors[result.risk as keyof typeof riskColors]}`}>
                  {result.risk} Risk
                </span>
              </div>
            </div>

            {/* Model Breakdown */}
            <div className="pt-4 border-t border-gray-100 dark:border-dark-deep">
              <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Model Predictions:</p>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {Object.entries(result.models).map(([model, score]) => (
                  <div key={model} className="p-3 bg-gray-50 dark:bg-dark-bg rounded-lg">
                    <p className="text-xs text-gray-500 dark:text-gray-400 uppercase">{model}</p>
                    <p className="text-lg font-semibold text-gray-900 dark:text-white">{(score * 100).toFixed(0)}%</p>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}