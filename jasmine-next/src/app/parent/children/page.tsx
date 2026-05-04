'use client';

import { Baby, Calendar, Phone, Mail, FileText, Plus } from 'lucide-react';
import { useState } from 'react';
import { motion } from 'framer-motion';

const children = [
  { id: 1, name: 'Emma', age: 6, dob: '2019-03-15', parent: 'John Thompson', phone: '+1 555-0123', email: 'john@email.com', assessments: 3, lastAssessment: '2026-05-01' },
  { id: 2, name: 'Liam', age: 4, dob: '2021-07-22', parent: 'John Thompson', phone: '+1 555-0123', email: 'john@email.com', assessments: 1, lastAssessment: '2026-04-01' },
];

export default function ParentChildrenPage() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">My Children</h1>
          <p className="text-gray-500 dark:text-gray-400">Manage your children&apos;s profiles</p>
        </div>
      </div>

      {/* Children Cards */}
      <div className="grid md:grid-cols-2 gap-6">
        {children.map((child, index) => (
          <motion.div
            key={child.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="p-6 bg-white dark:bg-dark-surface rounded-2xl border border-gray-200 dark:border-dark-deep"
          >
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-4">
                <div className="w-16 h-16 rounded-full bg-primary-light flex items-center justify-center">
                  <Baby className="w-8 h-8 text-primary" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white">{child.name}</h3>
                  <p className="text-gray-500 dark:text-gray-400">Age: {child.age} years</p>
                </div>
              </div>
            </div>

            <div className="space-y-3 text-sm">
              <div className="flex items-center gap-3 text-gray-600 dark:text-gray-300">
                <Calendar className="w-4 h-4 text-gray-400" />
                <span>Born: {child.dob}</span>
              </div>
              <div className="flex items-center gap-3 text-gray-600 dark:text-gray-300">
                <FileText className="w-4 h-4 text-gray-400" />
                <span>{child.assessments} assessment{child.assessments !== 1 ? 's' : ''} completed</span>
              </div>
              <div className="flex items-center gap-3 text-gray-600 dark:text-gray-300">
                <Phone className="w-4 h-4 text-gray-400" />
                <span>Last: {child.lastAssessment}</span>
              </div>
            </div>

            <div className="mt-4 pt-4 border-t border-gray-100 dark:border-dark-deep flex gap-2">
              <button className="flex-1 px-4 py-2 bg-primary-light/50 dark:bg-primary-dark/20 text-primary dark:text-primary-light rounded-lg font-medium hover:bg-primary-light transition-colors">
                View Results
              </button>
              <button className="flex-1 px-4 py-2 bg-gray-100 dark:bg-dark-deep text-gray-700 dark:text-gray-300 rounded-lg font-medium hover:bg-gray-200 dark:hover:bg-dark-deep transition-colors">
                Details
              </button>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}