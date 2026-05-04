'use client';

import { UserPlus, Search, MoreVertical, Phone, Mail, Calendar } from 'lucide-react';
import { useState } from 'react';
import { motion } from 'framer-motion';

const patients = [
  { id: 1, name: 'Emma Thompson', age: 6, parent: 'John Thompson', phone: '+1 555-0123', email: 'john@email.com', lastVisit: '2026-05-01', risk: 'High' },
  { id: 2, name: 'Liam Johnson', age: 5, parent: 'Sarah Johnson', phone: '+1 555-0124', email: 'sarah@email.com', lastVisit: '2026-04-25', risk: 'Moderate' },
  { id: 3, name: 'Sophie Williams', age: 7, parent: 'Mike Williams', phone: '+1 555-0125', email: 'mike@email.com', lastVisit: '2026-04-20', risk: 'Low' },
  { id: 4, name: 'James Brown', age: 4, parent: 'Lisa Brown', phone: '+1 555-0126', email: 'lisa@email.com', lastVisit: '2026-04-15', risk: 'Low' },
  { id: 5, name: 'Olivia Davis', age: 6, parent: 'Tom Davis', phone: '+1 555-0127', email: 'tom@email.com', lastVisit: '2026-04-10', risk: 'Moderate' },
];

const riskColors = {
  High: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400',
  Moderate: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400',
  Low: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
};

export default function ProfessionalPatientsPage() {
  const [search, setSearch] = useState('');
  const [selectedPatient, setSelectedPatient] = useState<number | null>(null);

  const filteredPatients = patients.filter(p => 
    p.name.toLowerCase().includes(search.toLowerCase()) ||
    p.parent.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Patients</h1>
          <p className="text-gray-500 dark:text-gray-400">Manage your patients</p>
        </div>
        <button className="inline-flex items-center gap-2 px-4 py-2.5 bg-primary hover:bg-primary-dark text-white font-medium rounded-xl transition-all">
          <UserPlus className="w-5 h-5" />
          Add Patient
        </button>
      </div>

      {/* Search */}
      <div className="relative max-w-md">
        <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
        <input
          type="text"
          placeholder="Search patients..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full pl-12 pr-4 py-3 bg-white dark:bg-dark-surface border border-gray-200 dark:border-dark-deep rounded-xl focus:outline-none focus:ring-2 focus:ring-primary"
        />
      </div>

      {/* Patients Table */}
      <div className="bg-white dark:bg-dark-surface rounded-2xl border border-gray-200 dark:border-dark-deep overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50 dark:bg-dark-deep">
              <tr>
                <th className="px-6 py-4 text-left text-sm font-semibold text-gray-900 dark:text-white">Name</th>
                <th className="px-6 py-4 text-left text-sm font-semibold text-gray-900 dark:text-white">Age</th>
                <th className="px-6 py-4 text-left text-sm font-semibold text-gray-900 dark:text-white">Parent</th>
                <th className="px-6 py-4 text-left text-sm font-semibold text-gray-900 dark:text-white">Contact</th>
                <th className="px-6 py-4 text-left text-sm font-semibold text-gray-900 dark:text-white">Last Visit</th>
                <th className="px-6 py-4 text-left text-sm font-semibold text-gray-900 dark:text-white">Risk</th>
                <th className="px-6 py-4"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100 dark:divide-dark-deep">
              {filteredPatients.map((patient) => (
                <motion.tr
                  key={patient.id}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="hover:bg-gray-50 dark:hover:bg-dark-deep transition-colors"
                >
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-full bg-primary-light flex items-center justify-center text-primary font-semibold">
                        {patient.name.charAt(0)}
                      </div>
                      <span className="font-medium text-gray-900 dark:text-white">{patient.name}</span>
                    </div>
                  </td>
                  <td className="px-6 py-4 text-gray-600 dark:text-gray-300">{patient.age}</td>
                  <td className="px-6 py-4 text-gray-600 dark:text-gray-300">{patient.parent}</td>
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-2">
                      <button className="p-2 text-gray-400 hover:text-primary rounded-lg hover:bg-gray-100 dark:hover:bg-dark-deep">
                        <Phone className="w-4 h-4" />
                      </button>
                      <button className="p-2 text-gray-400 hover:text-primary rounded-lg hover:bg-gray-100 dark:hover:bg-dark-deep">
                        <Mail className="w-4 h-4" />
                      </button>
                    </div>
                  </td>
                  <td className="px-6 py-4 text-gray-600 dark:text-gray-300">{patient.lastVisit}</td>
                  <td className="px-6 py-4">
                    <span className={`px-3 py-1 rounded-full text-sm font-medium ${riskColors[patient.risk as keyof typeof riskColors]}`}>
                      {patient.risk}
                    </span>
                  </td>
                  <td className="px-6 py-4">
                    <button className="p-2 text-gray-400 hover:text-gray-600 rounded-lg hover:bg-gray-100 dark:hover:bg-dark-deep">
                      <MoreVertical className="w-4 h-4" />
                    </button>
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}