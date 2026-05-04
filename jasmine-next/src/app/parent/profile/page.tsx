'use client';

import { useState } from 'react';
import { User, Mail, Phone, Save, Camera, Clock } from 'lucide-react';
import { motion } from 'framer-motion';

export default function ParentProfilePage() {
  const [name, setName] = useState('John Parent');
  const [email, setEmail] = useState('john@email.com');
  const [phone, setPhone] = useState('+1 555-0123');
  const [saving, setSaving] = useState(false);

  const handleSave = () => {
    setSaving(true);
    setTimeout(() => setSaving(false), 1500);
  };

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Profile</h1>
        <p className="text-gray-500 dark:text-gray-400">Manage your account settings</p>
      </div>

      {/* Avatar Section */}
      <div className="p-6 bg-white dark:bg-dark-surface rounded-2xl border border-gray-200 dark:border-dark-deep">
        <div className="flex items-center gap-6">
          <div className="relative">
            <div className="w-24 h-24 rounded-full bg-primary flex items-center justify-center text-white text-3xl font-bold">
              JP
            </div>
            <button className="absolute bottom-0 right-0 w-8 h-8 rounded-full bg-primary-dark flex items-center justify-center text-white border-2 border-white dark:border-dark-surface">
              <Camera className="w-4 h-4" />
            </button>
          </div>
          <div>
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white">{name}</h2>
            <p className="text-gray-500 dark:text-gray-400">Parent/Guardian</p>
            <p className="text-sm text-gray-400 dark:text-gray-500 mt-1">Member since January 2026</p>
          </div>
        </div>
      </div>

      {/* Profile Form */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="p-6 bg-white dark:bg-dark-surface rounded-2xl border border-gray-200 dark:border-dark-deep space-y-5"
      >
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Full Name
          </label>
          <div className="relative">
            <User className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full pl-12 pr-4 py-3 bg-gray-50 dark:bg-dark-bg border border-gray-200 dark:border-dark-deep rounded-xl focus:outline-none focus:ring-2 focus:ring-primary"
            />
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Email
          </label>
          <div className="relative">
            <Mail className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full pl-12 pr-4 py-3 bg-gray-50 dark:bg-dark-bg border border-gray-200 dark:border-dark-deep rounded-xl focus:outline-none focus:ring-2 focus:ring-primary"
            />
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Phone
          </label>
          <div className="relative">
            <Phone className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="tel"
              value={phone}
              onChange={(e) => setPhone(e.target.value)}
              className="w-full pl-12 pr-4 py-3 bg-gray-50 dark:bg-dark-bg border border-gray-200 dark:border-dark-deep rounded-xl focus:outline-none focus:ring-2 focus:ring-primary"
            />
          </div>
        </div>

        <div className="pt-4 flex justify-end">
          <button
            onClick={handleSave}
            disabled={saving}
            className="px-6 py-3 bg-primary hover:bg-primary-dark text-white font-medium rounded-xl transition-all flex items-center gap-2"
          >
            {saving ? (
              <>
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Saving...
              </>
            ) : (
              <>
                <Save className="w-5 h-5" />
                Save Changes
              </>
            )}
          </button>
        </div>
      </motion.div>

      {/* Account Info */}
      <div className="p-6 bg-white dark:bg-dark-surface rounded-2xl border border-gray-200 dark:border-dark-deep">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Account Information</h3>
        <div className="space-y-4">
          <div className="flex items-center justify-between py-3 border-b border-gray-100 dark:border-dark-deep">
            <div className="flex items-center gap-3">
              <Clock className="w-5 h-5 text-gray-400" />
              <span className="text-gray-700 dark:text-gray-300">Member since</span>
            </div>
            <span className="text-gray-900 dark:text-white font-medium">January 2026</span>
          </div>
          <div className="flex items-center justify-between py-3">
            <div className="flex items-center gap-3">
              <User className="w-5 h-5 text-gray-400" />
              <span className="text-gray-700 dark:text-gray-300">Account type</span>
            </div>
            <span className="text-primary font-medium">Parent/Guardian</span>
          </div>
        </div>
      </div>
    </div>
  );
}