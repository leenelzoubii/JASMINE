'use client';

import { useState, useEffect } from 'react';
import { User, Mail, Building2, Phone, Save, Camera, Clock } from 'lucide-react';
import { motion } from 'framer-motion';
import { getCurrentUser, updateUser } from '@/lib/auth';

export default function ProfessionalProfilePage() {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [specialty, setSpecialty] = useState('');
  const [phone, setPhone] = useState('');
  const [saving, setSaving] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [saved, setSaved] = useState(false);
  const [phoneError, setPhoneError] = useState('');

  useEffect(() => {
    setMounted(true);
    const user = getCurrentUser();
    if (user) {
      setName(user.name);
      setEmail(user.email);
      setSpecialty(user.specialty || '');
      setPhone(user.phone || '');
    }
  }, []);

  const handleSave = async () => {
    const user = getCurrentUser();
    if (!user) return;

    if (phone && !/^[+]?[\d\s\-()]*$/.test(phone)) {
      setPhoneError('Please enter a valid phone number');
      return;
    }
    setPhoneError('');

    setSaving(true);
    setSaved(false);
    try {
      await updateUser(user.id, { name, specialty, phone });
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
    } catch (err) {
      console.error(err);
    } finally {
      setSaving(false);
    }
  };

  if (!mounted) {
    return (
      <div className="max-w-2xl mx-auto space-y-6">
        <div className="p-6 bg-white dark:bg-dark-surface rounded-2xl border border-gray-200 dark:border-dark-deep animate-pulse h-32" />
      </div>
    );
  }

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
              {name ? name.charAt(0).toUpperCase() : '?'}
            </div>
          </div>
          <div>
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white">{name || 'Loading...'}</h2>
            <p className="text-gray-500 dark:text-gray-400">{specialty || 'Professional'}</p>
          </div>
        </div>
      </div>

      {/* Profile Form */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="p-6 bg-white dark:bg-dark-surface rounded-2xl border border-gray-200 dark:border-dark-deep space-y-5"
      >
        <div className="grid sm:grid-cols-2 gap-5">
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
                readOnly
                className="w-full pl-12 pr-4 py-3 bg-gray-100 dark:bg-dark-deep border border-gray-200 dark:border-dark-deep rounded-xl cursor-not-allowed opacity-60"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Specialty
            </label>
            <div className="relative">
              <Building2 className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                value={specialty}
                onChange={(e) => setSpecialty(e.target.value)}
                placeholder="e.g., Pediatric Neurology"
                className="w-full pl-12 pr-4 py-3 bg-gray-50 dark:bg-dark-bg border border-gray-200 dark:border-dark-deep rounded-xl focus:outline-none focus:ring-2 focus:ring-primary"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Phone <span className="text-gray-400 font-normal">(optional)</span>
            </label>
            <div className="relative">
              <Phone className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="tel"
                value={phone}
                onChange={(e) => { setPhone(e.target.value); setPhoneError(''); }}
                placeholder="e.g., +1 555-0123"
                className={`w-full pl-12 pr-4 py-3 bg-gray-50 dark:bg-dark-bg border rounded-xl focus:outline-none focus:ring-2 focus:ring-primary ${phoneError ? 'border-red-500' : 'border-gray-200 dark:border-dark-deep'}`}
              />
            </div>
            {phoneError && <p className="mt-1 text-sm text-red-500">{phoneError}</p>}
          </div>
        </div>

        <div className="pt-4 flex justify-end gap-3">
          {saved && (
            <span className="text-green-600 dark:text-green-400 text-sm flex items-center">Saved!</span>
          )}
          <button
            onClick={handleSave}
            disabled={saving}
            className="px-6 py-3 bg-primary hover:bg-primary-dark text-white font-medium rounded-xl transition-all flex items-center gap-2 disabled:opacity-50"
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
              <span className="text-gray-700 dark:text-gray-300">Account type</span>
            </div>
            <span className="text-primary font-medium">Professional</span>
          </div>
          <div className="flex items-center justify-between py-3">
            <div className="flex items-center gap-3">
              <Building2 className="w-5 h-5 text-gray-400" />
              <span className="text-gray-700 dark:text-gray-300">Specialty</span>
            </div>
            <span className="text-gray-900 dark:text-white font-medium">{specialty || 'Not set'}</span>
          </div>
        </div>
      </div>
    </div>
  );
}