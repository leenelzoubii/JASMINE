'use client';

import { useState, useEffect } from 'react';
import {
  User,
  Mail,
  Building2,
  Phone,
  Save,
  Clock,
  Lock,
  Eye,
  EyeOff,
  CheckCircle,
  XCircle,
} from 'lucide-react';
import { motion } from 'framer-motion';
import { getCurrentUser, updateUser, changeCurrentUserPassword } from '@/lib/auth';

export default function ProfessionalProfilePage() {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [specialty, setSpecialty] = useState('');
  const [phone, setPhone] = useState('');
  const [saving, setSaving] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [saved, setSaved] = useState(false);
  const [phoneError, setPhoneError] = useState('');
  const [showChangePassword, setShowChangePassword] = useState(false);
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showCurrentPassword, setShowCurrentPassword] = useState(false);
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [passwordError, setPasswordError] = useState('');
  const [passwordSuccess, setPasswordSuccess] = useState(false);
  const [changingPassword, setChangingPassword] = useState(false);

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

  const handleChangePassword = async () => {
    const user = getCurrentUser();
    if (!user) return;

    setPasswordError('');

    if (!currentPassword || !newPassword || !confirmPassword) {
      setPasswordError('All password fields are required.');
      return;
    }

    if (newPassword.length < 8) {
      setPasswordError('New password must be at least 8 characters.');
      return;
    }

    if (newPassword !== confirmPassword) {
      setPasswordError('New passwords do not match.');
      return;
    }

    setChangingPassword(true);

    try {
      await changeCurrentUserPassword(currentPassword, newPassword);

      setPasswordSuccess(true);
      setCurrentPassword('');
      setNewPassword('');
      setConfirmPassword('');

      setTimeout(() => {
        setPasswordSuccess(false);
        setShowChangePassword(false);
      }, 2000);
    } catch (err) {
      setPasswordError('Failed to change password. Please check your current password.');
      console.error(err);
    } finally {
      setChangingPassword(false);
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
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
              {name || 'Loading...'}
            </h2>
            <p className="text-gray-500 dark:text-gray-400">
              {specialty || 'Professional'}
            </p>
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
                onChange={(e) => {
                  setPhone(e.target.value);
                  setPhoneError('');
                }}
                placeholder="e.g., +1 555-0123"
                className={`w-full pl-12 pr-4 py-3 bg-gray-50 dark:bg-dark-bg border rounded-xl focus:outline-none focus:ring-2 focus:ring-primary ${
                  phoneError ? 'border-red-500' : 'border-gray-200 dark:border-dark-deep'
                }`}
              />
            </div>

            {phoneError && <p className="mt-1 text-sm text-red-500">{phoneError}</p>}
          </div>
        </div>

        <div className="pt-4 flex justify-end gap-3">
          {saved && (
            <span className="text-green-600 dark:text-green-400 text-sm flex items-center">
              Saved!
            </span>
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
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Account Information
        </h3>

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

            <span className="text-gray-900 dark:text-white font-medium">
              {specialty || 'Not set'}
            </span>
          </div>
        </div>
      </div>

      {/* Change Password Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="p-6 bg-white dark:bg-dark-surface rounded-2xl border border-gray-200 dark:border-dark-deep space-y-5"
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center">
              <Lock className="w-5 h-5 text-primary" />
            </div>

            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                Change Password
              </h3>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Update your account password
              </p>
            </div>
          </div>

          <button
            onClick={() => setShowChangePassword(!showChangePassword)}
            className="px-4 py-2 text-sm text-primary hover:bg-primary/10 rounded-lg transition-colors"
          >
            {showChangePassword ? 'Cancel' : 'Change'}
          </button>
        </div>

        {showChangePassword && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            className="space-y-4 pt-4 border-t border-gray-100 dark:border-dark-deep"
          >
            {passwordError && (
              <div className="p-3 rounded-xl bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 text-sm flex items-center gap-2">
                <XCircle className="w-4 h-4 shrink-0" />
                {passwordError}
              </div>
            )}

            {passwordSuccess && (
              <div className="p-3 rounded-xl bg-green-50 dark:bg-green-900/20 text-green-600 dark:text-green-400 text-sm flex items-center gap-2">
                <CheckCircle className="w-4 h-4 shrink-0" />
                Password changed successfully!
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Current Password
              </label>

              <div className="relative">
                <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type={showCurrentPassword ? 'text' : 'password'}
                  value={currentPassword}
                  onChange={(e) => setCurrentPassword(e.target.value)}
                  className="w-full pl-12 pr-12 py-3 bg-gray-50 dark:bg-dark-bg border border-gray-200 dark:border-dark-deep rounded-xl focus:outline-none focus:ring-2 focus:ring-primary"
                />

                <button
                  type="button"
                  onClick={() => setShowCurrentPassword(!showCurrentPassword)}
                  className="absolute right-4 top-1/2 -translate-y-1/2 p-1.5 bg-gray-100 dark:bg-gray-700 rounded-md text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 transition-colors"
                >
                  {showCurrentPassword ? (
                    <EyeOff className="w-4 h-4" />
                  ) : (
                    <Eye className="w-4 h-4" />
                  )}
                </button>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                New Password
              </label>

              <div className="relative">
                <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type={showNewPassword ? 'text' : 'password'}
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  className="w-full pl-12 pr-12 py-3 bg-gray-50 dark:bg-dark-bg border border-gray-200 dark:border-dark-deep rounded-xl focus:outline-none focus:ring-2 focus:ring-primary"
                />

                <button
                  type="button"
                  onClick={() => setShowNewPassword(!showNewPassword)}
                  className="absolute right-4 top-1/2 -translate-y-1/2 p-1.5 bg-gray-100 dark:bg-gray-700 rounded-md text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 transition-colors"
                >
                  {showNewPassword ? (
                    <EyeOff className="w-4 h-4" />
                  ) : (
                    <Eye className="w-4 h-4" />
                  )}
                </button>
              </div>

              <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                Must be at least 8 characters
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Confirm New Password
              </label>

              <div className="relative">
                <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type={showNewPassword ? 'text' : 'password'}
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  className="w-full pl-12 pr-4 py-3 bg-gray-50 dark:bg-dark-bg border border-gray-200 dark:border-dark-deep rounded-xl focus:outline-none focus:ring-2 focus:ring-primary"
                />
              </div>
            </div>

            <div className="flex justify-end pt-2">
              <button
                onClick={handleChangePassword}
                disabled={changingPassword}
                className="px-6 py-3 bg-primary hover:bg-primary-dark text-white font-medium rounded-xl transition-all flex items-center gap-2 disabled:opacity-50"
              >
                {changingPassword ? (
                  <>
                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Changing...
                  </>
                ) : (
                  <>
                    <Lock className="w-5 h-5" />
                    Change Password
                  </>
                )}
              </button>
            </div>
          </motion.div>
        )}
      </motion.div>
    </div>
  );
}