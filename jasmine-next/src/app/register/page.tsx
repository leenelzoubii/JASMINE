'use client';

import Link from 'next/link';
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Brain, Mail, Lock, User, Building2, AlertCircle, Check, ArrowLeft, Sparkles, Shield, Activity } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { registerUser } from "@/lib/auth";

const roles = [
  { id: 'professional', title: 'Healthcare Professional', icon: '🏥', description: 'Doctors, therapists, and specialists' },
  { id: 'parent', title: 'Parent/Guardian', icon: '👨‍👩‍👧', description: 'Parents monitoring their child' },
];

export default function RegisterPage() {
  const [step, setStep] = useState(1);
  const [selectedRole, setSelectedRole] = useState('');
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [specialty, setSpecialty] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (step === 1) {
      if (!selectedRole) {
        setError("Please select an account type");
        return;
      }
      setError('');
      setStep(2);
      return;
    }

    if (!name || !email || !password) {
      setError("Please fill in all required fields");
      return;
    }

    setLoading(true);

    try {
      const user = await registerUser(
        name, email, password,
        selectedRole as "parent" | "professional",
        selectedRole === 'professional' ? specialty : undefined
      );
      router.push(user.role === "parent" ? "/parent" : "/professional");
    } catch (error) {
      const errMsg = error instanceof Error ? error.message : "Registration failed.";
      setError(errMsg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex">
      {/* Left - Brand Panel */}
      <div className="hidden lg:flex lg:w-1/2 relative overflow-hidden items-center justify-center p-12"
        style={{ background: 'var(--gradient-hero)' }}>
        <div className="absolute inset-0 opacity-[0.04]">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 rounded-full bg-blue-400 blur-[120px]" />
          <div className="absolute bottom-1/4 right-1/4 w-80 h-80 rounded-full bg-emerald-400 blur-[100px]" />
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative z-10 text-center max-w-md"
        >
          <motion.div
            animate={{ rotate: [0, -5, 0] }}
            transition={{ repeat: Infinity, duration: 6, ease: 'easeInOut' }}
            className="w-20 h-20 rounded-2xl flex items-center justify-center mx-auto mb-8"
            style={{ background: 'var(--gradient-primary)' }}
          >
            <Brain className="w-10 h-10 text-white" />
          </motion.div>
          <h2 className="text-4xl font-bold mb-4" style={{ color: 'var(--text-primary)' }}>Get Started</h2>
          <p className="text-lg mb-8" style={{ color: 'var(--text-dim)' }}>
            Join the JASMINE platform for early autism screening
          </p>
          <div className="space-y-4 text-left">
            {[
              { icon: Sparkles, text: 'Multi-model ensemble predictions' },
              { icon: Shield, text: 'End-to-end privacy protection' },
              { icon: Activity, text: 'Real-time video analysis pipeline' },
            ].map((item, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.3 + i * 0.1 }}
                className="flex items-center gap-3"
              >
                <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: 'var(--primary-light)' }}>
                  <item.icon className="w-4 h-4" style={{ color: 'var(--primary)' }} />
                </div>
                <span style={{ color: 'var(--text-dim)' }}>{item.text}</span>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Right - Form */}
      <div className="flex-1 flex items-center justify-center px-4 py-12" style={{ backgroundColor: 'var(--background-alt)' }}>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="w-full max-w-md"
        >
          <div className="premium-card p-8">
            <div className="text-center mb-8">
              <Link href="/" className="inline-flex items-center gap-2 mb-4">
                <motion.div
                  whileHover={{ scale: 1.05, rotate: -3 }}
                  className="w-12 h-12 rounded-xl flex items-center justify-center"
                  style={{ background: 'var(--gradient-primary)' }}
                >
                  <Brain className="w-7 h-7 text-white" />
                </motion.div>
              </Link>
              <h1 className="text-2xl font-bold" style={{ color: 'var(--foreground)' }}>
                {step === 1 ? 'Create Account' : 'Complete Registration'}
              </h1>
              <p style={{ color: 'var(--text-muted)' }}>
                {step === 1 ? 'Choose your account type' : 'Fill in your details'}
              </p>
            </div>

            {error && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mb-6 p-4 rounded-xl flex items-start gap-3"
                style={{ backgroundColor: 'var(--risk-high-bg)', border: '1px solid var(--risk-high)' }}
              >
                <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" style={{ color: 'var(--risk-high)' }} />
                <span className="text-sm" style={{ color: 'var(--risk-high)' }}>{error}</span>
              </motion.div>
            )}

            <AnimatePresence mode="wait">
              {step === 1 ? (
                <motion.div
                  key="step1"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  className="space-y-4"
                >
                  {roles.map((role) => (
                    <motion.button
                      key={role.id}
                      onClick={() => setSelectedRole(role.id)}
                      whileHover={{ scale: 1.01 }}
                      whileTap={{ scale: 0.99 }}
                      className="w-full p-5 rounded-xl border-2 transition-all text-left"
                      style={{
                        borderColor: selectedRole === role.id ? 'var(--primary)' : 'var(--border)',
                        backgroundColor: selectedRole === role.id ? 'var(--primary-light)' : 'var(--background)',
                      }}
                    >
                      <div className="flex items-center gap-4">
                        <span className="text-3xl">{role.icon}</span>
                        <div className="flex-1">
                          <h3 className="font-semibold" style={{ color: 'var(--foreground)' }}>{role.title}</h3>
                          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>{role.description}</p>
                        </div>
                        {selectedRole === role.id && (
                          <motion.div
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                            className="w-6 h-6 rounded-full flex items-center justify-center"
                            style={{ backgroundColor: 'var(--primary)' }}
                          >
                            <Check className="w-4 h-4 text-white" />
                          </motion.div>
                        )}
                      </div>
                    </motion.button>
                  ))}
                </motion.div>
              ) : (
                <motion.div
                  key="step2"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                >
                  <form onSubmit={handleSubmit} className="space-y-5">
                    <div>
                      <label className="block text-sm font-medium mb-1.5" style={{ color: 'var(--foreground)' }}>
                        Full Name
                      </label>
                      <div className="relative">
                        <User className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4" style={{ color: 'var(--text-dim)' }} />
                        <input
                          type="text"
                          value={name}
                          onChange={(e) => setName(e.target.value)}
                          placeholder="Your name"
                          className="premium-input pl-11"
                        />
                      </div>
                    </div>

                    <div>
                      <label className="block text-sm font-medium mb-1.5" style={{ color: 'var(--foreground)' }}>
                        Email
                      </label>
                      <div className="relative">
                        <Mail className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4" style={{ color: 'var(--text-dim)' }} />
                        <input
                          type="email"
                          value={email}
                          onChange={(e) => setEmail(e.target.value)}
                          placeholder="you@example.com"
                          className="premium-input pl-11"
                        />
                      </div>
                    </div>

                    {selectedRole === 'professional' && (
                      <div>
                        <label className="block text-sm font-medium mb-1.5" style={{ color: 'var(--foreground)' }}>
                          Specialty (Optional)
                        </label>
                        <div className="relative">
                          <Building2 className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4" style={{ color: 'var(--text-dim)' }} />
                          <input
                            type="text"
                            value={specialty}
                            onChange={(e) => setSpecialty(e.target.value)}
                            placeholder="e.g., Pediatric Neurology"
                            className="premium-input pl-11"
                          />
                        </div>
                      </div>
                    )}

                    <div>
                      <label className="block text-sm font-medium mb-1.5" style={{ color: 'var(--foreground)' }}>
                        Password
                      </label>
                      <div className="relative">
                        <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4" style={{ color: 'var(--text-dim)' }} />
                        <input
                          type="password"
                          value={password}
                          onChange={(e) => setPassword(e.target.value)}
                          placeholder="Create a strong password"
                          className="premium-input pl-11"
                        />
                      </div>
                    </div>
                  </form>
                </motion.div>
              )}
            </AnimatePresence>

            <div className="flex gap-3 mt-6">
              {step === 2 && (
                <motion.button
                  whileHover={{ scale: 1.01 }}
                  whileTap={{ scale: 0.99 }}
                  onClick={() => { setStep(1); setError(''); }}
                  className="premium-btn premium-btn-ghost flex-1"
                >
                  <ArrowLeft className="w-4 h-4" />
                  Back
                </motion.button>
              )}
              <motion.button
                whileHover={{ scale: 1.01 }}
                whileTap={{ scale: 0.99 }}
                onClick={handleSubmit}
                disabled={loading}
                className="premium-btn premium-btn-primary flex-1"
              >
                {loading ? (
                  <span className="flex items-center gap-2">
                    <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    {step === 1 ? 'Continuing...' : 'Creating...'}
                  </span>
                ) : step === 1 ? 'Continue' : 'Create Account'}
              </motion.button>
            </div>

            <p className="mt-6 text-center" style={{ color: 'var(--text-muted)' }}>
              Already have an account?{' '}
              <Link href="/login" className="font-semibold hover:underline" style={{ color: 'var(--primary)' }}>
                Sign in
              </Link>
            </p>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
