'use client';

import Link from 'next/link';
import { Navbar } from '@/components/layout/navbar';
import { Brain, Shield, Users, Activity, ArrowRight, CheckCircle2 } from 'lucide-react';
import { motion } from 'framer-motion';

const features = [
  { icon: Brain, title: 'Pose Estimation', description: 'Advanced 2D pose detection using BODY-25 keypoints' },
  { icon: Activity, title: 'Multi-Model Analysis', description: 'Compare predictions from RF, SVM, LSTM, Transformer' },
  { icon: Shield, title: 'Privacy First', description: 'Only skeletal keypoints processed - no images stored' },
  { icon: Users, title: 'Interactive Visualization', description: 'Visualize skeletons with bounding boxes' },
];

export default function HomePage() {
  return (
    <div className="min-h-screen">
      <Navbar />
      
      {/* Hero */}
      <section className="pt-28 pb-16 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <h1 className="text-4xl md:text-5xl font-bold mb-4" style={{ color: 'var(--primary)' }}>
              JASMINE
            </h1>
            <p className="text-xl md:text-xl mb-4" style={{ color: 'var(--foreground)' }}>
              Joint Analysis and Screening for Motor Imbalances
            </p>
            <p className="text-lg mb-8 max-w-2xl mx-auto" style={{ color: 'var(--text-muted)' }}>
              A privacy-preserving autism spectrum disorder screening system using 2D pose estimation to analyze movement patterns.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                href="/register"
                className="inline-flex items-center justify-center gap-2 px-6 py-3 font-semibold rounded-xl transition-all"
                style={{ backgroundColor: 'var(--primary)', color: 'var(--text-primary)' }}
              >
                Get Started
                <ArrowRight className="w-5 h-5" />
              </Link>
              <Link
                href="/login"
                className="inline-flex items-center justify-center px-6 py-3 font-semibold rounded-xl"
                style={{ backgroundColor: 'var(--background-alt)', border: '1px solid var(--border)', color: 'var(--primary)' }}
              >
                Sign In
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Features */}
      <section className="py-16 px-4" style={{ backgroundColor: 'var(--background-alt)' }}>
        <div className="max-w-6xl mx-auto">
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                viewport={{ once: true }}
                className="p-6 rounded-2xl"
                style={{ backgroundColor: 'var(--background)', border: '1px solid var(--border)' }}
              >
                <div className="w-12 h-12 rounded-xl flex items-center justify-center mb-4" style={{ backgroundColor: 'var(--primary-light)' }}>
                  <feature.icon className="w-6 h-6" style={{ color: 'var(--primary)' }} />
                </div>
                <h3 className="text-lg font-semibold mb-2" style={{ color: 'var(--foreground)' }}>
                  {feature.title}
                </h3>
                <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-16 px-4">
        <div className="max-w-3xl mx-auto text-center">
          <h2 className="text-3xl font-bold mb-8" style={{ color: 'var(--foreground)' }}>
            How It Works
          </h2>
          <div className="flex flex-wrap justify-center gap-4">
            {['Video Input', 'Pose Detection', 'Feature Extraction', 'ML Prediction'].map((step, i) => (
              <div key={i} className="flex items-center gap-2">
                <div className="w-10 h-10 rounded-full flex items-center justify-center text-white font-bold" style={{ backgroundColor: 'var(--primary)' }}>
                  {i + 1}
                </div>
                <span className="font-medium" style={{ color: 'var(--foreground)' }}>{step}</span>
                {i < 3 && <ArrowRight className="w-4 h-4" style={{ color: 'var(--text-muted)' }} />}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-16 px-4" style={{ backgroundColor: 'var(--background-alt)' }}>
        <div className="max-w-xl mx-auto text-center">
          <h2 className="text-2xl font-bold mb-4" style={{ color: 'var(--foreground)' }}>
            Ready to Begin?
          </h2>
          <p className="mb-6" style={{ color: 'var(--text-muted)' }}>
            Join healthcare professionals using JASMINE for autism screening.
          </p>
          <Link
            href="/register"
            className="inline-flex items-center gap-2 px-8 py-4 font-semibold rounded-xl transition-all"
            style={{ backgroundColor: 'var(--primary)', color: 'var(--text-primary)' }}
          >
            Create Account
            <ArrowRight className="w-5 h-5" />
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 px-4" style={{ borderTop: '1px solid var(--border)' }}>
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
            © 2026 JASMINE. Research demo - not for clinical use.
          </p>
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4" style={{ color: 'var(--risk-low)' }} />
            <span className="text-sm" style={{ color: 'var(--text-muted)' }}>Privacy-preserving</span>
          </div>
        </div>
      </footer>
    </div>
  );
}