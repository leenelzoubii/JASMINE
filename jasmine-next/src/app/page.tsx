'use client';

import Link from 'next/link';
import { Navbar } from '@/components/layout/navbar';
import {
  Brain, Shield, Users, Activity, ArrowRight, CheckCircle2,
  Sparkles, BarChart3, Layers, Play,
} from 'lucide-react';
import { motion, useScroll, useTransform, useInView } from 'framer-motion';
import { useRef } from 'react';
import { AnimatedTitle } from '@/components/animated-title';

const features = [
  { icon: Brain, title: 'Pose Estimation', description: 'Advanced 2D pose detection using 25 BODY-25 keypoints per frame' },
  { icon: Layers, title: 'Multi-Model Ensemble', description: 'RF, SVM, LSTM & Transformer weighted for 92.1% accuracy' },
  { icon: Shield, title: 'Privacy First', description: 'Only skeletal keypoints processed — no images or video stored' },
  { icon: BarChart3, title: 'AI Explainability', description: 'Feature importance, per-model contributions & plain-language reasoning' },
];

const steps = [
  { num: '01', title: 'Upload Video', desc: 'MP4 file or YouTube link of a child performing standardized movements' },
  { num: '02', title: 'Pose Detection', desc: 'MediaPipe extracts 25 body keypoints — shoulders, elbows, hips, knees & more' },
  { num: '03', title: 'Feature Analysis', desc: '983 kinematic & statistical features capture movement patterns' },
  { num: '04', title: 'Ensemble Prediction', desc: '4 models combine for a robust ASD likelihood score with explainability' },
];

const stats = [
  { value: '92.1%', label: 'Accuracy' },
  { value: '0.98', label: 'ROC-AUC' },
  { value: '1,374', label: 'Subjects' },
  { value: '4', label: 'Ensemble Models' },
];

function SectionAnim({ children, className, style, delay = 0 }: {
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
  delay?: number;
}) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: false });

  return (
    <motion.div
      ref={ref}
      animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: 30 }}
      transition={{ duration: 0.6, delay, ease: [0.16, 1, 0.3, 1] }}
      className={className}
      style={style}
    >
      {children}
    </motion.div>
  );
}

function CardAnim({ children, className, delay = 0 }: {
  children: React.ReactNode;
  className?: string;
  delay?: number;
}) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: false, margin: '-60px 0px' });

  return (
    <motion.div
      ref={ref}
      animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: 24 }}
      transition={{ duration: 0.5, delay, ease: [0.16, 1, 0.3, 1] }}
      whileHover={{ y: -4 }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

export default function HomePage() {
  const heroRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: heroRef,
    offset: ['start start', 'end start'],
  });
  const heroOpacity = useTransform(scrollYProgress, [0, 1], [1, 0.3]);
  const heroScale = useTransform(scrollYProgress, [0, 1], [1, 0.95]);

  return (
    <div className="min-h-screen overflow-x-hidden">
      <Navbar />

      {/* === Cinematic Hero === */}
      <motion.section
        ref={heroRef}
        style={{ opacity: heroOpacity, scale: heroScale }}
        className="relative min-h-screen flex items-center justify-center overflow-hidden"
      >
        <div
          className="absolute inset-0 -z-10"
          style={{ background: 'var(--gradient-hero)' }}
        />
        <div className="absolute inset-0 -z-10 opacity-[0.04]">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 rounded-full bg-blue-400 blur-[120px]" />
          <div className="absolute bottom-1/4 right-1/4 w-80 h-80 rounded-full bg-emerald-400 blur-[100px]" />
        </div>

        <div className="max-w-5xl mx-auto px-4 text-center pt-24 pb-20">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-sm font-medium mb-6"
              style={{
                backgroundColor: 'rgba(42, 127, 158, 0.1)',
                color: 'var(--primary)',
                border: '1px solid rgba(42, 127, 158, 0.2)',
              }}
            >
              <Sparkles className="w-4 h-4" />
              AI-Powered Screening Platform
            </motion.div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
            className="text-5xl sm:text-6xl md:text-7xl lg:text-8xl leading-[1.05] tracking-tight mb-5"
          >
            <AnimatedTitle />
          </motion.div>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.4, ease: [0.16, 1, 0.3, 1] }}
            className="text-lg sm:text-xl max-w-2xl mx-auto mb-10"
            style={{ color: 'var(--text-dim)' }}
          >
            A privacy-first platform that analyzes children&apos;s movement patterns
            through pose estimation, helping healthcare professionals with early
            ASD screening.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.55, ease: [0.16, 1, 0.3, 1] }}
            className="flex flex-col sm:flex-row gap-4 justify-center items-center"
          >
            <Link
              href="/register"
              className="premium-btn premium-btn-primary text-base px-8 py-4 shadow-lg"
            >
              Get Started Free
              <ArrowRight className="w-5 h-5" />
            </Link>
            <Link
              href="/about"
              className="premium-btn premium-btn-ghost text-base px-8 py-4"
              style={{ color: 'var(--text-primary)', borderColor: 'rgba(255,255,255,0.15)' }}
            >
              <Play className="w-4 h-4" />
              Learn More
            </Link>
          </motion.div>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1, delay: 0.8 }}
            className="mt-16 flex items-center justify-center gap-8 text-sm"
            style={{ color: 'var(--text-dim)' }}
          >
            <span className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4" style={{ color: 'var(--primary)' }} />
              HIPAA Compliant
            </span>
            <span className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4" style={{ color: 'var(--primary)' }} />
              92.1% Accuracy
            </span>
            <span className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4" style={{ color: 'var(--primary)' }} />
              1,374 Subjects
            </span>
          </motion.div>
        </div>
      </motion.section>

      {/* === Stats Bar === */}
      <SectionAnim
        className="py-12 px-4"
        style={{ backgroundColor: 'var(--background-alt)' }}
      >
        <div className="max-w-5xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-8 text-center">
          {stats.map((stat, i) => (
            <CardAnim key={stat.label} delay={i * 0.08}>
              <p className="text-3xl md:text-4xl font-bold" style={{ color: 'var(--primary)' }}>
                {stat.value}
              </p>
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>{stat.label}</p>
            </CardAnim>
          ))}
        </div>
      </SectionAnim>

      {/* === Features === */}
      <section className="py-20 md:py-28 px-4">
        <div className="max-w-6xl mx-auto">
          <SectionAnim className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold mb-4" style={{ color: 'var(--foreground)' }}>
              Everything You Need for Accurate Screening
            </h2>
            <p className="text-lg max-w-2xl mx-auto" style={{ color: 'var(--text-muted)' }}>
              A comprehensive suite of tools designed for healthcare professionals
            </p>
          </SectionAnim>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((feature, i) => (
              <CardAnim key={i} delay={i * 0.08} className="premium-card p-6 group cursor-default">
                <div
                  className="w-12 h-12 rounded-xl flex items-center justify-center mb-4 transition-all duration-300 group-hover:scale-110 group-hover:shadow-lg"
                  style={{ background: 'var(--gradient-primary-subtle)' }}
                >
                  <feature.icon className="w-6 h-6" style={{ color: 'var(--primary)' }} />
                </div>
                <h3 className="text-lg font-semibold mb-2" style={{ color: 'var(--foreground)' }}>
                  {feature.title}
                </h3>
                <p className="text-sm leading-relaxed" style={{ color: 'var(--text-muted)' }}>
                  {feature.description}
                </p>
              </CardAnim>
            ))}
          </div>
        </div>
      </section>

      {/* === How It Works === */}
      <section
        className="py-20 md:py-28 px-4"
        style={{ backgroundColor: 'var(--background-alt)' }}
      >
        <div className="max-w-6xl mx-auto">
          <SectionAnim className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold mb-4" style={{ color: 'var(--foreground)' }}>
              From Video to Insight
            </h2>
            <p className="text-lg max-w-2xl mx-auto" style={{ color: 'var(--text-muted)' }}>
              The ML pipeline transforms raw video into clinically meaningful predictions
            </p>
          </SectionAnim>

          <div className="space-y-6">
            {steps.map((step, i) => (
              <CardAnim key={step.num} delay={i * 0.08} className="premium-card p-6 md:p-8 flex items-start gap-6">
                <div
                  className="flex-shrink-0 w-14 h-14 rounded-2xl flex items-center justify-center text-lg font-bold text-white"
                  style={{ background: 'var(--gradient-primary)' }}
                >
                  {step.num}
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-1" style={{ color: 'var(--foreground)' }}>
                    {step.title}
                  </h3>
                  <p className="text-base" style={{ color: 'var(--text-muted)' }}>{step.desc}</p>
                </div>
              </CardAnim>
            ))}
          </div>
        </div>
      </section>

      {/* === CTA === */}
      <section className="py-24 px-4 relative overflow-hidden">
        <div
          className="absolute inset-0 -z-10"
          style={{ background: 'var(--gradient-hero)' }}
        />
        <div className="absolute inset-0 -z-10 opacity-[0.05]">
          <div className="absolute top-0 right-0 w-72 h-72 rounded-full bg-blue-300 blur-[120px]" />
          <div className="absolute bottom-0 left-0 w-64 h-64 rounded-full bg-emerald-300 blur-[100px]" />
        </div>

        <div className="max-w-3xl mx-auto text-center">
          <SectionAnim>
            <h2 className="text-3xl md:text-5xl font-bold mb-6" style={{ color: 'var(--text-primary)' }}>
              Ready to Transform Your Screening Workflow?
            </h2>
            <p className="text-lg mb-10 max-w-xl mx-auto" style={{ color: 'var(--text-dim)' }}>
              Join healthcare professionals using JASMINE for early, privacy-preserving
              autism screening.
            </p>
            <Link
              href="/register"
              className="premium-btn premium-btn-primary text-base px-10 py-4 shadow-lg"
            >
              Create Free Account
              <ArrowRight className="w-5 h-5" />
            </Link>
          </SectionAnim>
        </div>
      </section>

      {/* === Footer === */}
      <footer
        className="py-10 px-4"
        style={{
          backgroundColor: 'var(--background-alt)',
          borderTop: '1px solid var(--border-light)',
        }}
      >
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <Brain className="w-5 h-5" style={{ color: 'var(--primary)' }} />
            <span className="font-semibold" style={{ color: 'var(--foreground)' }}>JASMINE</span>
            <span className="text-sm" style={{ color: 'var(--text-muted)' }}>
              © 2026 — Research demo
            </span>
          </div>
          <div className="flex items-center gap-6">
            <Link href="/about" className="text-sm transition-colors" style={{ color: 'var(--text-muted)' }}>
              About
            </Link>
            <Link href="/login" className="text-sm transition-colors" style={{ color: 'var(--text-muted)' }}>
              Sign In
            </Link>
            <span className="flex items-center gap-1.5 text-sm" style={{ color: 'var(--text-dim)' }}>
              <Shield className="w-3.5 h-3.5" />
              Privacy-preserving
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
}
