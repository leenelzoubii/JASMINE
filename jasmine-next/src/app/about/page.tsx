import { Brain, Activity, Shield, Users, Heart, Sparkles, ArrowRight } from 'lucide-react';
import Link from 'next/link';

export default function AboutPage() {
  return (
    <div className="min-h-screen" style={{ backgroundColor: 'var(--background-alt)' }}>
      <header className="glass-card sticky top-0 z-40" style={{ borderBottom: '1px solid var(--border-light)' }}>
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-3 group">
            <div className="w-9 h-9 rounded-xl flex items-center justify-center transition-transform group-hover:scale-105" style={{ background: 'var(--gradient-primary)' }}>
              <Brain className="w-5 h-5 text-white" />
            </div>
            <span className="text-lg font-bold" style={{ color: 'var(--primary)' }}>JASMINE</span>
          </Link>
          <Link href="/" className="premium-btn premium-btn-ghost text-sm py-2 px-4">
            Back to Home
          </Link>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-16 space-y-20">
        <section className="text-center space-y-5 max-w-3xl mx-auto">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-sm font-medium" style={{ backgroundColor: 'var(--primary-light)', color: 'var(--primary)' }}>
            <Sparkles className="w-4 h-4" />
            About the Platform
          </div>
          <h1 className="text-4xl md:text-5xl font-bold leading-tight" style={{ color: 'var(--foreground)' }}>
            Jordanian Autism Screening using{' '}
            <span className="bg-clip-text text-transparent" style={{ backgroundImage: 'var(--gradient-primary)' }}>
              Multimodal Intelligent Neurodevelopmental
            </span>{' '}
            Evaluation
          </h1>
          <p className="text-lg leading-relaxed" style={{ color: 'var(--text-muted)' }}>
            JASMINE is a research tool that analyzes children&apos;s movement patterns using
            pose estimation and machine learning to assist in early ASD screening.
            Built with privacy and clinical accuracy at its core.
          </p>
        </section>

        <section className="grid md:grid-cols-2 gap-6">
          {[
            {
              icon: Brain, title: 'Our Mission',
              desc: 'To provide healthcare professionals with accessible, AI-powered screening tools that can help identify early signs of Autism Spectrum Disorder through non-invasive movement analysis.'
            },
            {
              icon: Shield, title: 'Privacy First',
              desc: 'All video data is processed locally. No footage is stored or transmitted. We prioritize patient confidentiality and comply with healthcare data protection standards.'
            },
            {
              icon: Activity, title: 'Clinical Accuracy',
              desc: 'A weighted ensemble of 4 ML models achieves 97.1% accuracy and 0.997 ROC-AUC, validated on the MMASD dataset of 1,374 subjects.'
            },
            {
              icon: Users, title: 'Collaborative Care',
              desc: 'Role-based portals connect healthcare professionals and parents, enabling seamless result sharing, messaging, and care coordination.'
            },
          ].map((item, i) => (
            <div key={i} className="premium-card p-8">
              <div className="w-12 h-12 rounded-xl flex items-center justify-center mb-5" style={{ background: 'var(--gradient-primary-subtle)' }}>
                <item.icon className="w-6 h-6" style={{ color: 'var(--primary)' }} />
              </div>
              <h2 className="text-xl font-bold mb-3" style={{ color: 'var(--foreground)' }}>{item.title}</h2>
              <p className="leading-relaxed" style={{ color: 'var(--text-muted)' }}>{item.desc}</p>
            </div>
          ))}
        </section>

        <section className="premium-card p-8 md:p-10">
          <h2 className="text-2xl font-bold mb-8 text-center" style={{ color: 'var(--foreground)' }}>How It Works</h2>
          <div className="grid md:grid-cols-4 gap-8">
            {[
              { step: '01', title: 'Upload Video', desc: 'Record a child\'s natural movement on video (MP4) or provide a YouTube link.' },
              { step: '02', title: 'Pose Extraction', desc: 'MediaPipe extracts 25 body keypoints from each frame — no special hardware needed.' },
              { step: '03', title: 'ML Analysis', desc: 'Four models (RF, SVM, TCN, Transformer) analyze 983 kinematic and statistical features.' },
              { step: '04', title: 'Risk Assessment', desc: 'Weighted ensemble prediction provides ASD likelihood with full model-level breakdown.' },
            ].map((item) => (
              <div key={item.step} className="text-center">
                <div className="w-14 h-14 rounded-2xl flex items-center justify-center mx-auto mb-4 text-white font-bold text-lg" style={{ background: 'var(--gradient-primary)' }}>
                  {item.step}
                </div>
                <h3 className="font-semibold mb-2" style={{ color: 'var(--foreground)' }}>{item.title}</h3>
                <p className="text-sm leading-relaxed" style={{ color: 'var(--text-muted)' }}>{item.desc}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="p-6 rounded-xl flex items-start gap-4" style={{ backgroundColor: 'var(--risk-moderate-bg)', border: '1px solid var(--risk-moderate)' }}>
          <Activity className="w-6 h-6 flex-shrink-0 mt-0.5" style={{ color: 'var(--risk-moderate)' }} />
          <div>
            <h3 className="font-semibold mb-1" style={{ color: 'var(--risk-moderate)' }}>Research Use Only</h3>
            <p className="text-sm leading-relaxed" style={{ color: 'var(--risk-moderate)' }}>
              JASMINE is a research prototype and is NOT a diagnostic tool. Always consult
              qualified healthcare professionals for diagnosis and treatment of ASD.
            </p>
          </div>
        </section>

        <section className="text-center">
          <div className="w-14 h-14 rounded-xl flex items-center justify-center mx-auto mb-4" style={{ background: 'var(--gradient-primary-subtle)' }}>
            <Heart className="w-7 h-7" style={{ color: 'var(--primary)' }} />
          </div>
          <h2 className="text-2xl font-bold mb-2" style={{ color: 'var(--foreground)' }}>Research Team</h2>
          <p className="max-w-md mx-auto" style={{ color: 'var(--text-muted)' }}>
            Built with dedication by researchers and engineers committed to improving
            early ASD screening through accessible technology.
          </p>
        </section>
      </main>

      <footer className="py-8 text-center text-sm" style={{ color: 'var(--text-dim)', borderTop: '1px solid var(--border-light)' }}>
        <p>JASMINE — Jordanian Autism Screening using Multimodal Intelligent Neurodevelopmental Evaluation</p>
        <p className="mt-1">Research prototype — not for clinical use</p>
      </footer>
    </div>
  );
}
