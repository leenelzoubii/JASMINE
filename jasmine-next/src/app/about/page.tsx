import { Brain, Activity, Shield, Users, Heart } from 'lucide-react';
import Link from 'next/link';

export default function AboutPage() {
  return (
    <div className="min-h-screen" style={{ backgroundColor: 'var(--background-alt)' }}>
      <header className="p-6" style={{ backgroundColor: 'var(--background)', borderBottom: '1px solid var(--border)' }}>
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2">
            <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ backgroundColor: 'var(--primary)' }}>
              <Brain className="w-6 h-6 text-white" />
            </div>
            <span className="text-xl font-bold" style={{ color: 'var(--primary)' }}>JASMINE</span>
          </Link>
          <Link href="/" className="text-sm font-medium" style={{ color: 'var(--primary)' }}>
            Back to Home
          </Link>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-16 space-y-16">
        {/* Hero */}
        <section className="text-center space-y-4">
          <h1 className="text-4xl font-bold" style={{ color: 'var(--foreground)' }}>
            About JASMINE
          </h1>
          <p className="text-lg max-w-2xl mx-auto" style={{ color: 'var(--text-muted)' }}>
            JASMINE (Joint Assessment & Screening for Movement INtelligence Evaluation) 
            is a research tool that analyzes children&apos;s movement patterns using pose estimation 
            and machine learning to assist in early ASD screening.
          </p>
        </section>

        {/* Mission */}
        <section className="grid md:grid-cols-2 gap-8">
          <div className="p-8 rounded-2xl" style={{ backgroundColor: 'var(--background)', border: '1px solid var(--border)' }}>
            <Brain className="w-10 h-10 mb-4" style={{ color: 'var(--primary)' }} />
            <h2 className="text-xl font-bold mb-3" style={{ color: 'var(--foreground)' }}>Our Mission</h2>
            <p style={{ color: 'var(--text-muted)' }}>
              To provide healthcare professionals with accessible, AI-powered screening tools 
              that can help identify early signs of Autism Spectrum Disorder through non-invasive 
              movement analysis.
            </p>
          </div>
          <div className="p-8 rounded-2xl" style={{ backgroundColor: 'var(--background)', border: '1px solid var(--border)' }}>
            <Shield className="w-10 h-10 mb-4" style={{ color: 'var(--primary)' }} />
            <h2 className="text-xl font-bold mb-3" style={{ color: 'var(--foreground)' }}>Privacy First</h2>
            <p style={{ color: 'var(--text-muted)' }}>
              All video data is processed locally. No footage is stored or transmitted. 
              We prioritize patient confidentiality and comply with healthcare data protection standards.
            </p>
          </div>
        </section>

        {/* How It Works */}
        <section className="p-8 rounded-2xl" style={{ backgroundColor: 'var(--background)', border: '1px solid var(--border)' }}>
          <h2 className="text-2xl font-bold mb-8 text-center" style={{ color: 'var(--foreground)' }}>How It Works</h2>
          <div className="grid md:grid-cols-4 gap-6">
            {[
              { step: 1, title: 'Upload Video', desc: 'Record a child\'s natural movement on video (MP4) or provide a YouTube link.' },
              { step: 2, title: 'Pose Extraction', desc: 'MediaPipe extracts 25 body keypoints from each frame — no special hardware needed.' },
              { step: 3, title: 'ML Analysis', desc: 'Four models (RF, SVM, LSTM, Transformer) analyze kinematic and statistical features.' },
              { step: 4, title: 'Risk Assessment', desc: 'Ensemble prediction provides a final ASD risk score with model-level breakdown.' },
            ].map((item) => (
              <div key={item.step} className="text-center">
                <div className="w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-3 text-white font-bold" style={{ backgroundColor: 'var(--primary)' }}>
                  {item.step}
                </div>
                <h3 className="font-semibold mb-2" style={{ color: 'var(--foreground)' }}>{item.title}</h3>
                <p className="text-sm" style={{ color: 'var(--text-muted)' }}>{item.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Disclaimer */}
        <section className="p-6 rounded-2xl" style={{ backgroundColor: 'rgba(217, 119, 6, 0.1)', border: '1px solid #d97706' }}>
          <div className="flex items-start gap-4">
            <Activity className="w-6 h-6 text-amber-600 flex-shrink-0 mt-1" />
            <div>
              <h3 className="font-semibold text-amber-800 dark:text-amber-200 mb-2">Research Use Only</h3>
              <p className="text-sm text-amber-700 dark:text-amber-300">
                JASMINE is a research prototype and is NOT a diagnostic tool. It should not be used 
                for clinical decision-making. Always consult qualified healthcare professionals for 
                diagnosis and treatment of ASD or any other condition.
              </p>
            </div>
          </div>
        </section>

        {/* Team */}
        <section className="text-center">
          <Users className="w-10 h-10 mx-auto mb-4" style={{ color: 'var(--primary)' }} />
          <h2 className="text-2xl font-bold mb-2" style={{ color: 'var(--foreground)' }}>Research Team</h2>
          <p style={{ color: 'var(--text-muted)' }}>
            Built with <Heart className="w-4 h-4 inline" style={{ color: '#dc2626' }} /> by researchers and engineers 
            dedicated to improving early ASD screening through technology.
          </p>
        </section>
      </main>

      <footer className="py-8 text-center text-sm" style={{ color: 'var(--text-muted)', borderTop: '1px solid var(--border)' }}>
        <p>JASMINE — Joint Assessment & Screening for Movement INtelligence Evaluation</p>
        <p className="mt-1">Research prototype — not for clinical use</p>
      </footer>
    </div>
  );
}
