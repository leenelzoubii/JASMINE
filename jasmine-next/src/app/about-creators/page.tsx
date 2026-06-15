'use client';

import Link from 'next/link';
import { Navbar } from '@/components/layout/navbar';
import { Brain, Github, Linkedin, Sparkles, Quote, ArrowLeft } from 'lucide-react';
import { motion } from 'framer-motion';

const container = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.12, delayChildren: 0.2 } },
};

const fadeUp = {
  hidden: { opacity: 0, y: 30 },
  show: { opacity: 1, y: 0, transition: { duration: 0.6, ease: [0.16, 1, 0.3, 1] } },
};

const cardReveal = {
  hidden: { opacity: 0, y: 40, scale: 0.96 },
  show: { opacity: 1, y: 0, scale: 1, transition: { duration: 0.5, ease: [0.16, 1, 0.3, 1] } },
};

const creators = [
  {
    name: 'Leen El Zoubi',
    role: 'Frontend Developer and AI Engineer',
    description:
      'Responsible for the design and implementation of the frontend components of the system, including user interfaces, navigation flows, dashboards, and user experience enhancements. In addition, she contributed to the machine learning pipeline by participating in model development, training, evaluation, and performance optimization of the AI models used in the project.',
    image: '/images/leen.JPG',
    initials: 'LZ',
    gradient: 'from-[#2a7f9e] to-[#3a9b8a]',
    github: 'https://github.com/leenelzoubii',
    linkedin: 'https://www.linkedin.com/in/leen-el-zoubi-425342255/',
  },
  {
    name: 'Siba Al Jarrah',
    role: 'Backend Developer and Deployment Specialist',
    description:
      'Responsible for developing the backend API endpoints, authentication mechanisms, and server-side logic of the system. She also managed system deployment, cloud infrastructure, application hosting, and maintenance of the production environment to ensure reliable system performance.',
    image: '/images/siba.jpeg',
    initials: 'SJ',
    gradient: 'from-[#3a9b8a] to-[#2a7f9e]',
    github: 'https://github.com/sibaaljarrah',
    linkedin: 'https://www.linkedin.com/in/siba-al-jarrah/',
  },
  {
    name: 'Shahd Abu Baker',
    role: 'Data Scientist and Database Administrator',
    description:
      'Responsible for designing and implementing the database architecture, managing data storage, and handling all database-related operations. She gathered and organized the datasets used throughout the project, conducted exploratory data analysis (EDA), performed data cleaning, preprocessing, feature preparation, and dataset validation to ensure data quality and suitability for machine learning model training.',
    image: '/images/shahd.jpeg',
    initials: 'SB',
    gradient: 'from-[#2a7f9e] to-[#4a9bb8]',
    github: 'https://github.com/Shahed04ml',
    linkedin: 'https://www.linkedin.com/in/shahd-abu-baker-340255339/',
  },
];

export default function AboutCreatorsPage() {
  return (
    <div className="min-h-screen overflow-x-hidden">
      <Navbar />

      {/* Hero */}
      <section className="relative pt-28 pb-20 md:pt-36 md:pb-28 px-4 overflow-hidden">
        <div
          className="absolute inset-0 -z-10"
          style={{ background: 'var(--gradient-hero)' }}
        />
        <div className="absolute inset-0 -z-10 opacity-[0.04]">
          <div className="absolute top-1/3 left-1/4 w-96 h-96 rounded-full bg-blue-400 blur-[120px] animate-float" />
          <div className="absolute bottom-1/3 right-1/4 w-80 h-80 rounded-full bg-emerald-400 blur-[100px] animate-float" style={{ animationDelay: '-2s' }} />
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-72 h-72 rounded-full bg-teal-300 blur-[90px] animate-float" style={{ animationDelay: '-4s' }} />
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          className="max-w-4xl mx-auto text-center"
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
            className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-sm font-medium mb-6"
            style={{
              backgroundColor: 'rgba(42, 127, 158, 0.1)',
              color: 'var(--primary)',
              border: '1px solid rgba(42, 127, 158, 0.2)',
            }}
          >
            <Sparkles className="w-4 h-4" />
            The Team
          </motion.div>

          <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold leading-[1.1] mb-5" style={{ color: 'var(--text-primary)' }}>
            Meet the{' '}
            <span className="bg-clip-text text-transparent" style={{ backgroundImage: 'var(--gradient-primary)' }}>
              Creators
            </span>
          </h1>
          <p className="text-lg md:text-xl max-w-2xl mx-auto mb-8" style={{ color: 'var(--text-dim)' }}>
            The team behind this AI-powered healthcare intelligence platform
          </p>
          <div className="flex justify-center gap-3">
            <Link
              href="/"
              className="premium-btn premium-btn-ghost text-sm py-2.5 px-5"
              style={{ color: 'var(--text-primary)', borderColor: 'rgba(255,255,255,0.15)' }}
            >
              <ArrowLeft className="w-4 h-4" />
              Back to Home
            </Link>
          </div>
        </motion.div>
      </section>

      {/* Stats bar */}
      <motion.section
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true }}
        className="py-10 px-4"
        style={{ backgroundColor: 'var(--background-alt)' }}
      >
        <div className="max-w-3xl mx-auto grid grid-cols-3 gap-8 text-center">
          {[
            { value: '3', label: 'Team Members' },
            { value: '2', label: 'Disciplines' },
            { value: '1', label: 'Shared Mission' },
          ].map((stat, i) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 15 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1 }}
            >
              <p className="text-2xl md:text-3xl font-bold" style={{ color: 'var(--primary)' }}>
                {stat.value}
              </p>
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>{stat.label}</p>
            </motion.div>
          ))}
        </div>
      </motion.section>

      {/* Team Grid */}
      <section className="py-16 md:py-24 px-4">
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-14"
          >
            <h2 className="text-3xl md:text-4xl font-bold mb-3" style={{ color: 'var(--foreground)' }}>
              The People Behind JASMINE
            </h2>
            <div className="w-16 h-1 mx-auto rounded-full" style={{ background: 'var(--gradient-primary)' }} />
          </motion.div>

          <motion.div
            variants={container}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: '-50px' }}
            className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 md:gap-8"
          >
            {creators.map((creator, index) => (
              <motion.div
                key={creator.name}
                variants={cardReveal}
                whileHover={{ y: -6 }}
                className="group relative premium-card p-6 md:p-8 flex flex-col items-center text-center overflow-hidden"
              >
                {/* Gradient top accent */}
                <div
                  className="absolute top-0 left-0 right-0 h-1 opacity-60"
                  style={{ background: `linear-gradient(90deg, ${creator.gradient.includes('2a7f9e') ? 'var(--primary)' : ''}, var(--primary-muted))` }}
                />

                {/* Avatar */}
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  className="relative mb-5"
                >
                  {creator.image ? (
                    <img
                      src={creator.image}
                      alt={creator.name}
                      className="w-24 h-24 md:w-28 md:h-28 rounded-2xl object-cover shadow-lg"
                    />
                  ) : (
                    <div
                      className="w-24 h-24 md:w-28 md:h-28 rounded-2xl flex items-center justify-center text-white text-2xl md:text-3xl font-bold shadow-lg"
                      style={{ background: `linear-gradient(135deg, ${creator.gradient})` }}
                    >
                      {creator.initials}
                    </div>
                  )}
                  <div
                    className="absolute -bottom-1 -right-1 w-7 h-7 rounded-full flex items-center justify-center"
                    style={{ backgroundColor: 'var(--background)' }}
                  >
                    <Brain className="w-3.5 h-3.5" style={{ color: 'var(--primary)' }} />
                  </div>
                </motion.div>

                {/* Info */}
                <h3 className="text-xl font-bold mb-1" style={{ color: 'var(--foreground)' }}>
                  {creator.name}
                </h3>
                <p
                  className="text-sm font-medium mb-3 px-3 py-1 rounded-full"
                  style={{ backgroundColor: 'var(--primary-light)', color: 'var(--primary)' }}
                >
                  {creator.role}
                </p>
                <p className="text-sm leading-relaxed mb-5" style={{ color: 'var(--text-muted)' }}>
                  {creator.description}
                </p>

                {/* Social links */}
                <div className="flex items-center gap-3 mt-auto">
                  <motion.a
                    whileHover={{ scale: 1.1, y: -2 }}
                    whileTap={{ scale: 0.95 }}
                    href={creator.github}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="p-2.5 rounded-xl transition-all"
                    style={{ backgroundColor: 'var(--background-alt)' }}
                  >
                    <Github className="w-4 h-4" style={{ color: 'var(--text-secondary)' }} />
                  </motion.a>
                  <motion.a
                    whileHover={{ scale: 1.1, y: -2 }}
                    whileTap={{ scale: 0.95 }}
                    href={creator.linkedin}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="p-2.5 rounded-xl transition-all"
                    style={{ backgroundColor: 'var(--background-alt)' }}
                  >
                    <Linkedin className="w-4 h-4" style={{ color: 'var(--text-secondary)' }} />
                  </motion.a>
                </div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Quote / Mission */}
      <section className="py-16 md:py-20 px-4 relative overflow-hidden" style={{ backgroundColor: 'var(--background-alt)' }}>
        <div className="absolute inset-0 opacity-[0.02]">
          <div className="absolute top-0 right-0 w-64 h-64 rounded-full bg-blue-400 blur-[100px]" />
          <div className="absolute bottom-0 left-0 w-64 h-64 rounded-full bg-emerald-400 blur-[100px]" />
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="max-w-3xl mx-auto text-center relative"
        >
          <Quote className="w-10 h-10 mx-auto mb-6" style={{ color: 'var(--primary)' }} />
          <blockquote className="text-xl md:text-2xl font-medium leading-relaxed mb-6" style={{ color: 'var(--foreground)' }}>
            &ldquo;Built with purpose, driven by passion — making early autism screening
            more accessible through the power of AI and computer vision.&rdquo;
          </blockquote>
          <div className="w-12 h-0.5 mx-auto mb-4 rounded-full" style={{ background: 'var(--gradient-primary)' }} />
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>— The JASMINE Team</p>
        </motion.div>
      </section>

      {/* Footer */}
      <footer className="py-8 px-4" style={{ backgroundColor: 'var(--background)', borderTop: '1px solid var(--border-light)' }}>
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <Brain className="w-5 h-5" style={{ color: 'var(--primary)' }} />
            <span className="font-semibold" style={{ color: 'var(--foreground)' }}>JASMINE</span>
            <span className="text-sm" style={{ color: 'var(--text-muted)' }}>© 2026</span>
          </div>
          <div className="flex items-center gap-6 text-sm" style={{ color: 'var(--text-muted)' }}>
            <Link href="/" className="transition-colors hover:opacity-70">Home</Link>
            <Link href="/about" className="transition-colors hover:opacity-70">About</Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
