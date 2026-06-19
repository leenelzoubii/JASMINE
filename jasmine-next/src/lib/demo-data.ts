export const DEMO_DOCTOR_ID = 'demo-doctor';
export const DEMO_PARENT_ID = 'demo-parent';

// --- Patient IDs ---
export const CHILD_EMMA = 'demo-child-emma';
export const CHILD_LIAM = 'demo-child-liam';
export const CHILD_SOPHIA = 'demo-child-sophia';

export interface DemoLink {
  id: string;
  patientId: string;
  patientName: string;
  age: number;
  dob: string;
  about: string;
  professionalId: string;
  professionalName: string;
  parentId: string;
  parentEmail: string;
  parentName: string;
  accessGranted: boolean;
  sharedAssessments: string[];
  createdAt: { toMillis: () => number };
}

// --- Parent access links (for parent side) ---
const DEMO_LINKS: DemoLink[] = [
  {
    id: 'demo-link-emma',
    patientId: CHILD_EMMA,
    patientName: 'Emma Johnson',
    age: 6,
    dob: '2020-03-15',
    about: 'Emma is an energetic 6-year-old who loves drawing and playing with building blocks.',
    professionalId: DEMO_DOCTOR_ID,
    professionalName: 'Dr. Jasmine',
    parentId: DEMO_PARENT_ID,
    parentEmail: 'parent@demo.com',
    parentName: 'Sarah Johnson',
    accessGranted: true,
    sharedAssessments: ['demo-asm-emma-1', 'demo-asm-emma-2', 'demo-asm-emma-3'],
    createdAt: { toMillis: () => Date.now() - 180 * 86400000 } as any,
  },
  {
    id: 'demo-link-sophia',
    patientId: CHILD_SOPHIA,
    patientName: 'Sophia Chen',
    age: 5,
    dob: '2021-01-10',
    about: 'Sophia is a lively 5-year-old who enjoys music and dancing.',
    professionalId: DEMO_DOCTOR_ID,
    professionalName: 'Dr. Jasmine',
    parentId: 'demo-parent-sophia',
    parentEmail: 'mei.chen@example.com',
    parentName: 'Mei Chen',
    accessGranted: true,
    sharedAssessments: ['demo-asm-sophia-1', 'demo-asm-sophia-2'],
    createdAt: { toMillis: () => Date.now() - 90 * 86400000 } as any,
  },
];

// --- Seed patients for demo-doctor ---
const SEED_PATIENTS = [
  {
    id: CHILD_EMMA,
    name: 'Emma Johnson',
    dob: '2020-03-15',
    parentName: 'Sarah Johnson',
    email: 'parent@demo.com',
    phone: '+1-555-0101',
    lastVisit: '2026-05-25',
    risk: 'Moderate Risk',
  },
  {
    id: CHILD_LIAM,
    name: 'Liam Smith',
    dob: '2019-08-22',
    parentName: 'David Smith',
    email: 'david.smith@example.com',
    phone: '+1-555-0102',
    lastVisit: '2026-04-10',
    risk: 'Low Risk',
  },
  {
    id: CHILD_SOPHIA,
    name: 'Sophia Chen',
    dob: '2021-01-10',
    parentName: 'Mei Chen',
    email: 'mei.chen@example.com',
    phone: '+1-555-0103',
    lastVisit: '2026-06-01',
    risk: 'High Risk',
  },
];

// --- Seed assessments for demo-doctor ---
const SEED_ASSESSMENTS = [
  // Emma — 3 assessments, trending up
  {
    id: 'demo-asm-emma-1',
    userId: DEMO_DOCTOR_ID,
    patientId: CHILD_EMMA,
    patientName: 'Emma Johnson',
    date: '2026-05-25',
    ensemble_probability: 0.72,
    risk_level: 'Moderate Risk',
    confidence: 0.74,
    num_frames_processed: 145,
    source: 'upload',
    model_predictions: {
      rf: { probability: 0.68, risk_level: 'Moderate Risk' },
      svm: { probability: 0.75, risk_level: 'Moderate Risk' },
      tcn: { probability: 0.71, risk_level: 'Moderate Risk' },
      transformer: { probability: 0.74, risk_level: 'Moderate Risk' },
    },
    reviewed: true,
    shared: true,
    sharedNotes: 'Moderate indicators. Recommend follow-up in 3 months.',
    createdAt: { toMillis: () => Date.now() - 7 * 86400000 } as any,
  },
  {
    id: 'demo-asm-emma-2',
    userId: DEMO_DOCTOR_ID,
    patientId: CHILD_EMMA,
    patientName: 'Emma Johnson',
    date: '2026-03-15',
    ensemble_probability: 0.65,
    risk_level: 'Moderate Risk',
    confidence: 0.71,
    num_frames_processed: 120,
    source: 'upload',
    model_predictions: {
      rf: { probability: 0.62, risk_level: 'Moderate Risk' },
      svm: { probability: 0.68, risk_level: 'Moderate Risk' },
      tcn: { probability: 0.63, risk_level: 'Moderate Risk' },
      transformer: { probability: 0.67, risk_level: 'Moderate Risk' },
    },
    reviewed: true,
    shared: true,
    sharedNotes: '',
    createdAt: { toMillis: () => Date.now() - 80 * 86400000 } as any,
  },
  {
    id: 'demo-asm-emma-3',
    userId: DEMO_DOCTOR_ID,
    patientId: CHILD_EMMA,
    patientName: 'Emma Johnson',
    date: '2026-01-10',
    ensemble_probability: 0.58,
    risk_level: 'Moderate Risk',
    confidence: 0.69,
    num_frames_processed: 130,
    source: 'youtube',
    model_predictions: {
      rf: { probability: 0.55, risk_level: 'Moderate Risk' },
      svm: { probability: 0.61, risk_level: 'Moderate Risk' },
      tcn: { probability: 0.57, risk_level: 'Moderate Risk' },
      transformer: { probability: 0.59, risk_level: 'Moderate Risk' },
    },
    reviewed: true,
    shared: false,
    sharedNotes: '',
    createdAt: { toMillis: () => Date.now() - 165 * 86400000 } as any,
  },

  // Liam — 2 assessments, Low Risk
  {
    id: 'demo-asm-liam-1',
    userId: DEMO_DOCTOR_ID,
    patientId: CHILD_LIAM,
    patientName: 'Liam Smith',
    date: '2026-04-10',
    ensemble_probability: 0.28,
    risk_level: 'Low Risk',
    confidence: 0.81,
    num_frames_processed: 160,
    source: 'upload',
    model_predictions: {
      rf: { probability: 0.25, risk_level: 'Low Risk' },
      svm: { probability: 0.30, risk_level: 'Low Risk' },
      tcn: { probability: 0.27, risk_level: 'Low Risk' },
      transformer: { probability: 0.29, risk_level: 'Low Risk' },
    },
    reviewed: true,
    shared: false,
    sharedNotes: '',
    createdAt: { toMillis: () => Date.now() - 60 * 86400000 } as any,
  },
  {
    id: 'demo-asm-liam-2',
    userId: DEMO_DOCTOR_ID,
    patientId: CHILD_LIAM,
    patientName: 'Liam Smith',
    date: '2026-02-20',
    ensemble_probability: 0.32,
    risk_level: 'Low Risk',
    confidence: 0.78,
    num_frames_processed: 140,
    source: 'youtube',
    model_predictions: {
      rf: { probability: 0.29, risk_level: 'Low Risk' },
      svm: { probability: 0.35, risk_level: 'Low Risk' },
      tcn: { probability: 0.31, risk_level: 'Low Risk' },
      transformer: { probability: 0.33, risk_level: 'Low Risk' },
    },
    reviewed: true,
    shared: false,
    sharedNotes: '',
    createdAt: { toMillis: () => Date.now() - 110 * 86400000 } as any,
  },

  // Sophia — 2 assessments, High Risk
  {
    id: 'demo-asm-sophia-1',
    userId: DEMO_DOCTOR_ID,
    patientId: CHILD_SOPHIA,
    patientName: 'Sophia Chen',
    date: '2026-06-01',
    ensemble_probability: 0.88,
    risk_level: 'High Risk',
    confidence: 0.72,
    num_frames_processed: 110,
    source: 'upload',
    model_predictions: {
      rf: { probability: 0.90, risk_level: 'High Risk' },
      svm: { probability: 0.85, risk_level: 'High Risk' },
      tcn: { probability: 0.88, risk_level: 'High Risk' },
      transformer: { probability: 0.89, risk_level: 'High Risk' },
    },
    reviewed: true,
    shared: true,
    sharedNotes: 'High indicators detected. Urgent referral recommended for specialist evaluation.',
    createdAt: { toMillis: () => Date.now() - 14 * 86400000 } as any,
  },
  {
    id: 'demo-asm-sophia-2',
    userId: DEMO_DOCTOR_ID,
    patientId: CHILD_SOPHIA,
    patientName: 'Sophia Chen',
    date: '2026-05-01',
    ensemble_probability: 0.85,
    risk_level: 'High Risk',
    confidence: 0.70,
    num_frames_processed: 100,
    source: 'youtube',
    model_predictions: {
      rf: { probability: 0.87, risk_level: 'High Risk' },
      svm: { probability: 0.82, risk_level: 'High Risk' },
      tcn: { probability: 0.86, risk_level: 'High Risk' },
      transformer: { probability: 0.84, risk_level: 'High Risk' },
    },
    reviewed: true,
    shared: false,
    sharedNotes: '',
    createdAt: { toMillis: () => Date.now() - 45 * 86400000 } as any,
  },
];

// --- Seed access links (stored as PatientAccessLink-like objects in demo_accessLinks) ---
const SEED_ACCESS_LINKS = [
  {
    id: 'demo-link-1',
    patientId: CHILD_EMMA,
    patientName: 'Emma Johnson',
    professionalId: DEMO_DOCTOR_ID,
    professionalName: 'Dr. Jasmine',
    parentId: DEMO_PARENT_ID,
    parentEmail: 'parent@demo.com',
    parentName: 'Sarah Johnson',
    accessGranted: true,
    accessGrantedAt: Date.now() - 180 * 86400000,
    accessRevokedAt: null,
    sharedAssessments: ['demo-asm-emma-1', 'demo-asm-emma-2', 'demo-asm-emma-3'],
    createdAt: Date.now() - 180 * 86400000,
  },
  {
    id: 'demo-link-2',
    patientId: CHILD_SOPHIA,
    patientName: 'Sophia Chen',
    professionalId: DEMO_DOCTOR_ID,
    professionalName: 'Dr. Jasmine',
    parentId: 'demo-parent-sophia',
    parentEmail: 'mei.chen@example.com',
    parentName: 'Mei Chen',
    accessGranted: true,
    accessGrantedAt: Date.now() - 90 * 86400000,
    accessRevokedAt: null,
    sharedAssessments: ['demo-asm-sophia-1'],
    createdAt: Date.now() - 90 * 86400000,
  },
  {
    id: 'demo-link-3',
    patientId: CHILD_LIAM,
    patientName: 'Liam Smith',
    professionalId: DEMO_DOCTOR_ID,
    professionalName: 'Dr. Jasmine',
    parentId: 'demo-parent-liam',
    parentEmail: 'david.smith@example.com',
    parentName: 'David Smith',
    accessGranted: false,
    accessGrantedAt: null,
    accessRevokedAt: null,
    sharedAssessments: [],
    createdAt: Date.now() - 60 * 86400000,
  },
];

// --- Seed parent requests ---
const SEED_REQUESTS = [
  {
    id: 'demo-req-pending-1',
    professionalId: DEMO_DOCTOR_ID,
    professionalName: 'Dr. Jasmine',
    patientId: CHILD_LIAM,
    patientName: 'Liam Smith',
    parentEmail: 'david.smith@example.com',
    parentName: 'David Smith',
    status: 'pending',
    createdAt: { toMillis: () => Date.now() - 60 * 86400000 },
  },
  {
    id: 'demo-req-accepted-1',
    professionalId: DEMO_DOCTOR_ID,
    professionalName: 'Dr. Jasmine',
    patientId: CHILD_EMMA,
    patientName: 'Emma Johnson',
    parentEmail: 'parent@demo.com',
    parentId: DEMO_PARENT_ID,
    parentName: 'Sarah Johnson',
    status: 'accepted',
    createdAt: { toMillis: () => Date.now() - 180 * 86400000 },
  },
  {
    id: 'demo-req-accepted-2',
    professionalId: DEMO_DOCTOR_ID,
    professionalName: 'Dr. Jasmine',
    patientId: CHILD_SOPHIA,
    patientName: 'Sophia Chen',
    parentEmail: 'mei.chen@example.com',
    parentId: 'demo-parent-sophia',
    parentName: 'Mei Chen',
    status: 'accepted',
    createdAt: { toMillis: () => Date.now() - 90 * 86400000 },
  },
];

// --- Seed parent accounts ---
const SEED_PARENT_ACCOUNTS = {
  'parent@demo.com': {
    id: DEMO_PARENT_ID,
    email: 'parent@demo.com',
    name: 'Sarah Johnson',
    mustChangePassword: false,
    isActive: true,
    createdBy: DEMO_DOCTOR_ID,
  },
  'mei.chen@example.com': {
    id: 'demo-parent-sophia',
    email: 'mei.chen@example.com',
    name: 'Mei Chen',
    mustChangePassword: true,
    isActive: true,
    createdBy: DEMO_DOCTOR_ID,
  },
  'david.smith@example.com': {
    id: 'demo-parent-liam',
    email: 'david.smith@example.com',
    name: 'David Smith',
    mustChangePassword: true,
    isActive: true,
    createdBy: DEMO_DOCTOR_ID,
  },
};

// --- Seed function: populates all localStorage keys on first call ---
let seeded = false;

export function seedDemoData(): void {
  if (typeof window === 'undefined') return;
  if (seeded) return;
  seeded = true;

  // Seed patients for demo-doctor
  const patientKey = 'demo_patients_' + DEMO_DOCTOR_ID;
  if (!localStorage.getItem(patientKey)) {
    localStorage.setItem(patientKey, JSON.stringify(SEED_PATIENTS));
  }

  // Seed assessments for demo-doctor
  const asmKey = 'demo_assessments_' + DEMO_DOCTOR_ID;
  if (!localStorage.getItem(asmKey)) {
    localStorage.setItem(asmKey, JSON.stringify(SEED_ASSESSMENTS));
  }

  // Seed access links
  const linkKey = 'demo_accessLinks';
  if (!localStorage.getItem(linkKey)) {
    localStorage.setItem(linkKey, JSON.stringify(SEED_ACCESS_LINKS));
  }

  // Seed parent requests
  const reqKey = 'demo_allRequests';
  if (!localStorage.getItem(reqKey)) {
    localStorage.setItem(reqKey, JSON.stringify(SEED_REQUESTS));
  }

  // Seed parent accounts
  const acctKey = 'demo_parentAccounts_' + DEMO_DOCTOR_ID;
  if (!localStorage.getItem(acctKey)) {
    localStorage.setItem(acctKey, JSON.stringify(SEED_PARENT_ACCOUNTS));
  }
}

export function ensureDemoSeeded(): void {
  if (typeof window === 'undefined') return;
  seedDemoData();
}

// --- Export helpers used by pages ---

export function getDemoLinksByParent(): DemoLink[] {
  ensureDemoSeeded();
  return DEMO_LINKS.filter(l => l.parentId === DEMO_PARENT_ID || l.parentEmail === 'parent@demo.com');
}

export function getDemoAssessmentsByPatient(): any[] {
  ensureDemoSeeded();
  return SEED_ASSESSMENTS.filter(a => a.shared);
}

export function getDemoLinksByPatientId(patientId?: string): DemoLink[] {
  ensureDemoSeeded();
  if (patientId) return DEMO_LINKS.filter(l => l.patientId === patientId);
  return DEMO_LINKS;
}

export function getDemoConnections(userId: string): any[] {
  ensureDemoSeeded();
  if (userId === DEMO_PARENT_ID) {
    return SEED_REQUESTS.filter(r => r.status === 'accepted' && r.parentId === DEMO_PARENT_ID)
      .map(r => ({ id: r.id, professionalId: r.professionalId, professionalName: r.professionalName, parentId: r.parentId, parentName: r.parentName, patientId: r.patientId, patientName: r.patientName }));
  }
  if (userId === DEMO_DOCTOR_ID) {
    return SEED_REQUESTS.filter(r => r.status === 'accepted')
      .map(r => ({ id: r.id, professionalId: r.professionalId, professionalName: r.professionalName, parentId: r.parentId || '', parentName: r.parentName, patientId: r.patientId, patientName: r.patientName }));
  }
  return [];
}

export function getDemoLinksByProfessional(): DemoLink[] {
  ensureDemoSeeded();
  return DEMO_LINKS;
}

export function isDemoUser(userId: string): boolean {
  return userId === DEMO_DOCTOR_ID || userId === DEMO_PARENT_ID;
}

export const DEMO_CHILD_ID = CHILD_EMMA;