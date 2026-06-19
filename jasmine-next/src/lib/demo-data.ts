export const DEMO_DOCTOR_ID = 'demo-doctor';
export const DEMO_PARENT_ID = 'demo-parent';

// --- Patient IDs ---
export const CHILD_EMMA = 'demo-child-emma';
export const CHILD_YARA = 'demo-child-yara';
export const CHILD_KARIM = 'demo-child-karim';
export const CHILD_SAMI = 'demo-child-sami';

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
    patientName: 'Tala Alawneh',
    age: 6,
    dob: '2020-03-15',
    about: 'Tala is an energetic 6-year-old who loves drawing and playing with building blocks.',
    professionalId: DEMO_DOCTOR_ID,
    professionalName: 'Dr. Jasmine',
    parentId: DEMO_PARENT_ID,
    parentEmail: 'parent@demo.com',
    parentName: 'Sara Alawneh',
    accessGranted: true,
    sharedAssessments: ['demo-asm-emma-1', 'demo-asm-emma-2', 'demo-asm-emma-3'],
    createdAt: { toMillis: () => Date.now() - 180 * 86400000 } as any,
  },
];

// --- Seed patients for demo-doctor ---
const SEED_PATIENTS = [
  {
    id: CHILD_EMMA,
    name: 'Tala Alawneh',
    dob: '2020-03-15',
    parentName: 'Sara Alawneh',
    email: 'parent@demo.com',
    phone: '+962-79-111-1111',
    lastVisit: '2026-05-25',
    risk: 'Moderate Risk',
  },
  {
    id: CHILD_YARA,
    name: 'Yara Khalil',
    dob: '2022-06-10',
    parentName: 'Nadia Khalil',
    email: 'nadia.khalil@example.com',
    phone: '+962-79-222-2222',
    lastVisit: '2026-06-10',
    risk: 'Low Risk',
  },
  {
    id: CHILD_KARIM,
    name: 'Karim Hassan',
    dob: '2019-01-05',
    parentName: 'Mona Hassan',
    email: 'mona.hassan@example.com',
    phone: '+962-79-333-3333',
    lastVisit: '2026-06-01',
    risk: 'High Risk',
  },
  {
    id: CHILD_SAMI,
    name: 'Sami Dawood',
    dob: '2021-08-20',
    parentName: 'Layla Dawood',
    email: 'layla.dawood@example.com',
    phone: '+962-79-444-4444',
    lastVisit: '2026-05-15',
    risk: 'Moderate Risk',
  },
];

// --- Seed assessments for demo-doctor ---
const SEED_ASSESSMENTS = [
  // Tala — 3 assessments, trending up
  {
    id: 'demo-asm-emma-1',
    userId: DEMO_DOCTOR_ID,
    patientId: CHILD_EMMA,
    patientName: 'Tala Alawneh',
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
    patientName: 'Tala Alawneh',
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
    patientName: 'Tala Alawneh',
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

  // Yara — 1 assessment, Low Risk
  {
    id: 'demo-asm-yara-1',
    userId: DEMO_DOCTOR_ID,
    patientId: CHILD_YARA,
    patientName: 'Yara Khalil',
    date: '2026-06-10',
    ensemble_probability: 0.22,
    risk_level: 'Low Risk',
    confidence: 0.85,
    num_frames_processed: 170,
    source: 'upload',
    model_predictions: {
      rf: { probability: 0.20, risk_level: 'Low Risk' },
      svm: { probability: 0.24, risk_level: 'Low Risk' },
      tcn: { probability: 0.21, risk_level: 'Low Risk' },
      transformer: { probability: 0.23, risk_level: 'Low Risk' },
    },
    reviewed: true,
    shared: false,
    sharedNotes: '',
    createdAt: { toMillis: () => Date.now() - 5 * 86400000 } as any,
  },

  // Karim — 2 assessments, High Risk
  {
    id: 'demo-asm-karim-1',
    userId: DEMO_DOCTOR_ID,
    patientId: CHILD_KARIM,
    patientName: 'Karim Hassan',
    date: '2026-06-01',
    ensemble_probability: 0.91,
    risk_level: 'High Risk',
    confidence: 0.76,
    num_frames_processed: 105,
    source: 'upload',
    model_predictions: {
      rf: { probability: 0.93, risk_level: 'High Risk' },
      svm: { probability: 0.88, risk_level: 'High Risk' },
      tcn: { probability: 0.91, risk_level: 'High Risk' },
      transformer: { probability: 0.92, risk_level: 'High Risk' },
    },
    reviewed: true,
    shared: false,
    sharedNotes: '',
    createdAt: { toMillis: () => Date.now() - 14 * 86400000 } as any,
  },
  {
    id: 'demo-asm-karim-2',
    userId: DEMO_DOCTOR_ID,
    patientId: CHILD_KARIM,
    patientName: 'Karim Hassan',
    date: '2026-04-15',
    ensemble_probability: 0.87,
    risk_level: 'High Risk',
    confidence: 0.73,
    num_frames_processed: 115,
    source: 'youtube',
    model_predictions: {
      rf: { probability: 0.89, risk_level: 'High Risk' },
      svm: { probability: 0.84, risk_level: 'High Risk' },
      tcn: { probability: 0.87, risk_level: 'High Risk' },
      transformer: { probability: 0.88, risk_level: 'High Risk' },
    },
    reviewed: true,
    shared: false,
    sharedNotes: '',
    createdAt: { toMillis: () => Date.now() - 60 * 86400000 } as any,
  },

  // Sami — 1 assessment, Moderate Risk
  {
    id: 'demo-asm-sami-1',
    userId: DEMO_DOCTOR_ID,
    patientId: CHILD_SAMI,
    patientName: 'Sami Dawood',
    date: '2026-05-15',
    ensemble_probability: 0.55,
    risk_level: 'Moderate Risk',
    confidence: 0.68,
    num_frames_processed: 125,
    source: 'upload',
    model_predictions: {
      rf: { probability: 0.52, risk_level: 'Moderate Risk' },
      svm: { probability: 0.58, risk_level: 'Moderate Risk' },
      tcn: { probability: 0.54, risk_level: 'Moderate Risk' },
      transformer: { probability: 0.56, risk_level: 'Moderate Risk' },
    },
    reviewed: true,
    shared: false,
    sharedNotes: '',
    createdAt: { toMillis: () => Date.now() - 30 * 86400000 } as any,
  },
];

// --- Seed access links (stored as PatientAccessLink-like objects in demo_accessLinks) ---
const SEED_ACCESS_LINKS = [
  {
    id: 'demo-link-1',
    patientId: CHILD_EMMA,
    patientName: 'Tala Alawneh',
    professionalId: DEMO_DOCTOR_ID,
    professionalName: 'Dr. Jasmine',
    parentId: DEMO_PARENT_ID,
    parentEmail: 'parent@demo.com',
    parentName: 'Sara Alawneh',
    accessGranted: true,
    accessGrantedAt: Date.now() - 180 * 86400000,
    accessRevokedAt: null,
    sharedAssessments: ['demo-asm-emma-1', 'demo-asm-emma-2', 'demo-asm-emma-3'],
    createdAt: Date.now() - 180 * 86400000,
  },
];

// --- Seed parent requests ---
const SEED_REQUESTS = [
  {
    id: 'demo-req-accepted-1',
    professionalId: DEMO_DOCTOR_ID,
    professionalName: 'Dr. Jasmine',
    patientId: CHILD_EMMA,
    patientName: 'Tala Alawneh',
    parentEmail: 'parent@demo.com',
    parentId: DEMO_PARENT_ID,
    parentName: 'Sara Alawneh',
    status: 'accepted',
    createdAt: { toMillis: () => Date.now() - 180 * 86400000 },
  },
];

// --- Seed parent accounts ---
const SEED_PARENT_ACCOUNTS = {
  'parent@demo.com': {
    id: DEMO_PARENT_ID,
    email: 'parent@demo.com',
    name: 'Sara Alawneh',
    mustChangePassword: false,
    isActive: true,
    createdBy: DEMO_DOCTOR_ID,
  },
};

// --- Seed versioning: bump this when seed data changes ---
const SEED_VERSION = 'v2';

// --- Seed function: populates all localStorage keys ---
export function seedDemoData(): void {
  if (typeof window === 'undefined') return;

  const versionKey = 'demo_seed_version';
  if (localStorage.getItem(versionKey) === SEED_VERSION) return;

  // Version mismatch or first visit — clear old demo data and re-seed
  const demoKeys = [
    'demo_patients_' + DEMO_DOCTOR_ID,
    'demo_assessments_' + DEMO_DOCTOR_ID,
    'demo_accessLinks',
    'demo_allRequests',
    'demo_parentAccounts_' + DEMO_DOCTOR_ID,
  ];
  for (const key of demoKeys) localStorage.removeItem(key);

  localStorage.setItem('demo_patients_' + DEMO_DOCTOR_ID, JSON.stringify(SEED_PATIENTS));
  localStorage.setItem('demo_assessments_' + DEMO_DOCTOR_ID, JSON.stringify(SEED_ASSESSMENTS));
  localStorage.setItem('demo_accessLinks', JSON.stringify(SEED_ACCESS_LINKS));
  localStorage.setItem('demo_allRequests', JSON.stringify(SEED_REQUESTS));
  localStorage.setItem('demo_parentAccounts_' + DEMO_DOCTOR_ID, JSON.stringify(SEED_PARENT_ACCOUNTS));

  localStorage.setItem(versionKey, SEED_VERSION);
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