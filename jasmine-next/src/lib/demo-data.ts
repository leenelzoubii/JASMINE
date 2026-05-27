import { AssessmentResult } from './assessments';

const DEMO_DOCTOR_ID = 'demo-doctor';
const DEMO_PARENT_ID = 'demo-parent';
const DEMO_CHILD_ID = 'demo-child-1';

const DEMO_CHILD = {
  id: DEMO_CHILD_ID,
  patientId: DEMO_CHILD_ID,
  patientName: 'Emma',
  age: 6,
  dob: '2020-03-15',
  about: 'Emma is an energetic 6-year-old who loves drawing and playing with building blocks. She has been showing some signs that prompted this screening.',
  professionalId: DEMO_DOCTOR_ID,
  professionalName: 'Dr. Jasmine',
  parentId: DEMO_PARENT_ID,
  parentEmail: 'parent@demo.com',
  parentName: 'John Parent',
  accessGranted: true,
  sharedAssessments: ['demo-assessment-1'],
  createdAt: { toMillis: () => Date.now() } as any,
};

const DEMO_ASSESSMENTS: AssessmentResult[] = [
  {
    id: 'demo-assessment-1',
    userId: DEMO_DOCTOR_ID,
    patientId: DEMO_CHILD_ID,
    patientName: 'Emma',
    date: '2026-05-25',
    ensemble_probability: 0.72,
    risk_level: 'Moderate Risk',
    num_frames_processed: 145,
    source: 'youtube',
    model_predictions: {
      rf: { probability: 0.68, risk_level: 'Moderate Risk' },
      svm: { probability: 0.75, risk_level: 'Moderate Risk' },
      lstm: { probability: 0.71, risk_level: 'Moderate Risk' },
      transformer: { probability: 0.74, risk_level: 'Moderate Risk' },
    },
    reviewed: true,
    shared: true,
    sharedNotes: 'Emma shows moderate indicators. Recommend follow-up screening in 3 months and early intervention activities.',
    createdAt: { toMillis: () => Date.now() } as any,
  },
];

const DEMO_CONNECTIONS = [
  {
    id: 'demo-conn-1',
    professionalId: DEMO_DOCTOR_ID,
    professionalName: 'Dr. Jasmine',
    parentId: DEMO_PARENT_ID,
    parentName: 'John Parent',
    patientId: DEMO_CHILD_ID,
    patientName: 'Emma',
  },
];

export function isDemoUser(userId: string): boolean {
  return userId === DEMO_DOCTOR_ID || userId === DEMO_PARENT_ID;
}

export function getDemoLinksByParent(): typeof DEMO_CHILD[] {
  return [DEMO_CHILD];
}

export function getDemoAssessmentsByPatient(): AssessmentResult[] {
  return DEMO_ASSESSMENTS;
}

export function getDemoLinksByPatientId(): typeof DEMO_CHILD[] {
  return [DEMO_CHILD];
}

export function getDemoConnections(userId: string): typeof DEMO_CONNECTIONS {
  if (userId === DEMO_PARENT_ID) return DEMO_CONNECTIONS;
  if (userId === DEMO_DOCTOR_ID) return DEMO_CONNECTIONS;
  return [];
}

export function getDemoLinksByProfessional(): typeof DEMO_CHILD[] {
  return [DEMO_CHILD];
}

export { DEMO_CHILD_ID, DEMO_DOCTOR_ID, DEMO_PARENT_ID };
