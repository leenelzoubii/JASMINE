import {
  collection,
  addDoc,
  getDocs,
  getDoc,
  doc,
  query,
  where,
  updateDoc,
  serverTimestamp,
  Timestamp,
} from 'firebase/firestore';
import { db } from '@/lib/firebase';

export interface AssessmentResult {
  id: string;
  userId: string;
  patientId: string;
  patientName: string;
  date: string;
  ensemble_probability: number;
  risk_level: string;
  num_frames_processed?: number;
  source?: string;
  youtube_url?: string;
  model_predictions: Record<string, { probability: number; risk_level: string }>;
  reviewed: boolean;
  shared: boolean;
  sharedNotes?: string;
  createdAt: Timestamp;
}

export async function saveAssessment(
  userId: string,
  data: Omit<AssessmentResult, 'id' | 'createdAt' | 'reviewed' | 'shared'>
): Promise<string> {
  const ref = await addDoc(collection(db, 'users', userId, 'assessments'), {
    ...data,
    reviewed: false,
    shared: false,
    createdAt: serverTimestamp(),
  });
  return ref.id;
}

export async function reviewAssessment(userId: string, assessmentId: string): Promise<void> {
  await updateDoc(doc(db, 'users', userId, 'assessments', assessmentId), { reviewed: true });
}

export async function shareAssessment(userId: string, assessmentId: string, notes?: string): Promise<void> {
  await updateDoc(doc(db, 'users', userId, 'assessments', assessmentId), { shared: true, sharedNotes: notes || '' });
}

export async function getAssessmentById(userId: string, assessmentId: string): Promise<AssessmentResult | null> {
  const snap = await getDoc(doc(db, 'users', userId, 'assessments', assessmentId));
  if (!snap.exists()) return null;
  return { id: snap.id, ...snap.data() } as AssessmentResult;
}

export async function getAssessments(userId: string): Promise<AssessmentResult[]> {
  const q = query(collection(db, 'users', userId, 'assessments'));
  const snap = await getDocs(q);
  const results = snap.docs.map((d) => {
    const data = d.data();
    return { id: d.id, ...data } as AssessmentResult;
  });
  results.sort((a, b) => {
    const tA = (a.createdAt as any)?.toMillis?.() || 0;
    const tB = (b.createdAt as any)?.toMillis?.() || 0;
    return tB - tA;
  });
  return results;
}

export async function getAssessmentsByPatient(userId: string, patientId: string): Promise<AssessmentResult[]> {
  const q = query(
    collection(db, 'users', userId, 'assessments'),
    where('patientId', '==', patientId)
  );
  const snap = await getDocs(q);
  const results = snap.docs.map((d) => {
    const data = d.data();
    return { id: d.id, ...data } as AssessmentResult;
  });
  results.sort((a, b) => {
    const tA = (a.createdAt as any)?.toMillis?.() || 0;
    const tB = (b.createdAt as any)?.toMillis?.() || 0;
    return tB - tA;
  });
  return results;
}
