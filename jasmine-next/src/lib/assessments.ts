import {
  collection,
  addDoc,
  getDocs,
  getDoc,
  doc,
  query,
  where,
  updateDoc,
  deleteDoc,
  serverTimestamp,
  Timestamp,
} from 'firebase/firestore';
import { db } from '@/lib/firebase';
import { updateSharedAssessments, getPatientLinksByPatientId } from './patient-access';
import { ensureDemoSeeded } from './demo-data';

export interface AssessmentResult {
  id: string;
  userId: string;
  patientId: string;
  patientName: string;
  date: string;
  ensemble_probability: number;
  risk_level: string;
  confidence?: number;
  num_frames_processed?: number;
  source?: string;
  youtube_url?: string;
  video_name?: string;
  file_hash?: string;
  notes?: string;
  model_predictions: Record<string, { probability: number; risk_level: string }>;
  reviewed: boolean;
  shared: boolean;
  sharedNotes?: string;
  createdAt: Timestamp;
}

const DEMO_USER_IDS = ['demo-doctor', 'demo-parent'];

function isDemoUser(userId: string): boolean {
  return DEMO_USER_IDS.includes(userId);
}

function getDemoStorageKey(userId: string): string {
  return `demo_assessments_${userId}`;
}

function getDemoAssessments(userId: string): AssessmentResult[] {
  if (typeof window === 'undefined') return [];
  ensureDemoSeeded();
  try {
    return JSON.parse(localStorage.getItem(getDemoStorageKey(userId)) || '[]');
  } catch {
    return [];
  }
}

function saveDemoAssessments(userId: string, assessments: AssessmentResult[]): void {
  if (typeof window === 'undefined') return;
  localStorage.setItem(getDemoStorageKey(userId), JSON.stringify(assessments));
}

export async function saveAssessment(
  userId: string,
  data: Omit<AssessmentResult, 'id' | 'createdAt' | 'reviewed' | 'shared'>
): Promise<string> {
  if (isDemoUser(userId)) {
    const assessments = getDemoAssessments(userId);
    const id = 'demo-' + Date.now();
    const newAssessment: AssessmentResult = {
      ...data,
      id,
      reviewed: false,
      shared: false,
      createdAt: { toMillis: () => Date.now() } as any,
    };
    saveDemoAssessments(userId, [newAssessment, ...assessments]);
    return id;
  }
  const ref = await addDoc(collection(db, 'users', userId, 'assessments'), {
    ...data,
    reviewed: false,
    shared: false,
    createdAt: serverTimestamp(),
  });
  return ref.id;
}

export async function reviewAssessment(userId: string, assessmentId: string): Promise<void> {
  if (isDemoUser(userId)) {
    const assessments = getDemoAssessments(userId);
    const idx = assessments.findIndex(a => a.id === assessmentId);
    if (idx !== -1) {
      assessments[idx].reviewed = true;
      saveDemoAssessments(userId, assessments);
    }
    return;
  }
  await updateDoc(doc(db, 'users', userId, 'assessments', assessmentId), { reviewed: true });
}

export async function shareAssessment(userId: string, assessmentId: string, notes?: string, patientId?: string): Promise<void> {
  if (isDemoUser(userId)) {
    const assessments = getDemoAssessments(userId);
    const idx = assessments.findIndex(a => a.id === assessmentId);
    if (idx !== -1) {
      assessments[idx].shared = true;
      assessments[idx].sharedNotes = notes || '';
      saveDemoAssessments(userId, assessments);
    }
    return;
  }
  
  // 1. تحديث التقييم نفسه
  await updateDoc(doc(db, 'users', userId, 'assessments', assessmentId), { shared: true, sharedNotes: notes || '' });

  // 2. الجزء الجديد: ربط التقييم بحساب الأهل
  if (patientId) {
    const links = await getPatientLinksByPatientId(patientId);
    
    for (const link of links) {
      const currentShared = link.sharedAssessments || [];
      if (!currentShared.includes(assessmentId)) {
        await updateSharedAssessments(link.id, [...currentShared, assessmentId]);
      }
    }
  }
}

export async function getAssessmentById(userId: string, assessmentId: string): Promise<AssessmentResult | null> {
  if (isDemoUser(userId)) {
    const assessments = getDemoAssessments(userId);
    return assessments.find(a => a.id === assessmentId) || null;
  }
  const snap = await getDoc(doc(db, 'users', userId, 'assessments', assessmentId));
  if (!snap.exists()) return null;
  return { id: snap.id, ...snap.data() } as AssessmentResult;
}

export async function getAssessments(userId: string): Promise<AssessmentResult[]> {
  if (isDemoUser(userId)) {
    return getDemoAssessments(userId);
  }
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

export async function deleteAllAssessments(userId: string): Promise<void> {
  if (isDemoUser(userId)) {
    saveDemoAssessments(userId, []);
    return;
  }
  const q = query(collection(db, 'users', userId, 'assessments'));
  const snap = await getDocs(q);
  const promises = snap.docs.map((d) => deleteDoc(doc(db, 'users', userId, 'assessments', d.id)));
  await Promise.all(promises);
}

export async function getAssessmentsByPatient(userId: string, patientId: string): Promise<AssessmentResult[]> {
  if (isDemoUser(userId)) {
    const assessments = getDemoAssessments(userId);
    const filtered = assessments.filter(a => a.patientId === patientId);
    filtered.sort((a, b) => {
      const tA = (a.createdAt as any)?.toMillis?.() || 0;
      const tB = (b.createdAt as any)?.toMillis?.() || 0;
      return tB - tA;
    });
    return filtered;
  }
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

export async function updateAssessmentNotes(
  userId: string,
  assessmentId: string,
  notes: string
): Promise<void> {
  await updateDoc(doc(db, 'users', userId, 'assessments', assessmentId), { notes });
}

export async function checkDuplicateVideo(
  userId: string,
  patientId: string,
  videoName?: string,
  youtubeUrl?: string
): Promise<boolean> {
  if (!videoName && !youtubeUrl) return false;
  const assessments = await getAssessmentsByPatient(userId, patientId);
  return assessments.some((a) => {
    if (youtubeUrl && a.youtube_url && a.youtube_url === youtubeUrl) return true;
    if (videoName && a.video_name && a.video_name === videoName) return true;
    return false;
  });
}
