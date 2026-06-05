import {
  collection,
  doc,
  addDoc,
  getDocs,
  deleteDoc,
  updateDoc,
  getDoc,
  serverTimestamp,
  query,
  orderBy,
} from 'firebase/firestore';
import { db } from '@/lib/firebase';

export interface Patient {
  id: string;
  name: string;
  dob: string;
  parentName: string;
  email: string;
  phone: string;
  lastVisit: string;
  risk: string;
  createdAt?: unknown;
  updatedAt?: unknown;
}

function getPatientsRef(userId: string) {
  return collection(db, 'users', userId, 'patients');
}

export async function getPatients(userId: string): Promise<Patient[]> {
  const q = query(getPatientsRef(userId), orderBy('createdAt', 'desc'));
  const snap = await getDocs(q);
  return snap.docs.map((d) => ({ id: d.id, ...d.data() } as Patient));
}

export async function getPatient(userId: string, patientId: string): Promise<Patient | null> {
  const snap = await getDoc(doc(db, 'users', userId, 'patients', patientId));
  if (!snap.exists()) return null;
  return { id: snap.id, ...snap.data() } as Patient;
}

export async function addPatient(
  userId: string,
  data: Omit<Patient, 'id' | 'createdAt' | 'updatedAt'>
): Promise<Patient> {
  const ref = await addDoc(getPatientsRef(userId), {
    ...data,
    createdAt: serverTimestamp(),
    updatedAt: serverTimestamp(),
  });
  return { id: ref.id, ...data };
}

export async function updatePatient(
  userId: string,
  patientId: string,
  data: Partial<Omit<Patient, 'id' | 'createdAt'>>
): Promise<void> {
  await updateDoc(doc(db, 'users', userId, 'patients', patientId), {
    ...data,
    updatedAt: serverTimestamp(),
  });
}

export async function deletePatient(userId: string, patientId: string): Promise<void> {
  await deleteDoc(doc(db, 'users', userId, 'patients', patientId));
}