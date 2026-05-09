import {
  collection,
  doc,
  addDoc,
  getDocs,
  deleteDoc,
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

export async function deletePatient(userId: string, patientId: string): Promise<void> {
  await deleteDoc(doc(db, 'users', userId, 'patients', patientId));
}