/**
 * Parent request system - "Friend request" between doctor and parent
 * Stored in Firestore under `parentRequests/{id}`
 */

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
} from "firebase/firestore";
import { db } from "@/lib/firebase";

export interface ParentRequest {
  id: string;
  professionalId: string;
  professionalName: string;
  patientId: string;
  patientName: string;
  parentEmail: string;
  parentId?: string;
  parentName: string;
  status: "pending" | "accepted" | "declined";
  createdAt: Timestamp;
}

const DEMO_USER_IDS = ['demo-doctor', 'demo-parent'];

function isDemoUser(userId: string): boolean {
  return DEMO_USER_IDS.includes(userId);
}

/**
 * Demo requests are stored under a single shared localStorage key
 * so both doctor and parent can read/write the same pool.
 */
function getDemoAllRequestsKey(): string {
  return 'demo_allRequests';
}

function getDemoAllRequests(): ParentRequest[] {
  if (typeof window === 'undefined') return [];
  try {
    return JSON.parse(localStorage.getItem(getDemoAllRequestsKey()) || '[]');
  } catch { return []; }
}

function saveDemoAllRequests(requests: ParentRequest[]): void {
  if (typeof window === 'undefined') return;
  localStorage.setItem(getDemoAllRequestsKey(), JSON.stringify(requests));
}

function sortByCreatedAtDesc(items: ParentRequest[]): ParentRequest[] {
  return items.sort((a, b) => {
    const tA = (a.createdAt as any)?.toMillis?.() || 0;
    const tB = (b.createdAt as any)?.toMillis?.() || 0;
    return tB - tA;
  });
}

export async function sendParentRequest(data: {
  professionalId: string;
  professionalName: string;
  patientId: string;
  patientName: string;
  parentEmail: string;
  parentName: string;
}): Promise<ParentRequest> {
  if (isDemoUser(data.professionalId)) {
    const all = getDemoAllRequests();
    const newReq: ParentRequest = {
      id: 'demo-req-' + Date.now(),
      professionalId: data.professionalId,
      professionalName: data.professionalName,
      patientId: data.patientId,
      patientName: data.patientName,
      parentEmail: data.parentEmail.toLowerCase().trim(),
      parentName: data.parentName,
      status: 'pending',
      createdAt: { toMillis: () => Date.now() } as any,
    };
    saveDemoAllRequests([newReq, ...all]);
    return newReq;
  }
  const docRef = await addDoc(collection(db, "parentRequests"), {
    professionalId: data.professionalId,
    professionalName: data.professionalName,
    patientId: data.patientId,
    patientName: data.patientName,
    parentEmail: data.parentEmail.toLowerCase().trim(),
    parentName: data.parentName,
    status: "pending",
    createdAt: serverTimestamp(),
  });

  const snap = await getDoc(docRef);
  return { id: docRef.id, ...snap.data() } as ParentRequest;
}

export async function getParentRequestsByEmail(email: string): Promise<ParentRequest[]> {
  // For demo parent, read from shared localStorage
  if (email === 'parent@demo.com') {
    return sortByCreatedAtDesc(getDemoAllRequests().filter(r =>
      r.parentEmail === email.toLowerCase().trim() && r.status === 'pending'
    ));
  }
  const q = query(
    collection(db, "parentRequests"),
    where("parentEmail", "==", email.toLowerCase().trim()),
    where("status", "==", "pending")
  );
  const snap = await getDocs(q);
  return sortByCreatedAtDesc(snap.docs.map((d) => ({ id: d.id, ...d.data() } as ParentRequest)));
}

export async function getProfessionalRequests(professionalId: string): Promise<ParentRequest[]> {
  if (isDemoUser(professionalId)) {
    return sortByCreatedAtDesc(getDemoAllRequests().filter(r => r.professionalId === professionalId));
  }
  const q = query(
    collection(db, "parentRequests"),
    where("professionalId", "==", professionalId)
  );
  const snap = await getDocs(q);
  return sortByCreatedAtDesc(snap.docs.map((d) => ({ id: d.id, ...d.data() } as ParentRequest)));
}

export async function acceptParentRequest(requestId: string, parentId: string): Promise<void> {
  const all = getDemoAllRequests();
  const demoReq = all.find(r => r.id === requestId);
  if (demoReq) {
    demoReq.status = 'accepted';
    demoReq.parentId = parentId;
    saveDemoAllRequests(all);
    return;
  }
  const reqRef = doc(db, "parentRequests", requestId);
  const reqSnap = await getDoc(reqRef);
  if (!reqSnap.exists()) throw new Error("Request not found");

  const request = reqSnap.data() as ParentRequest;
  await updateDoc(reqRef, { status: "accepted", parentId });

  await addDoc(collection(db, "connections"), {
    professionalId: request.professionalId,
    professionalName: request.professionalName,
    parentId,
    parentName: request.parentName,
    patientId: request.patientId,
    patientName: request.patientName,
    createdAt: serverTimestamp(),
  });
}

export async function declineParentRequest(requestId: string): Promise<void> {
  const all = getDemoAllRequests();
  const idx = all.findIndex(r => r.id === requestId);
  if (idx !== -1) {
    all.splice(idx, 1);
    saveDemoAllRequests(all);
    return;
  }
  await deleteDoc(doc(db, "parentRequests", requestId));
}

export async function getUserConnections(userId: string): Promise<any[]> {
  const all = getDemoAllRequests();
  if (all.length > 0) {
    return all
      .filter(r => (r.professionalId === userId || r.parentId === userId) && r.status === 'accepted')
      .map(r => ({
        id: r.id,
        professionalId: r.professionalId,
        professionalName: r.professionalName,
        parentId: r.parentId,
        parentName: r.parentName,
        patientId: r.patientId,
        patientName: r.patientName,
      }));
  }
  const q1 = query(collection(db, "connections"), where("professionalId", "==", userId));
  const q2 = query(collection(db, "connections"), where("parentId", "==", userId));
  const [s1, s2] = await Promise.all([getDocs(q1), getDocs(q2)]);
  return [...s1.docs, ...s2.docs].map((d) => ({ id: d.id, ...d.data() }));
}
