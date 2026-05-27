import {
  collection,
  doc,
  addDoc,
  getDoc,
  getDocs,
  updateDoc,
  deleteDoc,
  query,
  where,
  serverTimestamp,
} from 'firebase/firestore';
import { db } from '@/lib/firebase';
import { createOrGetParentAccount } from './parent-accounts';

export interface PatientAccessLink {
  id: string;
  patientId: string;
  patientName: string;
  professionalId: string;
  professionalName?: string;
  parentId: string;
  parentEmail: string;
  parentName: string;
  accessGranted: boolean;
  accessGrantedAt: unknown;
  accessRevokedAt: unknown | null;
  sharedAssessments: string[];
  createdAt: unknown;
}

export interface PatientAccessInput {
  patientId: string;
  patientName: string;
  professionalId: string;
  professionalName?: string;
  parentName: string;
  parentEmail: string;
}

export interface CreateAccessResult {
  success: boolean;
  link?: PatientAccessLink;
  parentTempPassword?: string;
  error?: string;
}

export async function createPatientAccess(
  data: PatientAccessInput
): Promise<CreateAccessResult> {
  try {
    const parentResult = await createOrGetParentAccount(
      data.professionalId,
      { email: data.parentEmail, name: data.parentName, createdBy: data.professionalId },
      data.patientName
    );

    if (!parentResult.success || !parentResult.parent) {
      return { success: false, error: parentResult.error || 'Failed to create parent account' };
    }

    const linksRef = collection(db, 'patient_access_links');
    const existingQ = query(
      linksRef,
      where('patientId', '==', data.patientId),
      where('parentId', '==', parentResult.parent.id)
    );
    const existingSnap = await getDocs(existingQ);

    if (!existingSnap.empty) {
      const existingLink = existingSnap.docs[0];
      const linkData = existingLink.data();
      if (linkData.accessGranted) {
        return { success: true, link: { id: existingLink.id, ...linkData } as PatientAccessLink };
      }
      await updateDoc(doc(db, 'patient_access_links', existingLink.id), {
        accessGranted: true,
        accessGrantedAt: serverTimestamp(),
        accessRevokedAt: null,
      });
      return { success: true, link: { id: existingLink.id, ...linkData } as PatientAccessLink };
    }

    const newLink: Omit<PatientAccessLink, 'id'> = {
      patientId: data.patientId,
      patientName: data.patientName,
      professionalId: data.professionalId,
      professionalName: data.professionalName || 'Specialist',
      parentId: parentResult.parent.id,
      parentEmail: data.parentEmail.toLowerCase(),
      parentName: data.parentName,
      accessGranted: true,
      accessGrantedAt: serverTimestamp(),
      accessRevokedAt: null,
      sharedAssessments: [],
      createdAt: serverTimestamp(),
    };

    const docRef = await addDoc(linksRef, newLink);

    return {
      success: true,
      link: { id: docRef.id, ...newLink },
      parentTempPassword: parentResult.isNew ? parentResult.tempPassword : undefined,
    };
  } catch (err) {
    console.error('[PatientAccess] Error:', err);
    return { success: false, error: 'Failed to create patient access' };
  }
}

export async function getPatientLinksByProfessional(
  professionalId: string
): Promise<PatientAccessLink[]> {
  try {
    const linksRef = collection(db, 'patient_access_links');
    const q = query(linksRef, where('professionalId', '==', professionalId));
    const docSnap = await getDocs(q);
    return docSnap.docs.map(d => ({ id: d.id, ...d.data() } as PatientAccessLink));
  } catch (err) {
    console.error('[PatientAccess] Error:', err);
    return [];
  }
}

export async function getPatientLinksByParent(
  parentId: string
): Promise<PatientAccessLink[]> {
  try {
    const linksRef = collection(db, 'patient_access_links');
    const q = query(linksRef, where('parentId', '==', parentId), where('accessGranted', '==', true));
    const docSnap = await getDocs(q);
    return docSnap.docs.map(d => ({ id: d.id, ...d.data() } as PatientAccessLink));
  } catch (err) {
    console.error('[PatientAccess] Error:', err);
    return [];
  }
}

export async function getPatientLinksByPatientId(
  patientId: string
): Promise<PatientAccessLink[]> {
  try {
    const linksRef = collection(db, 'patient_access_links');
    const q = query(linksRef, where('patientId', '==', patientId), where('accessGranted', '==', true));
    const docSnap = await getDocs(q);
    return docSnap.docs.map(d => ({ id: d.id, ...d.data() } as PatientAccessLink));
  } catch (err) {
    console.error('[PatientAccess] Error:', err);
    return [];
  }
}

export async function revokePatientAccess(
  linkId: string
): Promise<{ success: boolean; error?: string }> {
  try {
    const linkRef = doc(db, 'patient_access_links', linkId);
    await updateDoc(linkRef, {
      accessGranted: false,
      accessRevokedAt: serverTimestamp(),
    });
    return { success: true };
  } catch (err) {
    console.error('[PatientAccess] Error revoking:', err);
    return { success: false, error: 'Failed to revoke access' };
  }
}

export async function grantPatientAccess(
  linkId: string
): Promise<{ success: boolean; error?: string }> {
  try {
    const linkRef = doc(db, 'patient_access_links', linkId);
    await updateDoc(linkRef, {
      accessGranted: true,
      accessGrantedAt: serverTimestamp(),
      accessRevokedAt: null,
    });
    return { success: true };
  } catch (err) {
    console.error('[PatientAccess] Error granting:', err);
    return { success: false, error: 'Failed to grant access' };
  }
}

export async function updateSharedAssessments(
  linkId: string,
  assessmentIds: string[]
): Promise<{ success: boolean; error?: string }> {
  try {
    const linkRef = doc(db, 'patient_access_links', linkId);
    await updateDoc(linkRef, {
      sharedAssessments: assessmentIds,
    });
    return { success: true };
  } catch (err) {
    console.error('[PatientAccess] Error updating assessments:', err);
    return { success: false, error: 'Failed to update shared assessments' };
  }
}

export async function getAccessLinkById(
  linkId: string
): Promise<PatientAccessLink | null> {
  try {
    const linkRef = doc(db, 'patient_access_links', linkId);
    const docSnap = await getDoc(linkRef);
    if (!docSnap.exists()) return null;
    return { id: docSnap.id, ...docSnap.data() } as PatientAccessLink;
  } catch (err) {
    console.error('[PatientAccess] Error:', err);
    return null;
  }
}
