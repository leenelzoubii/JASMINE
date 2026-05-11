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
import { hashPassword, generateTempPassword, setTempPassword, clearTempPassword } from '@/lib/password';
import { sendParentCredentials } from '@/lib/emails/service';

export interface ParentAccount {
  id: string;
  email: string;
  name: string;
  password?: string;
  tempPassword?: string | null;
  tempPasswordExpires?: Date | null;
  mustChangePassword: boolean;
  isActive: boolean;
  createdAt: unknown;
  createdBy: string;
}

export interface ParentAccountInput {
  email: string;
  name: string;
  createdBy: string;
}

export interface CreateParentResult {
  success: boolean;
  parent?: ParentAccount;
  tempPassword?: string;
  isNew?: boolean;
  error?: string;
}

const testMode = process.env.EMAIL_TEST_MODE === 'true';

export async function createOrGetParentAccount(
  professionalId: string,
  data: ParentAccountInput,
  childName: string
): Promise<CreateParentResult> {
  try {
    // TEST MODE: Skip Firestore, always send email
    if (testMode) {
      console.log('[ParentAccounts] TEST MODE: Skipping Firestore, sending test email');

      const tempPassword = generateTempPassword();
      await sendParentCredentials(data.email, data.name, childName, tempPassword);

      return {
        success: true,
        parent: {
          id: 'test-' + Date.now(),
          email: data.email.toLowerCase(),
          name: data.name,
          mustChangePassword: true,
          isActive: true,
          createdAt: null,
          createdBy: professionalId,
        },
        tempPassword,
        isNew: true,
      };
    }

    // NORMAL MODE: Use Firestore
    const accountsRef = collection(db, 'parent_accounts');
    const q = query(accountsRef, where('email', '==', data.email.toLowerCase()));
    const existingSnap = await getDocs(q);

    if (!existingSnap.empty) {
      const existingDoc = existingSnap.docs[0];
      const existingData = existingDoc.data();
      return {
        success: true,
        parent: { id: existingDoc.id, ...existingData } as ParentAccount,
        isNew: false,
      };
    }

    const tempPassword = generateTempPassword();
    const hashedTempPassword = await hashPassword(tempPassword);
    const expiresAt = new Date();
    expiresAt.setHours(expiresAt.getHours() + 24);

    const newAccount: Omit<ParentAccount, 'id'> = {
      email: data.email.toLowerCase(),
      name: data.name,
      password: hashedTempPassword,
      tempPassword: hashedTempPassword,
      tempPasswordExpires: expiresAt,
      mustChangePassword: true,
      isActive: true,
      createdAt: serverTimestamp(),
      createdBy: professionalId,
    };

    const docRef = await addDoc(accountsRef, newAccount);

    await sendParentCredentials(data.email, data.name, childName, tempPassword);

    return {
      success: true,
      parent: { id: docRef.id, ...newAccount },
      tempPassword,
      isNew: true,
    };
  } catch (err) {
    console.error('[ParentAccounts] Error:', err);
    return { success: false, error: 'Failed to create parent account' };
  }
}

export async function getParentAccountById(parentId: string): Promise<ParentAccount | null> {
  try {
    const docRef = doc(db, 'parent_accounts', parentId);
    const docSnap = await getDoc(docRef);
    if (!docSnap.exists()) return null;
    return { id: docSnap.id, ...docSnap.data() } as ParentAccount;
  } catch (err) {
    console.error('[ParentAccounts] Error getting parent:', err);
    return null;
  }
}

export async function getParentAccountByEmail(email: string): Promise<ParentAccount | null> {
  try {
    const accountsRef = collection(db, 'parent_accounts');
    const q = query(accountsRef, where('email', '==', email.toLowerCase()));
    const docSnap = await getDocs(q);
    if (docSnap.empty) return null;
    return { id: docSnap.docs[0].id, ...docSnap.docs[0].data() } as ParentAccount;
  } catch (err) {
    console.error('[ParentAccounts] Error:', err);
    return null;
  }
}

export async function resendParentCredentials(
  parentId: string,
  childName: string
): Promise<{ success: boolean; error?: string }> {
  try {
    // TEST MODE: Skip Firestore, just send email
    if (testMode) {
      console.log('[ParentAccounts] TEST MODE: Skipping Firestore for resend');

      const parent = await getParentAccountByEmail('test@example.com');
      const tempPassword = generateTempPassword();

      // Use a test email for resend
      await sendParentCredentials('resend-test@example.com', 'Test Parent', childName, tempPassword);

      return { success: true };
    }

    // NORMAL MODE: Use Firestore
    const parent = await getParentAccountById(parentId);
    if (!parent) return { success: false, error: 'Parent account not found' };

    const tempPassword = generateTempPassword();
    const hashedTempPassword = await hashPassword(tempPassword);
    const expiresAt = new Date();
    expiresAt.setHours(expiresAt.getHours() + 24);

    const parentRef = doc(db, 'parent_accounts', parentId);
    await updateDoc(parentRef, {
      tempPassword: hashedTempPassword,
      tempPasswordExpires: expiresAt,
      mustChangePassword: true,
      updatedAt: serverTimestamp(),
    });

    await sendParentCredentials(parent.email, parent.name, childName, tempPassword);

    return { success: true };
  } catch (err) {
    console.error('[ParentAccounts] Error resending:', err);
    return { success: false, error: 'Failed to resend credentials' };
  }
}

export async function deactivateParentAccount(
  parentId: string
): Promise<{ success: boolean; error?: string }> {
  try {
    const parentRef = doc(db, 'parent_accounts', parentId);
    await updateDoc(parentRef, {
      isActive: false,
      updatedAt: serverTimestamp(),
    });
    return { success: true };
  } catch (err) {
    console.error('[ParentAccounts] Error deactivating:', err);
    return { success: false, error: 'Failed to deactivate account' };
  }
}

export async function reactivateParentAccount(
  parentId: string
): Promise<{ success: boolean; error?: string }> {
  try {
    const parentRef = doc(db, 'parent_accounts', parentId);
    await updateDoc(parentRef, {
      isActive: true,
      updatedAt: serverTimestamp(),
    });
    return { success: true };
  } catch (err) {
    console.error('[ParentAccounts] Error reactivating:', err);
    return { success: false, error: 'Failed to reactivate account' };
  }
}

export async function getAllParentAccounts(
  professionalId: string
): Promise<ParentAccount[]> {
  try {
    const accountsRef = collection(db, 'parent_accounts');
    const q = query(accountsRef, where('createdBy', '==', professionalId));
    const docSnap = await getDocs(q);
    return docSnap.docs.map(d => ({ id: d.id, ...d.data() } as ParentAccount));
  } catch (err) {
    console.error('[ParentAccounts] Error getting all:', err);
    return [];
  }
}
