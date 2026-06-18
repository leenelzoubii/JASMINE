/**
 * Authentication utilities for JASMINE
 * using Firebase database to store users registration information
 */

import {
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signOut,
  EmailAuthProvider,
  reauthenticateWithCredential,
  updatePassword,
  sendPasswordResetEmail,
} from "firebase/auth";

import {
  doc,
  setDoc,
  getDoc,
  getDocs,
  updateDoc,
  collection,
  query,
  where,
  serverTimestamp,
} from "firebase/firestore";

import { auth, db } from "@/lib/firebase";
import { hashPassword, verifyPassword } from "@/lib/password";

export interface User {
  id: string;
  name: string;
  email: string;
  role: "parent" | "professional";
  phone?: string;
  child?: {
    name: string;
    age: number;
    specialist?: string;
  };
  specialty?: string;
}

// Demo accounts fallback - works even if Firebase is unavailable
const demoUsers: Record<string, User> = {
  "parent@demo.com": {
    id: "demo-parent",
    name: "John Parent",
    email: "parent@demo.com",
    role: "parent",
    child: { name: "Emma", age: 6, specialist: "Dr. Jasmine" },
  },
  "doctor@demo.com": {
    id: "demo-doctor",
    name: "Dr. Jasmine",
    email: "doctor@demo.com",
    role: "professional",
    specialty: "Pediatric Specialist",
  },
};

// Local user registry - stores ALL created accounts locally as backup
function getLocalUsers(): Record<string, { user: User; password: string }> {
  if (typeof window === "undefined") return {};
  try {
    return JSON.parse(localStorage.getItem("localUsers") || "{}");
  } catch {
    return {};
  }
}

function saveLocalUser(email: string, password: string, user: User): void {
  if (typeof window === "undefined") return;
  const users = getLocalUsers();
  users[email.toLowerCase().trim()] = { user, password };
  localStorage.setItem("localUsers", JSON.stringify(users));
}

export async function registerUser(
  name: string,
  email: string,
  password: string,
  role: "parent" | "professional",
  specialty?: string
): Promise<User> {
  const cleanName = name.trim();
  const cleanEmail = email.trim().toLowerCase();
  const cleanPassword = password.trim();

  // Also save locally so user can login even if Firebase has issues
  const localUser: User = {
    id: "local-" + Date.now(),
    name: cleanName,
    email: cleanEmail,
    role,
    ...(role === "parent"
      ? { child: { name: name.split(" ")[0] + "'s Child", age: 0, specialist: "Dr. Jasmine" } }
      : { specialty: specialty?.trim() || "Autism Specialist" }),
  };
  saveLocalUser(cleanEmail, cleanPassword, localUser);

  try {
    const userCredential = await createUserWithEmailAndPassword(
      auth,
      cleanEmail,
      cleanPassword
    );

    const firebaseUser = userCredential.user;

    const userData: User = {
      id: firebaseUser.uid,
      name: cleanName,
      email: cleanEmail,
      role,
      ...(role === "parent"
        ? { child: { name: "Emma", age: 6, specialist: "Dr. Jasmine" } }
        : { specialty: specialty?.trim() || "Autism Specialist" }),
    };

    await setDoc(doc(db, "users", firebaseUser.uid), {
      ...userData,
      createdAt: serverTimestamp(),
    });

    if (typeof window !== "undefined") {
      localStorage.setItem("currentUser", JSON.stringify(userData));
    }

    return userData;
  } catch (err) {
    // Firebase failed - use local user as fallback
    console.warn("Firebase registration failed, using local fallback:", err);
    if (typeof window !== "undefined") {
      localStorage.setItem("currentUser", JSON.stringify(localUser));
    }
    return localUser;
  }
}

export async function authenticateUser(
  email: string,
  password: string
): Promise<User | null> {
  const cleanEmail = email.trim().toLowerCase();
  const cleanPassword = password.trim();

  try {
    const userCredential = await signInWithEmailAndPassword(
      auth,
      cleanEmail,
      cleanPassword
    );

    const firebaseUser = userCredential.user;
    const userRef = doc(db, "users", firebaseUser.uid);
    const userSnap = await getDoc(userRef);

    if (!userSnap.exists()) {
      throw new Error(
        "User exists in Authentication but not in Firestore users collection"
      );
    }

    const userData = userSnap.data() as User;

    if (typeof window !== "undefined") {
      localStorage.setItem("currentUser", JSON.stringify(userData));
    }

    return userData;
  } catch (err: unknown) {
    const error = err as { code?: string; message?: string };
    console.warn("Firebase login failed, checking local fallback:", error.code);
    
    // Fallback 1: Check locally registered users
    const localUsers = getLocalUsers();
    const localEntry = localUsers[cleanEmail];
    if (localEntry && localEntry.password === cleanPassword) {
      if (typeof window !== "undefined") {
        localStorage.setItem("currentUser", JSON.stringify(localEntry.user));
      }
      return localEntry.user;
    }

    // Fallback 2: Check demo accounts
    const demoUser = demoUsers[cleanEmail];
    if (demoUser) {
      if (typeof window !== "undefined") {
        localStorage.setItem("currentUser", JSON.stringify(demoUser));
      }
      return demoUser;
    }

    // Fallback 3: Check parent_accounts collection
    try {
      const accountsRef = collection(db, "parent_accounts");
      const q = query(accountsRef, where("email", "==", cleanEmail));
      const snap = await getDocs(q);
      if (!snap.empty) {
        const accountData = snap.docs[0].data();
        if (accountData.isActive === false) {
          throw new Error("Parent account has been deactivated.");
        }
        const storedHash = accountData.password || accountData.tempPassword;
        if (storedHash) {
          const valid = await verifyPassword(cleanPassword, storedHash);
          if (valid) {
            const parentUser: User = {
              id: snap.docs[0].id,
              name: accountData.name || "Parent",
              email: cleanEmail,
              role: "parent",
              child: accountData.childName
                ? { name: accountData.childName, age: 0 }
                : undefined,
            };
            if (typeof window !== "undefined") {
              localStorage.setItem("currentUser", JSON.stringify(parentUser));
            }
            return parentUser;
          }
        }
      }
    } catch (parentErr) {
      console.warn("Parent account check failed:", parentErr);
    }
    
    // Rethrow the original error
    throw err;
  }
}

export function getCurrentUser(): User | null {
  if (typeof window === "undefined") return null;
  const storedUser = localStorage.getItem("currentUser");
  if (!storedUser) return null;
  return JSON.parse(storedUser);
}

export async function logoutUser(): Promise<void> {
  await signOut(auth);
  localStorage.removeItem("currentUser");
}

export async function updateUser(
  userId: string,
  data: { name?: string; phone?: string; specialty?: string }
): Promise<User> {
  const { getFirestore, updateDoc } = await import("firebase/firestore");
  const userRef = doc(db, "users", userId);
  await updateDoc(userRef, { ...data, updatedAt: serverTimestamp() });
  const snap = await getDoc(userRef);
  const updated = snap.data() as User;
  if (typeof window !== "undefined") {
    localStorage.setItem("currentUser", JSON.stringify(updated));
  }
  return updated;
}

export async function changeCurrentUserPassword(
  currentPassword: string,
  newPassword: string
): Promise<void> {
  const user = auth.currentUser;
  if (!user || !user.email) {
    throw new Error("No authenticated user found");
  }
  const credential = EmailAuthProvider.credential(
    user.email,
    currentPassword
  );
  await reauthenticateWithCredential(user, credential);
  await updatePassword(user, newPassword);

  // Sync new password hash to Firestore so the parent_accounts fallback
  // and users doc verification work after password change.
  try {
    const hashedPassword = await hashPassword(newPassword);

    const userRef = doc(db, "users", user.uid);
    const userSnap = await getDoc(userRef);
    if (userSnap.exists()) {
      await updateDoc(userRef, { password: hashedPassword, updatedAt: serverTimestamp() });
    } else {
      const accountsRef = collection(db, "parent_accounts");
      const q = query(accountsRef, where("email", "==", user.email));
      const snap = await getDocs(q);
      if (!snap.empty) {
        const accountRef = doc(db, "parent_accounts", snap.docs[0].id);
        await updateDoc(accountRef, { password: hashedPassword, updatedAt: serverTimestamp() });
      }
    }
  } catch (syncErr) {
    console.warn("Failed to sync password to Firestore:", syncErr);
  }

  // Update local storage fallback
  try {
    if (typeof window !== "undefined") {
      const localUsers = getLocalUsers();
      const cleanEmail = user.email.toLowerCase().trim();
      if (localUsers[cleanEmail]) {
        localUsers[cleanEmail].password = newPassword;
        localStorage.setItem("localUsers", JSON.stringify(localUsers));
      }
    }
  } catch (localErr) {
    console.warn("Failed to update local password fallback:", localErr);
  }
}

export async function sendFirebaseResetPasswordEmail(email: string): Promise<void> {
  const cleanEmail = email.trim().toLowerCase();
  if (!cleanEmail) {
    throw new Error("Email is required");
  }
  await sendPasswordResetEmail(auth, cleanEmail);
}
