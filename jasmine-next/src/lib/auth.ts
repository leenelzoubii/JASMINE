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
  updateDoc,
  serverTimestamp,
} from "firebase/firestore";

import { auth, db } from "@/lib/firebase";

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
  updatedAt?: unknown;
}

export async function registerUser(
  name: string,
  email: string,
  password: string,
  role: "parent" | "professional",
  specialty?: string
): Promise<User> {
  const cleanName = name.trim();
  const cleanEmail = email.trim();
  const cleanPassword = password.trim();

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
      ? {
          child: {
            name: "Emma",
            age: 6,
            specialist: "Dr. Jasmine",
          },
        }
      : {
          specialty: specialty?.trim() || "Autism Specialist",
        }),
  };

  await setDoc(doc(db, "users", firebaseUser.uid), {
    ...userData,
    createdAt: serverTimestamp(),
  });

  localStorage.setItem("currentUser", JSON.stringify(userData));

  return userData;
}

export async function authenticateUser(
  email: string,
  password: string
): Promise<User | null> {
  const cleanEmail = email.trim();
  const cleanPassword = password.trim();

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

  localStorage.setItem("currentUser", JSON.stringify(userData));

  return userData;
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
  const userRef = doc(db, "users", userId);
  await updateDoc(userRef, {
    ...data,
    updatedAt: serverTimestamp(),
  });
  const snap = await getDoc(userRef);
  const updated = snap.data() as User;
  localStorage.setItem("currentUser", JSON.stringify(updated));
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
}
export async function sendFirebaseResetPasswordEmail(email: string): Promise<void> {
  const cleanEmail = email.trim();

  if (!cleanEmail) {
    throw new Error("Email is required");
  }

  await sendPasswordResetEmail(auth, cleanEmail);
}