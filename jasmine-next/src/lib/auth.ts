/**
 * Authentication utilities for JASMINE
 * using Firebase database to store users registration information
 */

import {
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signOut,
} from "firebase/auth";

import {
  doc,
  setDoc,
  getDoc,
  serverTimestamp,
} from "firebase/firestore";

import { auth, db } from "@/lib/firebase";

export interface User {
  id: string;
  name: string;
  email: string;
  role: "parent" | "professional";
  child?: {
    name: string;
    age: number;
    specialist?: string;
  };
  specialty?: string;
}

export async function registerUser(
  name: string,
  email: string,
  password: string,
  role: "parent" | "professional"
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
          specialty: "Autism Specialist",
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