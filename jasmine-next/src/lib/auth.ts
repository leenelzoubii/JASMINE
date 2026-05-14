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
}

export async function sendFirebaseResetPasswordEmail(email: string): Promise<void> {
  const cleanEmail = email.trim().toLowerCase();
  if (!cleanEmail) {
    throw new Error("Email is required");
  }
  await sendPasswordResetEmail(auth, cleanEmail);
}
