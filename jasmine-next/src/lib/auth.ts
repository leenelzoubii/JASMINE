/**
 * Authentication utilities for JASMINE
 * Uses Firebase for auth + Firestore for user data storage
 */

import {
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signOut,
  EmailAuthProvider,
  reauthenticateWithCredential,
  updatePassword,
  sendPasswordResetEmail,
  deleteUser,
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

// Demo accounts fallback - works even if Firebase auth fails
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

  let firebaseUser;
  try {
    const userCredential = await createUserWithEmailAndPassword(
      auth,
      cleanEmail,
      cleanPassword
    );
    firebaseUser = userCredential.user;
  } catch (err: unknown) {
    const error = err as { code?: string; message?: string };
    if (error.code === "auth/email-already-in-use") {
      throw new Error(
        "This email is already registered. Please log in instead."
      );
    }
    if (error.code === "auth/weak-password") {
      throw new Error("Password should be at least 6 characters.");
    }
    if (error.code === "auth/invalid-email") {
      throw new Error("Please enter a valid email address.");
    }
    throw new Error("Registration failed. Please try again.");
  }

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

  try {
    await setDoc(doc(db, "users", firebaseUser.uid), {
      ...userData,
      createdAt: serverTimestamp(),
    });
  } catch {
    // Firestore write failed - undo Auth account creation
    try {
      await deleteUser(firebaseUser);
    } catch {
      // Cleanup failed - user exists in Auth but not Firestore
      // They can still login (we handle this edge case in authenticateUser)
    }
    throw new Error(
      "Failed to create user profile. Please try again."
    );
  }

  if (typeof window !== "undefined") {
    localStorage.setItem("currentUser", JSON.stringify(userData));
  }

  return userData;
}

export async function authenticateUser(
  email: string,
  password: string
): Promise<User | null> {
  const cleanEmail = email.trim().toLowerCase();
  const cleanPassword = password.trim();

  let firebaseUser;
  try {
    const userCredential = await signInWithEmailAndPassword(
      auth,
      cleanEmail,
      cleanPassword
    );
    firebaseUser = userCredential.user;
  } catch (err: unknown) {
    const error = err as { code?: string; message?: string };
    console.error("Firebase auth error:", error.code, error.message);
    
    if (
      error.code === "auth/invalid-credential" ||
      error.code === "auth/user-not-found" ||
      error.code === "auth/wrong-password"
    ) {
      // Fallback: check if this is a demo account
      const user = demoUsers[cleanEmail];
      if (user) {
        localStorage.setItem("currentUser", JSON.stringify(user));
        return user;
      }
      throw new Error("No account found with this email. Please create an account first.");
    }
    if (error.code === "auth/invalid-email") {
      throw new Error("Please enter a valid email address.");
    }
    if (error.code === "auth/too-many-requests") {
      throw new Error("Too many login attempts. Please try again later.");
    }
    throw new Error("Login failed. Please try again.");
  }

  // Fetch user data from Firestore
  const userRef = doc(db, "users", firebaseUser.uid);
  const userSnap = await getDoc(userRef);

  if (!userSnap.exists()) {
    // Edge case: Auth account exists but Firestore doc is missing
    // Re-create the Firestore doc on-the-fly
    const userData: User = {
      id: firebaseUser.uid,
      name: firebaseUser.displayName || firebaseUser.email?.split("@")[0] || "User",
      email: cleanEmail,
      role: "professional",
      specialty: "General",
    };

    await setDoc(userRef, {
      ...userData,
      createdAt: serverTimestamp(),
    });

    if (typeof window !== "undefined") {
      localStorage.setItem("currentUser", JSON.stringify(userData));
    }

    return userData;
  }

  const userData = userSnap.data() as User;

  if (typeof window !== "undefined") {
    localStorage.setItem("currentUser", JSON.stringify(userData));
  }

  return userData;
}

export function getCurrentUser(): User | null {
  if (typeof window === "undefined") return null;
  const storedUser = localStorage.getItem("currentUser");
  if (!storedUser) return null;
  try {
    return JSON.parse(storedUser);
  } catch {
    return null;
  }
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
