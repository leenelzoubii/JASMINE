import { initializeApp, getApps, cert } from "firebase-admin/app";

let initialized = false;

export function initAdmin() {
  if (initialized) return;
  
  const apps = getApps();
  if (apps.length > 0) {
    initialized = true;
    return;
  }

  const privateKey = process.env.FIREBASE_PRIVATE_KEY?.replace(/\\n/g, "\n");
  
  if (!privateKey || !process.env.FIREBASE_CLIENT_EMAIL || !process.env.FIREBASE_PROJECT_ID) {
    console.warn(
      "Firebase Admin not configured. Set FIREBASE_PROJECT_ID, FIREBASE_CLIENT_EMAIL, and FIREBASE_PRIVATE_KEY."
    );
    return;
  }

  initializeApp({
    credential: cert({
      projectId: process.env.FIREBASE_PROJECT_ID,
      clientEmail: process.env.FIREBASE_CLIENT_EMAIL,
      privateKey,
    }),
  });

  initialized = true;
}
