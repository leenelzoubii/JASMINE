/**
 * Notification system using Firestore
 */
import {
  collection,
  addDoc,
  getDocs,
  query,
  where,
  onSnapshot,
  updateDoc,
  doc,
  serverTimestamp,
  Timestamp,
} from "firebase/firestore";
import { db } from "@/lib/firebase";

export interface Notification {
  id: string;
  userId: string;
  type: "patient_added" | "patient_removed" | "message" | "parent_request" | "request_accepted" | "assessment_complete";
  title: string;
  message: string;
  link?: string;
  read: boolean;
  createdAt: Timestamp;
}

export async function addNotification(data: {
  userId: string;
  type: Notification["type"];
  title: string;
  message: string;
  link?: string;
}): Promise<void> {
  await addDoc(collection(db, "notifications"), {
    ...data,
    read: false,
    createdAt: serverTimestamp(),
  });
}

export function subscribeToNotifications(
  userId: string,
  callback: (notifications: Notification[]) => void
) {
  // No orderBy to avoid needing composite index - sort client side
  const q = query(
    collection(db, "notifications"),
    where("userId", "==", userId)
  );

  return onSnapshot(q, (snapshot) => {
    const notifs = snapshot.docs.map((d) => ({ id: d.id, ...d.data() } as Notification));
    notifs.sort((a, b) => {
      const tA = (a.createdAt as any)?.toMillis?.() || 0;
      const tB = (b.createdAt as any)?.toMillis?.() || 0;
      return tB - tA; // newest first
    });
    callback(notifs.slice(0, 30));
  });
}

export async function markNotificationRead(notificationId: string): Promise<void> {
  await updateDoc(doc(db, "notifications", notificationId), { read: true });
}

export async function markAllNotificationsRead(userId: string): Promise<void> {
  const q = query(
    collection(db, "notifications"),
    where("userId", "==", userId),
    where("read", "==", false)
  );
  const snap = await getDocs(q);
  const updates = snap.docs.map((d) => updateDoc(doc(db, "notifications", d.id), { read: true }));
  await Promise.all(updates);
}

export function subscribeToUnreadCount(
  userId: string,
  callback: (count: number) => void
) {
  const q = query(
    collection(db, "notifications"),
    where("userId", "==", userId),
    where("read", "==", false)
  );

  return onSnapshot(q, (snapshot) => {
    callback(snapshot.size);
  });
}
