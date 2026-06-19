import {
  collection,
  addDoc,
  getDocs,
  getDoc,
  doc,
  updateDoc,
  setDoc,
  query,
  where,
  onSnapshot,
  writeBatch,
  serverTimestamp,
  Timestamp,
} from "firebase/firestore";
import { db } from "@/lib/firebase";
import { ensureDemoSeeded } from "./demo-data";

export interface Message {
  id: string;
  conversationId: string;
  senderId: string;
  receiverId: string;
  text: string;
  createdAt: Timestamp | { toMillis: () => number };
  read: boolean;
  status: "sent" | "delivered" | "read";
}

const DEMO_USER_IDS = ['demo-doctor', 'demo-parent'];

function isDemoUser(userId: string): boolean {
  return DEMO_USER_IDS.includes(userId);
}

function getDemoMessagesKey(conversationId: string): string {
  return `demo_messages_${conversationId}`;
}

function getDemoMessages(conversationId: string): Message[] {
  if (typeof window === 'undefined') return [];
  ensureDemoSeeded();
  try {
    return JSON.parse(localStorage.getItem(getDemoMessagesKey(conversationId)) || '[]');
  } catch {
    return [];
  }
}

function saveDemoMessages(conversationId: string, messages: Message[]): void {
  if (typeof window === 'undefined') return;
  localStorage.setItem(getDemoMessagesKey(conversationId), JSON.stringify(messages));
}

export async function sendMessage(
  senderId: string,
  receiverId: string,
  text: string
): Promise<void> {
  const conversationId = [senderId, receiverId].sort().join("_");

  if (isDemoUser(senderId)) {
    const messages = getDemoMessages(conversationId);
    const newMsg: Message = {
      id: 'demo-msg-' + Date.now(),
      conversationId,
      senderId,
      receiverId,
      text,
      createdAt: { toMillis: () => Date.now() },
      read: false,
      status: "sent",
    };
    saveDemoMessages(conversationId, [...messages, newMsg]);
    return;
  }

  await addDoc(collection(db, "messages"), {
    conversationId,
    senderId,
    receiverId,
    text,
    read: false,
    status: "sent",
    createdAt: serverTimestamp(),
  });

  const convRef = doc(db, "conversations", conversationId);
  const convSnap = await getDoc(convRef);

  if (convSnap.exists()) {
    const data = convSnap.data();
    await updateDoc(convRef, {
      lastMessage: text,
      lastMessageTime: serverTimestamp(),
      [`unreadCount.${receiverId}`]: (data.unreadCount?.[receiverId] || 0) + 1,
    });
  } else {
    await setDoc(convRef, {
      participantIds: [senderId, receiverId],
      lastMessage: text,
      lastMessageTime: serverTimestamp(),
      unreadCount: { [receiverId]: 1 },
      createdAt: serverTimestamp(),
    });
  }
}

export async function markMessagesDelivered(
  currentUserId: string,
  otherUserId: string
): Promise<void> {
  if (isDemoUser(currentUserId)) return;
  const conversationId = [currentUserId, otherUserId].sort().join("_");
  const q = query(
    collection(db, "messages"),
    where("conversationId", "==", conversationId),
    where("receiverId", "==", currentUserId),
    where("status", "==", "sent")
  );
  const snap = await getDocs(q);
  if (snap.empty) return;
  const batch = writeBatch(db);
  snap.docs.forEach((d) => {
    batch.update(doc(db, "messages", d.id), { status: "delivered" });
  });
  await batch.commit();
}

export async function markConversationAsRead(
  currentUserId: string,
  otherUserId: string
): Promise<void> {
  if (isDemoUser(currentUserId)) return;
  const conversationId = [currentUserId, otherUserId].sort().join("_");

  const q = query(
    collection(db, "messages"),
    where("conversationId", "==", conversationId),
    where("receiverId", "==", currentUserId),
    where("read", "==", false)
  );
  const snap = await getDocs(q);
  if (!snap.empty) {
    const batch = writeBatch(db);
    snap.docs.forEach((d) => {
      batch.update(doc(db, "messages", d.id), { read: true, status: "read" });
    });
    await batch.commit();
  }

  const convRef = doc(db, "conversations", conversationId);
  const convSnap = await getDoc(convRef);
  if (convSnap.exists()) {
    const data = convSnap.data();
    if ((data.unreadCount?.[currentUserId] || 0) > 0) {
      await updateDoc(convRef, {
        [`unreadCount.${currentUserId}`]: 0,
      });
    }
  }
}

export function subscribeToMessages(
  userId1: string,
  userId2: string,
  callback: (messages: Message[]) => void
) {
  const conversationId = [userId1, userId2].sort().join("_");

  if (isDemoUser(userId1)) {
    const load = () => {
      const msgs = getDemoMessages(conversationId);
      msgs.sort((a, b) => {
        const tA = (a.createdAt as any)?.toMillis?.() || 0;
        const tB = (b.createdAt as any)?.toMillis?.() || 0;
        return tA - tB;
      });
      callback(msgs);
    };
    load();
    const interval = setInterval(load, 2000);
    return () => clearInterval(interval);
  }

  const q = query(
    collection(db, "messages"),
    where("conversationId", "==", conversationId)
  );

  return onSnapshot(q, (snapshot) => {
    const msgs = snapshot.docs.map((d) => ({ id: d.id, ...d.data() } as Message));
    msgs.sort((a, b) => {
      const tA = (a.createdAt as any)?.toMillis?.() || 0;
      const tB = (b.createdAt as any)?.toMillis?.() || 0;
      return tA - tB;
    });
    callback(msgs);
  });
}

export async function getUserConversations(userId: string): Promise<any[]> {
  const q = query(
    collection(db, "conversations"),
    where("participantIds", "array-contains", userId)
  );
  const snap = await getDocs(q);
  return snap.docs.map((d) => ({ id: d.id, ...d.data() }));
}
