/**
 * Real messaging system using Firestore
 */
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
  serverTimestamp,
  Timestamp,
} from "firebase/firestore";
import { db } from "@/lib/firebase";

export interface Message {
  id: string;
  conversationId: string;
  senderId: string;
  receiverId: string;
  text: string;
  createdAt: Timestamp;
  read: boolean;
}

export async function sendMessage(
  senderId: string,
  receiverId: string,
  text: string
): Promise<void> {
  const conversationId = [senderId, receiverId].sort().join("_");

  await addDoc(collection(db, "messages"), {
    conversationId,
    senderId,
    receiverId,
    text,
    read: false,
    createdAt: serverTimestamp(),
  });

  // Upsert conversation metadata
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

export function subscribeToMessages(
  userId1: string,
  userId2: string,
  callback: (messages: Message[]) => void
) {
  const conversationId = [userId1, userId2].sort().join("_");
  // No orderBy to avoid composite index - sort client side
  const q = query(
    collection(db, "messages"),
    where("conversationId", "==", conversationId)
  );

  return onSnapshot(q, (snapshot) => {
    const msgs = snapshot.docs.map((d) => ({ id: d.id, ...d.data() } as Message));
    msgs.sort((a, b) => {
      const tA = (a.createdAt as any)?.toMillis?.() || 0;
      const tB = (b.createdAt as any)?.toMillis?.() || 0;
      return tA - tB; // oldest first
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
