/**
 * Real messaging system using Firestore
 */
import {
  collection,
  addDoc,
  getDocs,
  query,
  where,
  orderBy,
  onSnapshot,
  serverTimestamp,
  Timestamp,
  limit,
} from "firebase/firestore";
import { db } from "@/lib/firebase";

export interface Message {
  id: string;
  senderId: string;
  receiverId: string;
  text: string;
  createdAt: Timestamp;
  read: boolean;
}

export interface Conversation {
  id: string;
  participantIds: string[];
  participantNames: Record<string, string>;
  participantRoles: Record<string, string>;
  lastMessage?: string;
  lastMessageTime?: Timestamp;
  unreadCount?: number;
}

/**
 * Send a message
 */
export async function sendMessage(
  senderId: string,
  receiverId: string,
  text: string
): Promise<void> {
  const conversationId = [senderId, receiverId].sort().join("_");

  // Add message
  await addDoc(collection(db, "messages"), {
    conversationId,
    senderId,
    receiverId,
    text,
    read: false,
    createdAt: serverTimestamp(),
  });

  // Update conversation metadata
  const convRef = doc(db, "conversations", conversationId);
  const convSnap = await getDoc(convRef);
  if (convSnap.exists()) {
    await updateDoc(convRef, {
      lastMessage: text,
      lastMessageTime: serverTimestamp(),
      [`unreadCount.${receiverId}`]: (convSnap.data().unreadCount?.[receiverId] || 0) + 1,
    });
  }
}

/**
 * Get messages for a conversation (between two users)
 */
export function subscribeToMessages(
  userId1: string,
  userId2: string,
  callback: (messages: Message[]) => void
) {
  const conversationId = [userId1, userId2].sort().join("_");
  const q = query(
    collection(db, "messages"),
    where("conversationId", "==", conversationId),
    orderBy("createdAt", "asc"),
    limit(100)
  );

  return onSnapshot(q, (snapshot) => {
    const msgs = snapshot.docs.map((d) => ({ id: d.id, ...d.data() } as Message));
    callback(msgs);
  });
}

/**
 * Get conversations for a user
 */
export async function getUserConversations(userId: string): Promise<any[]> {
  const q = query(
    collection(db, "conversations"),
    where("participantIds", "array-contains", userId)
  );
  const snap = await getDocs(q);
  return snap.docs.map((d) => ({ id: d.id, ...d.data() }));
}
