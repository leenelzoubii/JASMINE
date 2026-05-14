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
  conversationId: string;
  senderId: string;
  receiverId: string;
  text: string;
  createdAt: Timestamp;
  read: boolean;
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

  // Add message document
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
    // Create conversation doc
    await updateDoc(convRef, {
      participantIds: [senderId, receiverId],
      lastMessage: text,
      lastMessageTime: serverTimestamp(),
      unreadCount: { [receiverId]: 1 },
      createdAt: serverTimestamp(),
    });
  }
}

/**
 * Subscribe to messages for a conversation
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
    where("participantIds", "array-contains", userId),
    orderBy("lastMessageTime", "desc")
  );
  const snap = await getDocs(q);
  return snap.docs.map((d) => ({ id: d.id, ...d.data() }));
}
