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
  writeBatch,
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
  status: "sent" | "delivered" | "read";
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
    status: "sent",
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

/**
 * Mark all messages in a conversation as delivered for the current user.
 * Updates messages sent BY the other user (receiverId matches current user)
 * from "sent" to "delivered".
 */
export async function markMessagesDelivered(
  currentUserId: string,
  otherUserId: string
): Promise<void> {
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

/**
 * Mark all unread messages in a conversation as READ for the current user.
 * Also sets status to "read" on messages the user received.
 * Resets unreadCount for this user to 0.
 */
export async function markConversationAsRead(
  currentUserId: string,
  otherUserId: string
): Promise<void> {
  const conversationId = [currentUserId, otherUserId].sort().join("_");

  // Mark all unread received messages as read
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

  // Reset unread count for current user
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
