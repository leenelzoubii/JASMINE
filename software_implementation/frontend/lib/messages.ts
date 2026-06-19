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
  orderBy,
  limit,
} from "firebase/firestore";
import { db } from "@/lib/firebase";

export interface Message {
  id: string;
  conversationId: string;
  senderId: string;
  receiverId: string;
  text: string;
  imageUrl?: string;
  createdAt: Timestamp;
  read: boolean;
  status: "sent" | "delivered" | "read";
}

export async function sendMessage(
  senderId: string,
  receiverId: string,
  text: string,
  imageUrl?: string
): Promise<void> {
  const conversationId = [senderId, receiverId].sort().join("_");

  await addDoc(collection(db, "messages"), {
    conversationId,
    senderId,
    receiverId,
    text,
    imageUrl: imageUrl || null,
    read: false,
    status: "sent",
    createdAt: serverTimestamp(),
  });

  const convRef = doc(db, "conversations", conversationId);
  const convSnap = await getDoc(convRef);

  if (convSnap.exists()) {
    const data = convSnap.data();
    await updateDoc(convRef, {
      lastMessage: text || "(image)",
      lastMessageTime: serverTimestamp(),
      [`unreadCount.${receiverId}`]: (data.unreadCount?.[receiverId] || 0) + 1,
    });
  } else {
    await setDoc(convRef, {
      participantIds: [senderId, receiverId],
      lastMessage: text || "(image)",
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
  const q = query(
    collection(db, "messages"),
    where("conversationId", "==", conversationId),
    orderBy("createdAt", "asc"),
    limit(500)
  );

  return onSnapshot(q, (snapshot) => {
    const msgs = snapshot.docs.map((d) => ({ id: d.id, ...d.data() } as Message));
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

// --- Typing indicator ---
let typingTimers: Record<string, ReturnType<typeof setTimeout>> = {};

export async function setTypingStatus(
  userId: string,
  conversationId: string,
  isTyping: boolean
) {
  const typingRef = doc(db, "typing", conversationId);
  if (isTyping) {
    await setDoc(typingRef, { [userId]: true, updatedAt: serverTimestamp() }, { merge: true });
    if (typingTimers[conversationId]) clearTimeout(typingTimers[conversationId]);
    typingTimers[conversationId] = setTimeout(async () => {
      await setDoc(typingRef, { [userId]: false }, { merge: true });
    }, 3000);
  } else {
    await setDoc(typingRef, { [userId]: false }, { merge: true });
  }
}

export function subscribeTypingStatus(
  conversationId: string,
  otherUserId: string,
  callback: (isTyping: boolean) => void
) {
  const typingRef = doc(db, "typing", conversationId);
  return onSnapshot(typingRef, (snap) => {
    if (snap.exists()) {
      const data = snap.data();
      callback(!!data[otherUserId]);
    } else {
      callback(false);
    }
  });
}
