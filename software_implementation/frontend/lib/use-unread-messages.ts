"use client";

import { useState, useEffect } from "react";
import {
  collection,
  query,
  where,
  onSnapshot,
} from "firebase/firestore";
import { db } from "@/lib/firebase";

/**
 * Hook that returns the total unread message count for a user
 * across all their conversations.
 */
export function useUnreadMessages(userId: string | null): number {
  const [count, setCount] = useState(0);

  useEffect(() => {
    if (!userId) return;

    const q = query(
      collection(db, "conversations"),
      where("participantIds", "array-contains", userId)
    );

    const unsub = onSnapshot(q, (snapshot) => {
      let total = 0;
      snapshot.docs.forEach((doc) => {
        const data = doc.data();
        total += data.unreadCount?.[userId] || 0;
      });
      setCount(total);
    });

    return () => unsub();
  }, [userId]);

  return count;
}
