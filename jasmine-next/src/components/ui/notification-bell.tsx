"use client";

import { useState, useEffect, useRef } from "react";
import { Bell, BellDot, X, CheckCheck, MessageSquare, UserPlus, FileText, UserCheck, Activity, Trash2, Clock } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import Link from "next/link";
import { getCurrentUser } from "@/lib/auth";
import {
  subscribeToNotifications,
  subscribeToUnreadCount,
  markNotificationRead,
  markAllNotificationsRead,
  Notification,
} from "@/lib/notifications";
import { showToast } from "./toast";

const iconMap: Record<string, React.ReactNode> = {
  message: <MessageSquare className="w-4 h-4" />,
  parent_request: <UserPlus className="w-4 h-4" />,
  patient_added: <FileText className="w-4 h-4" />,
  patient_removed: <Trash2 className="w-4 h-4" />,
  request_accepted: <UserCheck className="w-4 h-4" />,
  assessment_complete: <Activity className="w-4 h-4" />,
};

function playDingSound() {
  try {
    const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.frequency.value = 800;
    osc.type = "sine";
    gain.gain.setValueAtTime(0.3, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.3);
    osc.start(ctx.currentTime);
    osc.stop(ctx.currentTime + 0.3);
  } catch {
    // Audio not available
  }
}

function timeAgo(date: any): string {
  if (!date) return "";
  const ms = date?.toMillis?.() || date?.seconds * 1000 || 0;
  if (!ms) return "";
  const seconds = Math.floor((Date.now() - ms) / 1000);
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `${days}d ago`;
  return new Date(ms).toLocaleDateString();
}

export function NotificationBell() {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [open, setOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const prevCountRef = useRef(0);

  useEffect(() => {
    const user = getCurrentUser();
    if (!user) return;

    const unsubNotifs = subscribeToNotifications(user.id, setNotifications);
    const unsubCount = subscribeToUnreadCount(user.id, (count) => {
      // Play sound when new notification arrives
      if (count > prevCountRef.current && count > 0) {
        playDingSound();
      }
      prevCountRef.current = count;
      setUnreadCount(count);
    });

    return () => {
      unsubNotifs();
      unsubCount();
    };
  }, []);

  // Track previous notification IDs for toast popup
  const prevIdsRef = useRef<Set<string> | null>(null);
  useEffect(() => {
    if (notifications.length === 0) return;
    const currentIds = new Set(notifications.map((n) => n.id));
    // Skip first load - only toast NEW notifications
    if (prevIdsRef.current === null) {
      prevIdsRef.current = currentIds;
      return;
    }
    // Find new notifications
    notifications.forEach((n) => {
      if (!prevIdsRef.current!.has(n.id)) {
        showToast("success", n.title, n.message);
      }
    });
    prevIdsRef.current = currentIds;
  }, [notifications]);

  // Close on click outside
  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    if (open) document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [open]);

  const handleMarkRead = async (id: string) => {
    await markNotificationRead(id);
  };

  const handleMarkAllRead = async () => {
    const user = getCurrentUser();
    if (user) {
      await markAllNotificationsRead(user.id);
    }
  };

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={() => setOpen(!open)}
        className="relative p-2.5 rounded-xl transition-all"
        style={{ backgroundColor: "var(--background-alt)" }}
      >
        {unreadCount > 0 ? (
          <>
            <BellDot className="w-5 h-5" style={{ color: "var(--primary)" }} />
            <span className="absolute -top-1 -right-1 w-5 h-5 rounded-full bg-red-500 text-white text-xs flex items-center justify-center font-bold">
              {unreadCount > 9 ? "9+" : unreadCount}
            </span>
          </>
        ) : (
          <Bell className="w-5 h-5" style={{ color: "var(--text-muted)" }} />
        )}
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.95 }}
            className="absolute right-0 top-12 w-80 sm:w-96 rounded-2xl shadow-xl z-50"
            style={{ backgroundColor: "var(--background)", border: "1px solid var(--border)" }}
          >
            <div className="flex items-center justify-between p-4 border-b" style={{ borderColor: "var(--border)" }}>
              <h3 className="font-semibold" style={{ color: "var(--foreground)" }}>Notifications</h3>
              {unreadCount > 0 && (
                <button
                  onClick={handleMarkAllRead}
                  className="text-xs flex items-center gap-1 px-2 py-1 rounded-lg"
                  style={{ color: "var(--primary)" }}
                >
                  <CheckCheck className="w-3 h-3" />
                  Mark all read
                </button>
              )}
            </div>

            <div className="max-h-80 overflow-y-auto">
              {notifications.length === 0 ? (
                <div className="py-8 text-center">
                  <Bell className="w-8 h-8 mx-auto mb-2" style={{ color: "var(--text-muted)" }} />
                  <p className="text-sm" style={{ color: "var(--text-muted)" }}>No notifications yet</p>
                </div>
              ) : (
                notifications.map((n) => (
                  <Link
                    key={n.id}
                    href={n.link || "#"}
                    onClick={() => {
                      handleMarkRead(n.id);
                      setOpen(false);
                    }}
                    className="flex items-start gap-3 p-4 transition-colors hover:bg-gray-50 dark:hover:bg-dark-deep"
                    style={{
                      backgroundColor: n.read ? "transparent" : "rgba(74, 155, 184, 0.05)",
                    }}
                  >
                    <div
                      className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0"
                      style={{ backgroundColor: "var(--primary-light)" }}
                    >
                      {iconMap[n.type] || <Bell className="w-4 h-4" style={{ color: "var(--primary)" }} />}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium" style={{ color: "var(--foreground)" }}>
                        {n.title}
                      </p>
                      <p className="text-xs" style={{ color: "var(--text-muted)" }}>
                        {n.message}
                      </p>
                      <div className="flex items-center gap-1 mt-1">
                        <Clock className="w-3 h-3" style={{ color: "var(--text-muted)" }} />
                        <span className="text-xs" style={{ color: "var(--text-muted)" }}>
                          {timeAgo(n.createdAt)}
                        </span>
                      </div>
                    </div>
                    {!n.read && (
                      <div className="w-2 h-2 rounded-full flex-shrink-0 mt-1.5" style={{ backgroundColor: "var(--primary)" }} />
                    )}
                  </Link>
                ))
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
