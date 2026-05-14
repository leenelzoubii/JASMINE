"use client";

import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { CheckCircle, AlertCircle, X } from "lucide-react";

export interface ToastData {
  id: string;
  type: "success" | "error";
  title: string;
  message: string;
}

let addToastFn: ((toast: ToastData) => void) | null = null;

export function showToast(type: "success" | "error", title: string, message: string) {
  if (addToastFn) {
    addToastFn({ id: Date.now().toString(), type, title, message });
  }
}

export function ToastContainer() {
  const [toasts, setToasts] = useState<ToastData[]>([]);

  const addToast = useCallback((toast: ToastData) => {
    setToasts((prev) => [...prev, toast]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== toast.id));
    }, 4000);
  }, []);

  useEffect(() => {
    addToastFn = addToast;
    return () => { addToastFn = null; };
  }, [addToast]);

  return (
    <div className="fixed top-4 right-4 z-[100] flex flex-col gap-3 pointer-events-none">
      <AnimatePresence>
        {toasts.map((toast) => (
          <motion.div
            key={toast.id}
            initial={{ opacity: 0, x: 100, scale: 0.9 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            exit={{ opacity: 0, x: 100, scale: 0.9 }}
            className="pointer-events-auto flex items-start gap-3 px-5 py-4 rounded-2xl shadow-lg max-w-sm"
            style={{
              backgroundColor: toast.type === "success" ? "#16a34a" : "#dc2626",
              color: "white",
            }}
          >
            {toast.type === "success" ? (
              <CheckCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
            ) : (
              <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
            )}
            <div className="flex-1 min-w-0">
              <p className="font-semibold text-sm">{toast.title}</p>
              <p className="text-sm opacity-90">{toast.message}</p>
            </div>
            <button
              onClick={() => setToasts((prev) => prev.filter((t) => t.id !== toast.id))}
              className="p-0.5 hover:opacity-70"
            >
              <X className="w-4 h-4" />
            </button>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}
