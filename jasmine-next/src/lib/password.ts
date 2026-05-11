import bcrypt from 'bcryptjs';
import { db } from '@/lib/firebase';
import { doc, updateDoc, getDoc, serverTimestamp } from 'firebase/firestore';

const SALT_ROUNDS = 12;
const RESET_TOKEN_EXPIRY_HOURS = 24;

export async function hashPassword(password: string): Promise<string> {
  return bcrypt.hash(password, SALT_ROUNDS);
}

export async function verifyPassword(password: string, hash: string): Promise<boolean> {
  return bcrypt.compare(password, hash);
}

export function generateTempPassword(): string {
  const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz23456789';
  let password = '';
  for (let i = 0; i < 10; i++) {
    password += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return password;
}

export function generateResetToken(): { token: string; expiresAt: Date } {
  const token = Array.from({ length: 32 }, () =>
    Math.random().toString(36).charAt(2)
  ).join('').substring(0, 32);
  const expiresAt = new Date();
  expiresAt.setHours(expiresAt.getHours() + RESET_TOKEN_EXPIRY_HOURS);
  return { token, expiresAt };
}

export async function updateUserPassword(userId: string, newPassword: string): Promise<void> {
  const hashedPassword = await hashPassword(newPassword);
  const userRef = doc(db, 'users', userId);
  await updateDoc(userRef, {
    password: hashedPassword,
    mustChangePassword: false,
    tempPasswordExpires: null,
    updatedAt: serverTimestamp(),
  });
}

export async function setTempPassword(userId: string, tempPassword: string, expiresAt: Date): Promise<void> {
  const hashedTemp = await hashPassword(tempPassword);
  const userRef = doc(db, 'users', userId);
  await updateDoc(userRef, {
    tempPassword: hashedTemp,
    tempPasswordExpires: expiresAt,
    mustChangePassword: true,
    updatedAt: serverTimestamp(),
  });
}

export async function clearTempPassword(userId: string): Promise<void> {
  const userRef = doc(db, 'users', userId);
  const userSnap = await getDoc(userRef);
  const userData = userSnap.data();

  if (userData && userData.tempPassword) {
    await updateDoc(userRef, {
      tempPassword: null,
      tempPasswordExpires: null,
      updatedAt: serverTimestamp(),
    });
  }
}

export function isTokenExpired(expiresAt: Date | null): boolean {
  if (!expiresAt) return true;
  return new Date() > new Date(expiresAt);
}
