/**
 * Logout user and redirect to login
 */
import { logoutUser } from '@/lib/auth';

export async function logout() {
  await logoutUser();
  window.location.href = '/login?loggedout=true';
}

/**
 * Get current logged in user
 */
export function getCurrentUser() {
  if (typeof window === 'undefined') return null;
  const storedUser = localStorage.getItem('currentUser');
  if (!storedUser) return null;
  try {
    return JSON.parse(storedUser);
  } catch {
    return null;
  }
}

/**
 * Check if user is logged in
 */
export function isLoggedIn() {
  return getCurrentUser() !== null;
}