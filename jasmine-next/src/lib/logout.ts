/**
 * Logout user and redirect to login
 */
import { useRouter } from 'next/navigation';

export function logout() {
  if (typeof window !== 'undefined') {
    localStorage.removeItem('jasmine_user');
    localStorage.removeItem('jasmine_role');
  }
  window.location.href = '/login?loggedout=true';
}

/**
 * Get current logged in user
 */
export function getCurrentUser() {
  if (typeof window === 'undefined') return null;
  
  const userStr = localStorage.getItem('jasmine_user');
  if (!userStr) return null;
  
  try {
    return JSON.parse(userStr);
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