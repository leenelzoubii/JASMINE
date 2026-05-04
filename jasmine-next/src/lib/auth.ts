/**
 * Authentication utilities for JASMINE
 * Uses simple JSON file for demo - replace with real database in production
 */

import usersData from '@/data/users.json';

export interface User {
  id: string;
  email: string;
  password: string;
  name: string;
  role: 'parent' | 'professional';
  child?: {
    name: string;
    age: number;
    specialist: string;
  };
  specialty?: string;
}

export interface LoginResult {
  success: boolean;
  user?: User;
  error?: string;
}

/**
 * Authenticate user by email and password
 */
export function authenticateUser(email: string, password: string): LoginResult {
  const user = usersData.users.find(
    (u: User) => u.email.toLowerCase() === email.toLowerCase() && u.password === password
  );

  if (!user) {
    return {
      success: false,
      error: 'Invalid email or password. Only demo accounts can login.'
    };
  }

  // Return user without password
  const { password: _, ...userWithoutPassword } = user;
  return {
    success: true,
    user: userWithoutPassword as User
  };
}

/**
 * Check if user exists by email
 */
export function userExists(email: string): boolean {
  return usersData.users.some(
    (u: User) => u.email.toLowerCase() === email.toLowerCase()
  );
}

/**
 * Get user role by email
 */
export function getUserRole(email: string): string | null {
  const user = usersData.users.find(
    (u: User) => u.email.toLowerCase() === email.toLowerCase()
  );
  return user?.role || null;
}