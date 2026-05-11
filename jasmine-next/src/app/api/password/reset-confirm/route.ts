import { NextRequest, NextResponse } from 'next/server';
import { doc, getDoc, updateDoc, serverTimestamp } from 'firebase/firestore';
import { db } from '@/lib/firebase';
import { hashPassword } from '@/lib/password';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { userId, token, newPassword } = body;

    if (!userId || !token || !newPassword) {
      return NextResponse.json({ success: false, error: 'User ID, token, and new password are required' }, { status: 400 });
    }

    if (newPassword.length < 8) {
      return NextResponse.json({ success: false, error: 'Password must be at least 8 characters' }, { status: 400 });
    }

    const userRef = doc(db, 'users', userId);
    const userSnap = await getDoc(userRef);

    if (!userSnap.exists()) {
      return NextResponse.json({ success: false, error: 'Invalid user' }, { status: 400 });
    }

    const userData = userSnap.data();

    if (userData.resetToken !== token) {
      return NextResponse.json({ success: false, error: 'Invalid reset token' }, { status: 400 });
    }

    const expiresAt = userData.resetTokenExpires?.toDate();
    if (!expiresAt || new Date() > expiresAt.toDate()) {
      return NextResponse.json({ success: false, error: 'Reset token has expired' }, { status: 400 });
    }

    const hashedPassword = await hashPassword(newPassword);

    await updateDoc(userRef, {
      password: hashedPassword,
      resetToken: null,
      resetTokenExpires: null,
      mustChangePassword: false,
      updatedAt: serverTimestamp(),
    });

    return NextResponse.json({ success: true, message: 'Password reset successfully' });
  } catch (err) {
    console.error('[API/Password] Reset confirm error:', err);
    return NextResponse.json({ success: false, error: 'Failed to reset password' }, { status: 500 });
  }
}
