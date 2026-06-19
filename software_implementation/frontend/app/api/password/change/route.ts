import { NextRequest, NextResponse } from 'next/server';
import { doc, getDoc, updateDoc, serverTimestamp } from 'firebase/firestore';
import { db } from '@/lib/firebase';
import { hashPassword, verifyPassword } from '@/lib/password';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { userId, oldPassword, newPassword } = body;

    if (!userId || !oldPassword || !newPassword) {
      return NextResponse.json({ success: false, error: 'User ID, old password, and new password are required' }, { status: 400 });
    }

    if (newPassword.length < 8) {
      return NextResponse.json({ success: false, error: 'Password must be at least 8 characters' }, { status: 400 });
    }

    const userRef = doc(db, 'users', userId);
    const userSnap = await getDoc(userRef);

    if (!userSnap.exists()) {
      return NextResponse.json({ success: false, error: 'User not found' }, { status: 400 });
    }

    const userData = userSnap.data();
    const storedPassword = userData.password;

    if (storedPassword) {
      const isValid = await verifyPassword(oldPassword, storedPassword);
      if (!isValid) {
        return NextResponse.json({ success: false, error: 'Current password is incorrect' }, { status: 400 });
      }
    }

    const hashedPassword = await hashPassword(newPassword);

    await updateDoc(userRef, {
      password: hashedPassword,
      mustChangePassword: false,
      updatedAt: serverTimestamp(),
    });

    return NextResponse.json({ success: true, message: 'Password changed successfully' });
  } catch (err) {
    console.error('[API/Password] Change password error:', err);
    return NextResponse.json({ success: false, error: 'Failed to change password' }, { status: 500 });
  }
}
