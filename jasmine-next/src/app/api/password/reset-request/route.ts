import { NextRequest, NextResponse } from 'next/server';
import { generateResetToken } from '@/lib/password';
import { sendPasswordReset } from '@/lib/emails/service';

const testMode = process.env.EMAIL_TEST_MODE === 'true';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { email } = body;

    console.log('[API/Password] Reset request for email:', email);
    console.log('[API/Password] Test mode:', testMode);

    if (!email) {
      return NextResponse.json({ success: false, error: 'Email is required' }, { status: 400 });
    }

    // TEST MODE: Skip Firestore, always try to send email
    if (testMode) {
      console.log('[API/Password] TEST MODE: Skipping Firestore, sending test email');

      const { token } = generateResetToken();
      const resetUrl = `${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/reset-password?token=${token}&userId=test`;

      const emailResult = await sendPasswordReset(email, resetUrl);
      console.log('[API/Password] TEST MODE - Email result:', emailResult);

      if (!emailResult.success) {
        console.error('[API/Password] TEST MODE - FAILED to send email:', emailResult.error);
        return NextResponse.json({ success: false, error: 'Failed to send email: ' + emailResult.error }, { status: 500 });
      }

      return NextResponse.json({ success: true, message: 'Test email sent successfully!' });
    }

    // NORMAL MODE: Use Firestore (requires Firebase Admin SDK - not yet implemented)
    return NextResponse.json({ success: false, error: 'Normal mode not yet implemented - Firestore requires Admin SDK' }, { status: 500 });

  } catch (err) {
    console.error('[API/Password] Reset request error:', err);
    return NextResponse.json({ success: false, error: 'Failed to process reset request' }, { status: 500 });
  }
}