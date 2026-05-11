'use server';

import nodemailer from 'nodemailer';
import { MailtrapTransport } from 'mailtrap';

const APP_URL = process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000';
const FROM_EMAIL = process.env.FROM_EMAIL || 'hello@example.com';
const FROM_NAME = 'JASMINE';

console.log('[Email] Service initialized');
console.log('[Email] MAILTRAP_API_KEY present:', !!process.env.MAILTRAP_API_KEY);
console.log('[Email] MAILTRAP_INBOX_ID present:', !!process.env.MAILTRAP_INBOX_ID);

export interface EmailResult {
  success: boolean;
  error?: string;
}

function getTransport() {
  const token = process.env.MAILTRAP_API_KEY;
  const testInboxId = process.env.MAILTRAP_INBOX_ID;

  console.log('[Email] Token length:', token?.length);
  console.log('[Email] Inbox ID:', testInboxId);

  if (!token) {
    console.error('[Email] FATAL: MAILTRAP_API_KEY is not set');
    return null;
  }

  if (!testInboxId) {
    console.error('[Email] FATAL: MAILTRAP_INBOX_ID is not set');
    return null;
  }

  const inboxIdNum = parseInt(testInboxId);
  if (isNaN(inboxIdNum)) {
    console.error('[Email] FATAL: MAILTRAP_INBOX_ID is not a valid number:', testInboxId);
    return null;
  }

  try {
    const transport = nodemailer.createTransport(
      MailtrapTransport({
        token,
        sandbox: true,
        testInboxId: inboxIdNum,
      })
    );
    console.log('[Email] Transport created successfully');
    return transport;
  } catch (err) {
    console.error('[Email] FATAL: Failed to create transport:', err);
    return null;
  }
}

async function sendEmailViaMailtrap(to: string, subject: string, html: string): Promise<EmailResult> {
  try {
    const transport = getTransport();
    
    if (!transport) {
      console.error('[Email] FATAL: Email transport not initialized - email NOT sent');
      console.error('[Email] To:', to, '| Subject:', subject);
      return { success: false, error: 'Email service not configured. Contact administrator.' };
    }

    console.log('[Email] Attempting to send to:', to);

    const info = await transport.sendMail({
      from: {
        name: FROM_NAME,
        address: FROM_EMAIL,
      },
      to,
      subject,
      html,
      text: html.replace(/<[^>]*>/g, ''),
    });

    console.log('[Email] Sent successfully! Message ID:', (info as any).messageId);
    return { success: true };
  } catch (err) {
    console.error('[Email] Failed to send email:', err);
    return { success: false, error: err instanceof Error ? err.message : 'Failed to send email' };
  }
}

export async function sendEmail(to: string, subject: string, html: string): Promise<EmailResult> {
  return sendEmailViaMailtrap(to, subject, html);
}

export async function sendParentCredentials(
  to: string,
  parentName: string,
  childName: string,
  tempPassword: string
): Promise<EmailResult> {
  const loginUrl = `${APP_URL}/login?role=parent`;

  const html = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Your JASMINE Account</title>
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f5f8fa;">
  <div style="max-width: 600px; margin: 0 auto; padding: 40px 20px;">
    <div style="background: white; border-radius: 16px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
      <div style="background: #3589a8; padding: 32px; text-align: center;">
        <h1 style="color: white; margin: 0; font-size: 28px;">JASMINE</h1>
        <p style="color: rgba(255,255,255,0.8); margin: 8px 0 0 0;">Professional Portal</p>
      </div>
      
      <div style="padding: 32px;">
        <h2 style="color: #1a1a1a; margin: 0 0 16px 0;">Hello ${parentName},</h2>
        
        <p style="color: #3f3f46; line-height: 1.6; margin: 0 0 24px 0;">
          You have been granted access to view your child's assessment data through JASMINE. 
          An account has been created for you.
        </p>
        
        <div style="background: #f5f8fa; border-radius: 12px; padding: 20px; margin-bottom: 24px;">
          <p style="margin: 0 0 12px 0; color: #3f3f46; font-size: 14px;">
            <strong>Child:</strong> ${childName}
          </p>
          <p style="margin: 0 0 12px 0; color: #3f3f46; font-size: 14px;">
            <strong>Email:</strong> ${to}
          </p>
          <p style="margin: 0; color: #3f3f46; font-size: 14px;">
            <strong>Temporary Password:</strong> ${tempPassword}
          </p>
        </div>
        
        <div style="text-align: center; margin-bottom: 24px;">
          <a href="${loginUrl}" style="display: inline-block; background: #3589a8; color: white; text-decoration: none; padding: 14px 32px; border-radius: 10px; font-weight: 600;">
            Log in to JASMINE
          </a>
        </div>
        
        <p style="color: #71717a; font-size: 13px; margin: 0;">
          <strong>Important:</strong> You will be required to change your password on first login. 
          This temporary password will expire in 24 hours.
        </p>
      </div>
      
      <div style="background: #f5f8fa; padding: 20px; text-align: center;">
        <p style="color: #71717a; font-size: 12px; margin: 0;">
          If you did not expect this email, please ignore it or contact the professional who created your account.
        </p>
      </div>
    </div>
  </div>
</body>
</html>
`;

  return sendEmailViaMailtrap(to, 'Your JASMINE Account Details', html);
}

export async function sendPasswordReset(to: string, resetUrl: string): Promise<EmailResult> {
  const html = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Reset Your Password</title>
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f5f8fa;">
  <div style="max-width: 600px; margin: 0 auto; padding: 40px 20px;">
    <div style="background: white; border-radius: 16px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
      <div style="background: #3589a8; padding: 32px; text-align: center;">
        <h1 style="color: white; margin: 0; font-size: 28px;">JASMINE</h1>
      </div>
      
      <div style="padding: 32px;">
        <h2 style="color: #1a1a1a; margin: 0 0 16px 0;">Reset Your Password</h2>
        
        <p style="color: #3f3f46; line-height: 1.6; margin: 0 0 24px 0;">
          We received a request to reset your password. Click the button below to create a new password.
        </p>
        
        <div style="text-align: center; margin-bottom: 24px;">
          <a href="${resetUrl}" style="display: inline-block; background: #3589a8; color: white; text-decoration: none; padding: 14px 32px; border-radius: 10px; font-weight: 600;">
            Reset Password
          </a>
        </div>
        
        <p style="color: #71717a; font-size: 13px; margin: 0 0 12px 0;">
          This link will expire in 24 hours.
        </p>
        
        <p style="color: #71717a; font-size: 13px; margin: 0;">
          If you did not request a password reset, you can safely ignore this email.
        </p>
      </div>
    </div>
  </div>
</body>
</html>
`;

  return sendEmailViaMailtrap(to, 'Reset Your JASMINE Password', html);
}
