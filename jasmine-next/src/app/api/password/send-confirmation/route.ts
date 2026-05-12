import { NextResponse } from 'next/server';
import nodemailer from 'nodemailer';

export const runtime = 'nodejs';

const FIREBASE_PROJECT_ID = 'jasmine-4671c';
const FIREBASE_API_KEY = 'AIzaSyDQ1trSa5rCJXZMr6xnvmNhyLBRvIfQL_k';

async function userExistsInFirestore(email: string): Promise<boolean> {
  const url = `https://firestore.googleapis.com/v1/projects/${FIREBASE_PROJECT_ID}/databases/(default)/documents:runQuery?key=${FIREBASE_API_KEY}`;

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      structuredQuery: {
        from: [
          {
            collectionId: 'users',
          },
        ],
        where: {
          fieldFilter: {
            field: {
              fieldPath: 'email',
            },
            op: 'EQUAL',
            value: {
              stringValue: email,
            },
          },
        },
        limit: 1,
      },
    }),
  });

  if (!response.ok) {
    console.error('Firestore REST check failed:', await response.text());
    return false;
  }

  const data = await response.json();

  return data.some((item: { document?: unknown }) => item.document);
}

export async function POST(request: Request) {
  try {
    const { email } = await request.json();

    if (!email) {
      return NextResponse.json(
        { success: false, error: 'Email is required' },
        { status: 400 }
      );
    }

    const cleanEmail = email.trim().toLowerCase();

    const exists = await userExistsInFirestore(cleanEmail);

    if (!exists) {
      return NextResponse.json(
        {
          success: false,
          error: 'No account found with this email address.',
        },
        { status: 404 }
      );
    }

    const appUrl = process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000';
    const confirmUrl = `${appUrl}/confirm-reset?email=${encodeURIComponent(cleanEmail)}`;

    const port = Number(process.env.MAILTRAP_PORT || 587);

    const transporter = nodemailer.createTransport({
      host: process.env.MAILTRAP_HOST,
      port,
      secure: port === 465,
      auth: {
        user: process.env.MAILTRAP_USER,
        pass: process.env.MAILTRAP_PASS,
      },
      connectionTimeout: 60000,
      greetingTimeout: 60000,
      socketTimeout: 60000,
      tls: {
        rejectUnauthorized: false,
      },
    });

    await transporter.verify();

    await transporter.sendMail({
      from: process.env.MAIL_FROM || 'JASMINE <no-reply@jasmine-demo.com>',
      to: cleanEmail,
      subject: 'Confirm your JASMINE password reset request',
      html: `
        <div style="font-family: Arial, sans-serif; line-height: 1.6;">
          <h2>JASMINE Password Reset Request</h2>

          <p>We received a request to reset the password for this account:</p>

          <p><strong>${cleanEmail}</strong></p>

          <p>If this was you, click the button below:</p>

          <p>
            <a
              href="${confirmUrl}"
              style="display:inline-block;padding:12px 18px;background:#3894b5;color:white;text-decoration:none;border-radius:8px;"
            >
              Yes, this is me
            </a>
          </p>

          <p>If you did not request this, you can ignore this email.</p>
        </div>
      `,
    });

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Send confirmation email error:', error);

    return NextResponse.json(
      { success: false, error: 'Failed to send confirmation email' },
      { status: 500 }
    );
  }
}