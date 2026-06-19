import { NextRequest, NextResponse } from 'next/server';
import { getParentAccountByEmail, resendParentCredentials, deactivateParentAccount, reactivateParentAccount } from '@/lib/parent-accounts';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const email = searchParams.get('email');

    if (!email) {
      return NextResponse.json({ success: false, error: 'Email is required' }, { status: 400 });
    }

    const parent = await getParentAccountByEmail(email);

    if (!parent) {
      return NextResponse.json({ exists: false });
    }

    return NextResponse.json({
      exists: true,
      parent: {
        id: parent.id,
        email: parent.email,
        name: parent.name,
        isActive: parent.isActive,
      },
    });
  } catch (err) {
    console.error('[API/ParentAccounts] Error:', err);
    return NextResponse.json({ success: false, error: 'Failed to check parent account' }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { parentId, childName } = body;

    if (!parentId || !childName) {
      return NextResponse.json({ success: false, error: 'Parent ID and child name are required' }, { status: 400 });
    }

    const result = await resendParentCredentials(parentId, childName);

    if (!result.success) {
      return NextResponse.json({ success: false, error: result.error }, { status: 400 });
    }

    return NextResponse.json({ success: true, message: 'Credentials sent successfully' });
  } catch (err) {
    console.error('[API/ParentAccounts] Error:', err);
    return NextResponse.json({ success: false, error: 'Failed to send credentials' }, { status: 500 });
  }
}

export async function PATCH(request: NextRequest) {
  try {
    const body = await request.json();
    const { parentId, action } = body;

    if (!parentId || !action) {
      return NextResponse.json({ success: false, error: 'Parent ID and action are required' }, { status: 400 });
    }

    let result;
    if (action === 'deactivate') {
      result = await deactivateParentAccount(parentId);
    } else if (action === 'reactivate') {
      result = await reactivateParentAccount(parentId);
    } else {
      return NextResponse.json({ success: false, error: 'Invalid action' }, { status: 400 });
    }

    if (!result.success) {
      return NextResponse.json({ success: false, error: result.error }, { status: 400 });
    }

    return NextResponse.json({ success: true, message: `Account ${action}d successfully` });
  } catch (err) {
    console.error('[API/ParentAccounts] Error:', err);
    return NextResponse.json({ success: false, error: 'Failed to update account' }, { status: 500 });
  }
}
