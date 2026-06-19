import { NextRequest, NextResponse } from 'next/server';
import {
  createPatientAccess,
  getPatientLinksByProfessional,
  getPatientLinksByParent,
  getPatientLinksByPatientId,
  revokePatientAccess,
  grantPatientAccess,
  updateSharedAssessments,
  getAccessLinkById,
} from '@/lib/patient-access';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const professionalId = searchParams.get('professionalId');
    const parentId = searchParams.get('parentId');
    const patientId = searchParams.get('patientId');

    if (professionalId) {
      const links = await getPatientLinksByProfessional(professionalId);
      return NextResponse.json({ success: true, links });
    }

    if (parentId) {
      const links = await getPatientLinksByParent(parentId);
      return NextResponse.json({ success: true, links });
    }

    if (patientId) {
      const links = await getPatientLinksByPatientId(patientId);
      return NextResponse.json({ success: true, links });
    }

    return NextResponse.json({ success: false, error: 'Professional ID, Parent ID, or Patient ID is required' }, { status: 400 });
  } catch (err) {
    console.error('[API/PatientAccess] Error:', err);
    return NextResponse.json({ success: false, error: 'Failed to get patient access links' }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { patientId, patientName, professionalId, parentName, parentEmail } = body;

    if (!patientId || !patientName || !professionalId || !parentName || !parentEmail) {
      return NextResponse.json({ success: false, error: 'All fields are required' }, { status: 400 });
    }

    const result = await createPatientAccess({
      patientId,
      patientName,
      professionalId,
      parentName,
      parentEmail,
    });

    if (!result.success) {
      return NextResponse.json({ success: false, error: result.error }, { status: 400 });
    }

    return NextResponse.json({
      success: true,
      link: result.link,
      isNewParent: !!result.parentTempPassword,
      message: result.parentTempPassword
        ? 'Parent account created and credentials sent'
        : 'Parent access granted successfully',
    });
  } catch (err) {
    console.error('[API/PatientAccess] Error:', err);
    return NextResponse.json({ success: false, error: 'Failed to create patient access' }, { status: 500 });
  }
}

export async function PATCH(request: NextRequest) {
  try {
    const body = await request.json();
    const { linkId, action, assessmentIds } = body;

    if (!linkId || !action) {
      return NextResponse.json({ success: false, error: 'Link ID and action are required' }, { status: 400 });
    }

    let result;
    switch (action) {
      case 'revoke':
        result = await revokePatientAccess(linkId);
        break;
      case 'grant':
        result = await grantPatientAccess(linkId);
        break;
      case 'updateAssessments':
        if (!assessmentIds) {
          return NextResponse.json({ success: false, error: 'Assessment IDs are required' }, { status: 400 });
        }
        result = await updateSharedAssessments(linkId, assessmentIds);
        break;
      default:
        return NextResponse.json({ success: false, error: 'Invalid action' }, { status: 400 });
    }

    if (!result.success) {
      return NextResponse.json({ success: false, error: result.error }, { status: 400 });
    }

    return NextResponse.json({ success: true, message: `Action ${action} completed successfully` });
  } catch (err) {
    console.error('[API/PatientAccess] Error:', err);
    return NextResponse.json({ success: false, error: 'Failed to update patient access' }, { status: 500 });
  }
}
