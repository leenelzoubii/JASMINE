import { promises as fs } from 'fs';
import path from 'path';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { name, email, password, role, specialty } = body;

    // Validate required fields
    if (!name || !email || !password || !role) {
      return Response.json(
        { success: false, error: 'Missing required fields' },
        { status: 400 }
      );
    }

    // Read existing users
    const dataDir = path.join(process.cwd(), 'src', 'data');
    const filePath = path.join(dataDir, 'users.json');
    const fileContent = await fs.readFile(filePath, 'utf-8');
    const data = JSON.parse(fileContent);

    // Check if email already exists
    const existingUser = data.users.find(
      (u: any) => u.email.toLowerCase() === email.toLowerCase()
    );
    
    if (existingUser) {
      return Response.json(
        { success: false, error: 'Email already registered' },
        { status: 400 }
      );
    }

    // Create new user object
    const newUser: any = {
      id: `user-${Date.now()}`,
      email: email.toLowerCase(),
      password, // In production, hash this!
      name,
      role,
    };

    // Add role-specific fields
    if (role === 'professional') {
      newUser.specialty = specialty || 'General';
    } else if (role === 'parent') {
      newUser.child = {
        name: 'Child Name',
        age: 0,
        specialist: 'Not assigned',
      };
    }

    // Add user to array
    data.users.push(newUser);

    // Write back to file
    await fs.writeFile(filePath, JSON.stringify(data, null, 2));

    // Return success (without password)
    const { password: _, ...userWithoutPassword } = newUser;
    return Response.json({
      success: true,
      message: 'Account created successfully',
      user: userWithoutPassword,
    });
  } catch (error) {
    console.error('Registration error:', error);
    return Response.json(
      { success: false, error: 'Registration failed' },
      { status: 500 }
    );
  }
}
