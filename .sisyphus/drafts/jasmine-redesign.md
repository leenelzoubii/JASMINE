# Draft: JASMINE Redesign - Next.js + Full Features

## Project Overview
- **Name**: JASMINE - Joint Analysis and Screening for Motor Imbalances
- **Type**: Medical screening web application
- **Current Stack**: Streamlit (to be replaced)
- **Target Stack**: Next.js 14 + Python FastAPI backend

## Requirements (User-Confirmed)

### 1. New Color Scheme
- Primary: `#74b3ce` (teal)
- Dark mode: `#172A3A` (deep navy), `#004346` (dark teal), `#508991` (muted teal)
- Light mode: `#d6f3f4` (light teal), `#74b3ce` (primary), `#ffffff` backgrounds

### 2. Theme System
- Toggle between dark/light themes
- Persistent theme preference
- Smooth transitions between themes

### 3. Pages Structure
- **Main Page**: Landing page with features, call-to-action
- **Login/Register**: Separate authentication pages
- **Dashboard**: Role-specific (Professional vs Parent)
- **Professional Portal**:
  - Dashboard overview
  - Patient management
  - Assessments
  - Messaging with parents
  - Profile
- **Parent Portal**:
  - Dashboard overview
  - Child profiles
  - Results/assessments
  - Messaging with professional
  - Profile

### 4. Features to Add
- User profiles (editable info, avatar, etc.)
- Messaging system (real-time chat between professional and parent)
- Patient/child profiles management
- Assessment submission and results viewing
- Dark/Light theme toggle with smooth animations

### 5. UI Requirements
- Production-ready, sleek, modern
- Smooth animations (page transitions, hover effects)
- Medical professional aesthetic
- Responsive design

### 6. Migrate or Fresh Start
- **Start fresh** - current data is placeholders

## Technical Decisions (Implicit)
- Next.js 14 with App Router
- TypeScript
- Tailwind CSS for styling
- FastAPI (Python) for backend API
- Prisma + PostgreSQL or SQLite for data
- WebSocket for real-time messaging
- Zustand or React Context for state
- next-themes for dark/light mode

## Open Questions
- [ ] Database preference: PostgreSQL (production) or SQLite (simpler)?
- [ ] Authentication: NextAuth.js or custom?
- [ ] Real-time: WebSocket or polling?
- [ ] Deployment target: Vercel + Render/Docker?

## Scope Boundaries
- INCLUDE: Full redesign, Next.js frontend, FastAPI backend, messaging
- EXCLUDE: Keep ML model logic (reuse existing Python code)