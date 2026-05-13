# JASMINE Next.js Redesign - Complete Work Plan

## TL;DR

> **Quick Summary**: Complete redesign from Streamlit to Next.js 14 + FastAPI backend. Implement dark/light themes, professional and parent portals, real-time messaging, profiles, and sleek production-ready UI.
>
> **Deliverables**: 
> - Next.js 14 frontend with App Router
> - FastAPI Python backend for ML inference and messaging
> - Dark/Light theme system with smooth transitions
> - Professional Portal with dashboard, patients, assessments, messaging
> - Parent Portal with dashboard, children profiles, results, messaging
> - User profiles with avatars
> - Real-time chat between professional and parent
> - New color scheme (#74b3ce primary, #172A3A dark, #d6f3f4 light)
> - Smooth animations and transitions
>
> **Estimated Effort**: XL (large project)
> **Parallel Execution**: YES - 6 waves
> **Critical Path**: Setup → Theme System → Auth → Portals → Messaging → Polish

---

## Context

### Original Request
User wants to completely redesign the JASMINE autism screening app from Streamlit to a production-ready Next.js application with:
- New color scheme: #d6f3f4, #74b3ce, #004346, #508991, #172A3A
- Dark and light themes with smooth transitions
- Separate main page and login/register pages
- Professional portal improved
- Parent portal added
- Messaging between professional and parent
- User profiles
- Production-ready, sleek, modern UI
- Smooth animations

### User Preferences Confirmed
- Switch from Streamlit to Next.js (React + Python backend)
- Fresh start (current data is placeholders)
- All in one comprehensive plan

### Technical Requirements
- Next.js 14 with App Router
- TypeScript for type safety
- Tailwind CSS for styling (matches color scheme)
- FastAPI Python backend
- SQLite database (simpler than PostgreSQL for this scale)
- WebSocket for real-time messaging
- next-themes for dark/light mode

---

## Work Objectives

### Core Objective
Build a production-ready medical screening web application that allows:
1. **Professionals** to manage patients, run assessments, message parents
2. **Parents** to view their child's results, message professionals

### Concrete Deliverables
- [ ] Next.js 14 frontend with App Router structure
- [ ] FastAPI backend with /api routes
- [ ] Dark/Light theme system (persisted, animated)
- [ ] Main landing page (/): Features, CTA, navigation
- [ ] Login page (/login): Form, validation, redirect
- [ ] Register page (/register): Role selection, form
- [ ] Professional dashboard (/professional): Stats, recent patients
- [ ] Professional patients (/professional/patients): List, add, view
- [ ] Professional assessments (/professional/assessments): Run, view results
- [ ] Professional messages (/professional/messages): Chat with parents
- [ ] Professional profile (/professional/profile): Edit info, avatar
- [ ] Parent dashboard (/parent): Stats, children
- [ ] Parent children (/parent/children): List, view profile
- [ ] Parent results (/parent/results): View assessment results
- [ ] Parent messages (/parent/messages): Chat with professional
- [ ] Parent profile (/parent/profile): Edit info, avatar
- [ ] User profile page with avatar upload
- [ ] Real-time messaging with WebSocket
- [ ] Smooth page animations (framer-motion)

### Definition of Done
- [ ] All pages render without errors
- [ ] Theme toggle works with smooth transition
- [ ] Authentication flow complete
- [ ] Professional can add patient and run assessment
- [ ] Parent can view results
- [ ] Messages deliver in real-time
- [ ] All forms validate properly
- [ ] Mobile responsive

### Must Have
- Dark/Light theme with #74b3ce primary color
- Separate landing, login, register pages
- Role-based routing (Professional vs Parent)
- Messaging system
- User profiles with avatars

### Must NOT Have
- Streamlit dependency
- Hardcoded demo data as "production" data
- Broken navigation
- Unstyled pages

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: NO (new project)
- **Automated tests**: YES (TDD approach)
- **Framework**: Vitest + React Testing Library + pytest

### QA Policy
Every task includes agent-executed QA scenarios. Evidence saved to `.sisyphus/evidence/`.

- **Frontend**: Playwright - Navigate, interact, assert DOM, screenshot
- **Backend**: curl/API tests - Send requests, assert JSON responses
- **Real-time**: WebSocket client - Connect, send, receive messages

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Foundation - can start immediately):
├── Task 1: Initialize Next.js 14 project with TypeScript [quick]
├── Task 2: Set up Tailwind CSS with custom color scheme [quick]
├── Task 3: Configure next-themes for dark/light mode [quick]
├── Task 4: Create theme context and provider [quick]
├── Task 5: Create base layout with navigation [quick]
└── Task 6: Set up FastAPI backend skeleton [quick]

Wave 2 (Authentication - after Wave 1):
├── Task 7: Design and implement landing page [visual-engineering]
├── Task 8: Create login page with form [visual-engineering]
├── Task 9: Create register page with role selection [visual-engineering]
├── Task 10: Set up NextAuth.js or custom auth [unspecified-high]
├── Task 11: Create database schema (SQLite + Prisma) [unspecified-high]
└── Task 12: Implement backend auth routes [unspecified-high]

Wave 3 (Professional Portal - after Wave 2):
├── Task 13: Create professional layout and sidebar [visual-engineering]
├── Task 14: Professional dashboard with stats [visual-engineering]
├── Task 15: Patient management (list, add, view) [unspecified-high]
├── Task 16: Assessment runs and results view [deep]
├── Task 17: Profile page with avatar upload [visual-engineering]
└── Task 18: Reuse ML model inference endpoints [deep]

Wave 4 (Parent Portal - after Wave 2, parallel with Wave 3):
├── Task 19: Create parent layout and sidebar [visual-engineering]
├── Task 20: Parent dashboard [visual-engineering]
├── Task 21: Children profiles management [unspecified-high]
├── Task 22: View assessment results [unspecified-high]
├── Task 23: Parent profile with avatar upload [visual-engineering]
└── Task 24: Link parent to professional [unspecified-high]

Wave 5 (Messaging System - after Waves 3&4):
├── Task 25: WebSocket server setup in FastAPI [deep]
├── Task 26: Message database schema [quick]
├── Task 27: Professional messaging UI [visual-engineering]
├── Task 28: Parent messaging UI [visual-engineering]
├── Task 29: Real-time message delivery [deep]
└── Task 30: Message notifications [visual-engineering]

Wave 6 (Polish & Integration - after Wave 5):
├── Task 31: Add smooth animations (framer-motion) [visual-engineering]
├── Task 32: Mobile responsive design [quick]
├── Task 33: Error boundaries and loading states [quick]
├── Task 34: SEO metadata [quick]
└── Task 35: Production build optimization [quick]

Wave FINAL (Verification - after ALL tasks):
├── Task F1: Plan Compliance Audit (oracle)
├── Task F2: Code Quality Review (unspecified-high)
├── Task F3: Real Manual QA (unspecified-high)
├── Task F4: Scope Fidelity Check (deep)
-> Present results -> Get explicit user okay
```

### Dependency Matrix

- **1-6**: - - 7-12, 1
- **7-12**: 1-6 - 13-24, 2
- **13-18**: 7-12 - 25-30, 3
- **19-24**: 7-12 - 25-30, 3
- **25-30**: 13-24 - 31-35, 4
- **31-35**: 25-30 - F1-F4, 5

### Agent Dispatch Summary

- **1**: **6** - T1-T6 → `quick`
- **2**: **6** - T7-T12 → `visual-engineering` + `unspecified-high`
- **3**: **6** - T13-T18 → `visual-engineering` + `unspecified-high` + `deep`
- **4**: **6** - T19-T24 → `visual-engineering` + `unspecified-high`
- **5**: **6** - T25-T30 → `deep` + `visual-engineering`
- **6**: **5** - T31-T35 → `visual-engineering` + `quick`
- **FINAL**: **4** - F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

### Wave 1: Foundation Tasks

- [ ] 1. Initialize Next.js 14 project with TypeScript

  **What to do**:
  - Run `npx create-next-app@latest jasmine-next --typescript --tailwind --eslint`
  - Configure: App Router, src directory, import alias @/*
  - Install dependencies: next-themes, zustand, framer-motion, axios
  - Clean up default page content

  **Must NOT do**:
  - Keep default Next.js branding
  - Leave unused files

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Standard initialization task
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2-6)
  - **Blocks**: Tasks 7-35, all downstream
  - **Blocked By**: None (can start immediately)

  **References**:
  - Next.js 14 docs: https://nextjs.org/docs
  - create-next-app options

  **Acceptance Criteria**:
  - [ ] `npm run dev` starts without errors
  - [ ] http://localhost:3000 shows blank Next.js page

  **QA Scenarios**:
  ```
  Scenario: Project starts successfully
    Tool: Bash
    Preconditions: None
    Steps:
      1. Run npm run dev in background
      2. Wait 10s for server
      3. curl http://localhost:3000
    Expected Result: HTML page returned (200 status)
    Evidence: .sisyphus/evidence/task-1-start.{ext}
  ```

  **Commit**: YES
  - Message: `chore: initialize Next.js 14 project`
  - Files: `package.json`, `tsconfig.json`, `next.config.*`, `src/app/*`

- [ ] 2. Set up Tailwind CSS with custom color scheme

  **What to do**:
  - Update tailwind.config.ts with custom colors:
    ```typescript
    colors: {
      primary: {
        DEFAULT: '#74b3ce',
        light: '#d6f3f4',
        dark: '#004346',
        muted: '#508991',
      },
      dark: {
        DEFAULT: '#172A3A',
        deep: '#0f1d25',
        surface: '#1e3a4c',
      },
    }
    ```
  - Extend theme with these colors
  - Create base styles in globals.css

  **Must NOT do**:
  - Keep default blue colors
  - Hardcode colors in components

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Configuration task

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 1)
  - **Blocks**: All styling tasks (7-35)
  - **Blocked By**: Task 1

  **References**:
  - Tailwind CSS config docs

  **Acceptance Criteria**:
  - [ ] tailwind.config.ts has custom colors
  - [ ] `bg-primary` utility works

  **QA Scenarios**:
  ```
  Scenario: Custom colors work
    Tool: Bash
    Preconditions: npm run dev running
    Steps:
      1. Add test div with bg-primary
      2. Verify color in computed style
    Expected Result: #74b3ce
    ```

  **Commit**: YES (group with 1)
  - Files: `tailwind.config.ts`, `src/app/globals.css`

- [ ] 3. Configure next-themes for dark/light mode

  **What to do**:
  - Install next-themes
  - Create ThemeProvider component
  - Wrap app with ThemeProvider
  - Configure light/dark color mappings:
    - Light: light bg (#ffffff), primary (#74b3ce)
    - Dark: dark bg (#172A3A), primary (#74b3ce)

  **Must NOT do**:
  - Hardcode theme colors in components
  - Ignore system preference

  **Recommended Agent Profile**:
  - **Category**: `quick`

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 1)
  - **Blocks**: All theme-dependent tasks
  - **Blocked By**: Task 1

  **References**:
  - next-themes npm package

  **Acceptance Criteria**:
  - [ ] Theme toggle appears in UI
  - [ ] Theme persists on reload

  **QA Scenarios**:
  ```
  Scenario: Theme toggle works
    Tool: Playwright
    Preconditions: Page loaded
    Steps:
      1. Click theme toggle
      2. Check background color
      3. Reload page
      4. Verify theme persists
    Expected Result: Theme changes and persists
    Evidence: .sisyphus/evidence/task-3-theme.{ext}
  ```

  **Commit**: YES (group with 1)

- [ ] 4. Create theme context and provider

  **What to do**:
  - Create custom hook useTheme that wraps next-themes
  - Add theme-aware component utilities
  - Create CSS variables for theme tokens

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 1)
  - **Blocked By**: Task 3

  **Acceptance Criteria**:
  - [ ] useTheme hook available
  - [ ] Components can access theme value

- [ ] 5. Create base layout with navigation

  **What to do**:
  - Create src/app/layout.tsx with proper metadata
  - Add navbar with: Logo, Navigation links, Theme toggle, Login/Register buttons
  - Add footer
  - Make responsive (hamburger on mobile)

  **Must NOT do**:
  - Include authenticated nav before auth
  - Hardcode colors

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: UI component with styling

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 1)
  - **Blocked By**: Task 1

  **References**:
  - Current Streamlit navbar for reference

  **Acceptance Criteria**:
  - [ ] Navbar shows on all pages
  - [ ] Responsive on mobile

  **QA Scenarios**:
  ```
  Scenario: Navbar renders correctly
    Tool: Playwright
    Preconditions: Page loaded
    Steps:
      1. Navigate to homepage
      2. Verify navbar visible
      3. Resize to mobile
      4. Verify hamburger appears
    Evidence: .sisyphus/evidence/task-5-navbar.{ext}
  ```

- [ ] 6. Set up FastAPI backend skeleton

  **What to do**:
  - Create backend/ directory
  - Create main.py with FastAPI app
  - Add CORS, static files config
  - Create requirements.txt for backend
  - Add /api/health endpoint
  - Set up uvicorn to run separately

  **Must NOT do**:
  - Mix frontend/backend in same process (production)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Standard API setup

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 1)
  - **Blocked By**: Task 1 (can start immediately)

  **References**:
  - FastAPI docs

  **Acceptance Criteria**:
  - [ ] `uvicorn backend.main:app` runs
  - [ ] /api/health returns 200

---

### Wave 2: Authentication Tasks

- [ ] 7. Design and implement landing page

  **What to do**:
  - Create src/app/page.tsx (landing)
  - Hero section with gradient background (#74b3ce)
  - Features grid (4 cards matching design):
    - Pose Estimation - 🧠
    - Multi-Model Analysis - 📊
    - Privacy First - 🔒
    - Interactive Visualization - 📈
  - CTA button "Get Started" → /register
  - Testimonials section (placeholder)
  - Footer with links

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: Landing page with modern UI
  - **Skills**: [`frontend-ui-ux`]

  **Parallelization**:
  - **Can Run In Parallel**: NO (Wave 2 - depends on 1-6)
  - **Blocks**: None (parallel with 8-12)
  - **Blocked By**: Tasks 1-6

  **Acceptance Criteria**:
  - [ ] Landing page renders
  - [ ] Features display correctly
  - [ ] CTA button works
  - [ ] Dark/light mode works

  **QA Scenarios**:
  ```
  Scenario: Landing page renders correctly
    Tool: Playwright
    Preconditions: Server running
    Steps:
      1. Navigate to /
      2. Screenshot hero section
      3. Verify feature cards
      4. Toggle theme
      5. Screenshot again
    Expected Result: All sections visible, theme works
    Evidence: .sisyphus/evidence/task-7-landing.{ext}
  ```

- [ ] 8. Create login page with form

  **What to do**:
  - Create src/app/login/page.tsx
  - Clean form with email/username + password
  - "Remember me" checkbox
  - "Forgot password?" link (UI only)
  - Form validation (required fields)
  - Error messages for invalid credentials
  - Login button with loading state
  - Link to /register
  - Theme-aware styling

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`

  **Parallelization**:
  - **Can Run In Parallel**: NO (Wave 2)
  - **Parallel Group**: Wave 2 (with Task 9)
  - **Blocked By**: Tasks 1-6

  **Acceptance Criteria**:
  - [ ] Form renders
  - [ ] Validation works
  - [ ] Wrong credentials show error
  - [ ] Success redirects to dashboard

  **QA Scenarios**:
  ```
  Scenario: Login form validation
    Tool: Playwright
    Steps:
      1. Submit empty form
      2. Verify validation messages
      3. Submit invalid credentials
      4. Verify error message
      5. Submit valid credentials
      6. Verify redirect
    Expected Result: All validations work
    Evidence: .sisyphus/evidence/task-8-login.{ext}
  ```

- [ ] 9. Create register page with role selection

  **What to do**:
  - Create src/app/register/page.tsx
  - Toggle: Professional vs Parent (two cards)
  - Form fields based on role:
    - Professional: Name, Email, Password, Specialty (optional)
    - Parent: Name, Email, Password
  - Terms checkbox
  - Form validation
  - Success → redirect to login with message
  - Link to /login

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`

  **Parallelization**:
  - **Can Run In Parallel**: NO (Wave 2)
  - **Parallel Group**: Wave 2 (with Task 8)
  - **Blocked By**: Tasks 1-6

  **Acceptance Criteria**:
  - [ ] Role selection works
  - [ ] Form shows correct fields
  - [ ] Registration creates user

- [ ] 10. Set up NextAuth.js or custom auth

  **What to do**:
  - Install next-auth or design custom JWT auth
  - Configure providers (credentials)
  - Create auth options
  - Set up session management
  - Protect routes with middleware
  - Store tokens in cookies

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Complex auth security

  **Parallelization**:
  - **Can Run In Parallel**: NO (Wave 2)
  - **Parallel Group**: Wave 2 (with Tasks 7-9)
  - **Blocked By**: Tasks 1-6, 11 (database needed for auth)

  **Acceptance Criteria**:
  - [ ] Login creates session
  - [ ] Session persists
  - [ ] Logout clears session

- [ ] 11. Create database schema (SQLite + Prisma)

  **What to do**:
  - Install Prisma
  - Initialize with SQLite
  - Define schema:
    ```prisma
    model User {
      id        Int      @id @default(autoincrement())
      email     String   @unique
      password  String
      name     String
      role     Role     @default(PARENT)
      avatar   String?
      specialty String?
      createdAt DateTime @default(now())
      updatedAt DateTime @updatedAt
      
      // Relations
      patients  Patient[]
      sentMessages    Message[]
      receivedMessages Message[]
    }
    
    model Patient {
      id          Int      @id @default(autoincrement())
      name       String
      dateOfBirth DateTime?
      gender     String?
      parentId   Int?
      parent    User?    @relation("ParentRelation", fields: [parentId], references: [id])
      professionalId Int
      professional User @relation("ProfessionalRelation", fields: [professionalId], references: [id])
      assessments Assessment[]
      createdAt   DateTime @default(now())
    }
    
    model Assessment {
      id        Int      @id @default(autoincrement())
      patientId Int
      patient  Patient  @relation(fields: [patientId], references: [id])
      score    Float
      risk     String
      status   String   @default("pending")
      result   String?
      createdAt DateTime @default(now())
    }
    
    model Message {
      id        Int      @id @default(autoincrement())
      senderId  Int
      sender    User     @relation("SentMessages", fields: [senderId], references: [id])
      receiverId Int
      receiver  User     @relation("ReceivedMessages", fields: [receiverId], references: [id])
      content   String
      read      Boolean  @default(false)
      createdAt DateTime @default(now())
    }
    
    enum Role {
      PROFESSIONAL
      PARENT
    }
    ```
  - Run prisma db push

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`

  **Parallelization**:
  - **Can Run In Parallel**: NO (Wave 2)
  - **Parallel Group**: Wave 2 (with Tasks 7-10)
  - **Blocked By**: Task 6

  **Acceptance Criteria**:
  - [ ] Database created
  - [ ] Tables match schema

- [ ] 12. Implement backend auth routes

  **What to do**:
  - Create backend/routes/auth.py
  - POST /api/auth/register
  - POST /api/auth/login
  - POST /api/auth/logout
  - GET /api/auth/me
  - Add JWT token generation
  - Connect to frontend

  **Parallelization**:
  - **Can Run In Parallel**: NO (Wave 2)
  - **Parallel Group**: Wave 2 (with Tasks 7-11)
  - **Blocked By**: Task 11

  **Acceptance Criteria**:
  - [ ] /api/auth/register works
  - [ ] /api/auth/login works
  - [ ] Returns valid token

---

### Wave 3: Professional Portal

- [ ] 13. Create professional layout and sidebar

  **What to do**:
  - Create src/app/professional/layout.tsx
  - Sidebar with navigation:
    - Dashboard
    - Patients
    - Assessments
    - Messages
    - Profile
  - Top bar with user info
  - Responsive design

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`

  **Parallelization**:
  - **Can Run In Parallel**: NO (Wave 3)
  - **Blocks**: Tasks 14-18
  - **Blocked By**: Tasks 7-12

  **Acceptance Criteria**:
  - [ ] Sidebar shows
  - [ ] Navigation works

- [ ] 14. Professional dashboard with stats

  **What to do**:
  - Create src/app/professional/page.tsx
  - Stats cards:
    - Total Patients
    - Pending Assessments
    - Messages (unread)
    - This Month's Assessments
  - Recent patients list
  - Recent activity

  **QA Scenarios**:
  ```
  Scenario: Dashboard shows stats
    Tool: Playwright
    Steps:
      1. Login as professional
      2. Navigate to /professional
      3. Verify stat cards
    Expected Result: Stats displayed
  ```

- [ ] 15. Patient management (list, add, view)

  **What to do**:
  - Create src/app/professional/patients/page.tsx (list)
  - Create src/app/professional/patients/add/page.tsx
  - Create src/app/professional/patients/[id]/page.tsx (view)
  - Table with search/filter
  - Add patient form
  - Patient detail view

  **Parallelization**:
  - **Can Run In Parallel**: NO (Wave 3)
  - **Parallel Group**: Wave 3 (with Tasks 13-18)
  - **Blocked By**: Task 13

- [ ] 16. Assessment runs and results view

  **What to do**:
  - Create src/app/professional/assessments/page.tsx
  - Run assessment form (select patient, upload data)
  - Results display with model predictions
  - Risk level indicator

  **Acceptance Criteria**:
  - [ ] Can run inference
  - [ ] Results show

- [ ] 17. Profile page with avatar upload

  **What to do**:
  - Create src/app/professional/profile/page.tsx
  - Edit profile form
  - Avatar upload with preview
  - Save/cancel buttons

  **Acceptance Criteria**:
  - [ ] Form saves
  - [ ] Avatar uploads

- [ ] 18. Reuse ML model inference endpoints

  **What to do**:
  - Create backend/routes/inference.py
  - POST /api/inference/predict
  - Integrate existing ML model loaders
  - Return formatted predictions

  **Parallelization**:
  - **Can Run In Parallel**: NO (Wave 3)
  - **Parallel Group**: Wave 3 (with Task 16)
  - **Blocked By**: Task 6, 16

---

### Wave 4: Parent Portal (Parallel with Wave 3)

- [ ] 19. Create parent layout and sidebar

  **What to do**:
  - Create src/app/parent/layout.tsx
  - Sidebar navigation:
    - Dashboard
    - Children
    - Results
    - Messages
    - Profile

- [ ] 20. Parent dashboard

  **What to do**:
  - Create src/app/parent/page.tsx
  - Overview cards:
    - Linked Children
    - Recent Results
    - Unread Messages
  - Quick actions

- [ ] 21. Children profiles management

  **What to do**:
  - Create src/app/parent/children/page.tsx
  - List children
  - Add child form (professional adds, parent views)
  - Child detail view

- [ ] 22. View assessment results

  **What to do**:
  - Create src/app/parent/results/page.tsx
  - List of past assessments
  - Detailed result view
  - Risk level display

- [ ] 23. Parent profile with avatar upload

  **What to do**:
  - Create src/app/parent/profile/page.tsx

- [ ] 24. Link parent to professional

  **What to do**:
  - Parent can request to link to a professional
  - Professional approves/denies
  - Linked parent shows in professional's patients

---

### Wave 5: Messaging System

- [ ] 25. WebSocket server setup in FastAPI

  **What to do**:
  - Install websockets package
  - Create WebSocket endpoint
  - Handle connect/disconnect/message

  **Acceptance Criteria**:
  - [ ] WebSocket connects
  - [ ] Messages broadcast

- [ ] 26. Message database schema

  **What to do**:
  - Add to Prisma schema if needed
  - Message model already in schema

- [ ] 27. Professional messaging UI

  **What to do**:
  - Create src/app/professional/messages/page.tsx
  - Chat list sidebar
  - Chat view with messages
  - Send message input

  **Acceptance Criteria**:
  - [ ] Can view conversations
  - [ ] Can send/receive messages

- [ ] 28. Parent messaging UI

  **What to do**:
  - Create src/app/parent/messages/page.tsx

- [ ] 29. Real-time message delivery

  **What to do**:
  - Connect frontend WebSocket
  - Handle incoming messages
  - Update UI in real-time

- [ ] 30. Message notifications

  **What to do**:
  - Badge on messages nav
  - Toast notifications for new messages

---

### Wave 6: Polish & Integration

- [ ] 31. Add smooth animations (framer-motion)

  **What to do**:
  - Page transition animations
  - Hover effects on cards/buttons
  - Staggered list animations
  - Theme transition animation

  **Acceptance Criteria**:
  - [ ] Page transitions smooth
  - [ ] Hover effects work

- [ ] 32. Mobile responsive design

  **What to do**:
  - Test all pages on mobile
  - Fix layout issues
  - Touch-friendly interactions

- [ ] 33. Error boundaries and loading states

  **What to do**:
  - Add error.js for error boundaries
  - Add loading.js for loading states
  - Graceful error handling

- [ ] 34. SEO metadata

  **What to do**:
  - Add metadata to pages
  - Open Graph tags
  - Favicon

- [ ] 35. Production build optimization

  **What to do**:
  - Run npm run build
  - Fix any errors
  - Optimize bundle

---

## Final Verification Wave

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists. For each "Must NOT Have": search codebase for forbidden patterns.

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run tsc --noEmit + eslint. Review for: any types, empty catches, console.log in prod, unused imports. Check AI slop patterns.

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Execute EVERY QA scenario from EVERY task. Save evidence to .sisyphus/evidence/final-qa/.

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: verify everything in spec was built, nothing beyond spec was built.

---

## Commit Strategy

Wave commits:
- **1**: `chore: initialize Next.js project with Theme setup`
- **2**: `feat: add authentication (login, register)`
- **3**: `feat: add professional portal`
- **4**: `feat: add parent portal`
- **5**: `feat: add real-time messaging`
- **6**: `feat: add animations and polish`

---

## Success Criteria

### Verification Commands
```bash
# Frontend
npm run build  # Expected: success
npm run lint  # Expected: no errors

# Backend
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000  # Expected: starts

# End-to-end
curl http://localhost:3000       # Expected: HTML
curl http://localhost:8000/api/health  # Expected: JSON
```

### Final Checklist
- [ ] All pages render correctly
- [ ] Theme toggle smooth
- [ ] Auth flow works
- [ ] Professional portal functional
- [ ] Parent portal functional
- [ ] Messaging works
- [ ] No critical errors
- [ ] Mobile responsive
- [ ] Production build passes