# JASMINE Frontend

Next.js 14 web application for the JASMINE autism screening system.

## Prerequisites

- Node.js 18+
- Python 3.13 (for the ML backend)
- Firebase project (or use built-in demo accounts)

## Setup

### 1. Install Dependencies

```bash
npm install --legacy-peer-deps
```

### 2. Environment Variables

Create a `.env.local` file:

```env
NEXT_PUBLIC_FIREBASE_API_KEY=your_api_key
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=your_project.firebaseapp.com
NEXT_PUBLIC_FIREBASE_PROJECT_ID=your_project_id
NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=your_project.appspot.com
NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=your_sender_id
NEXT_PUBLIC_FIREBASE_APP_ID=your_app_id
NEXT_PUBLIC_ML_BACKEND_URL=http://localhost:8000
```

### 3. Start ML Backend

```bash
# In a separate terminal
uvicorn backend.main:app --reload --port 8000
```

### 4. Start Frontend

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Demo Accounts

| Email | Password | Role |
|-------|----------|------|
| parent@demo.com | demo123 | Parent |
| doctor@demo.com | demo123 | Professional |

Demo accounts work without Firebase (localStorage fallback).

## Environment

- **Frontend**: http://localhost:3000
- **Backend**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Build

```bash
npm run build
npm start
```
