import { NextResponse } from 'next/server';

export async function GET() {
  // When Next.js app is running in local dev mode, the base URL points to the
  // `promptfoo view` server.
  return NextResponse.json({
    apiBaseUrl: 'http://localhost:15500'
  });
}
