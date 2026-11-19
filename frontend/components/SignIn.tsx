"use client";

import { useState } from "react";

import { signIn } from "@/lib/api";

export default function SignIn({ onSignedIn }: { onSignedIn: () => void }) {
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await signIn(name.trim());
      onSignedIn();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="signin">
      <form onSubmit={submit}>
        <h1>TaxLens</h1>
        <p className="muted">
          Answers from IRS forms and publications for the tax year you mean, with the exact line cited.
        </p>
        <label htmlFor="name">Your name</label>
        <input id="name" value={name} onChange={(e) => setName(e.target.value)} autoComplete="username"
               pattern="[A-Za-z0-9_.@\-]{2,64}" title="2 to 64 letters, numbers, or . _ @ -" required />
        <button className="btn" type="submit" disabled={busy || name.trim().length < 2}>
          {busy ? "Signing in…" : "Sign in"}
        </button>
        {error && <p className="error" role="alert">{error}</p>}
        <p className="muted" style={{ fontSize: "0.85rem" }}>
          Your name keeps your question history separate. This is a development sign-in.
        </p>
      </form>
    </main>
  );
}
