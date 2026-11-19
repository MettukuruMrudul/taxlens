// ESLint 9 "flat" config. Next.js 16 no longer lints during `next build`,
// so CI runs `npm run lint` as its own step.
import nextVitals from "eslint-config-next/core-web-vitals";

const config = [
  ...nextVitals,
  { ignores: [".next/**", "node_modules/**", "next-env.d.ts"] },
];

export default config;
