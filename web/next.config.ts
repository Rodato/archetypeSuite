import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Emit a self-contained server bundle for slim Docker images / non-Vercel hosts.
  output: "standalone",
};

export default nextConfig;
