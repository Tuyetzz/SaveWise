import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Dev mode blocks cross-origin requests to dev assets by default; needed
  // when running `next dev` behind the nginx proxy. Production (`next
  // start`) is unaffected.
  allowedDevOrigins: ["hackathon.marcusnguyen.dev"],
};

export default nextConfig;
