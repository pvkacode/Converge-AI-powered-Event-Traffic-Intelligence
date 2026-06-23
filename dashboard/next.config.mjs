/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  experimental: {
    optimizePackageImports: ["recharts", "@phosphor-icons/react", "motion/react"],
  },
};

export default nextConfig;
