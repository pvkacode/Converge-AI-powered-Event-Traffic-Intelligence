/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The CSV datasets live in the sibling `outputs/` directory and are read at
  // runtime via the Node fs API in server code (lib/csv.ts). Nothing is bundled.
};

export default nextConfig;
