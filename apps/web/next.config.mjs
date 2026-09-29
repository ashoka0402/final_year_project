/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The dev-tools badge floats over the map legend; the app is judged on the map.
  devIndicators: false,
  transpilePackages: ["deck.gl", "@deck.gl/mapbox", "@deck.gl/react"],
  async rewrites() {
    // Proxy the API in dev so the browser talks to one origin (no CORS preflight
    // on every map interaction, and NEXT_PUBLIC_API_URL stays server-side).
    const api = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    return [{ source: "/api/v1/:path*", destination: `${api}/api/v1/:path*` }];
  },
};
export default nextConfig;
