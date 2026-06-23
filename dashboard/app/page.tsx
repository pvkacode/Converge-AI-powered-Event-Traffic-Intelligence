import { HeroPage } from "@/components/hero/HeroPage";
import { loadHeroStats } from "@/lib/hero-stats";

export const revalidate = 30;

export default function LandingPage() {
  const stats = loadHeroStats();
  return <HeroPage stats={stats} />;
}
