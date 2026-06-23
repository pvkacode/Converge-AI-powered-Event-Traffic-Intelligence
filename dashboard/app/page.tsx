import { HeroPage } from "@/components/hero/HeroPage";
import { loadHeroStats } from "@/lib/hero-stats";
import { PAGE_REVALIDATE_SECONDS } from "@/lib/page-config";

export const revalidate = PAGE_REVALIDATE_SECONDS;

export default function LandingPage() {
  const stats = loadHeroStats();
  return <HeroPage stats={stats} />;
}
