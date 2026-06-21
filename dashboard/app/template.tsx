"use client";

import { motion, useReducedMotion } from "motion/react";

const spring = { type: "spring" as const, stiffness: 300, damping: 30 };

// Avoid initial opacity:0 on SSR — if CSS/JS chunks fail to load, that inline style
// leaves the page invisible or unstyled. initial={false} skips the hidden state.
export default function Template({ children }: { children: React.ReactNode }) {
  const reduceMotion = useReducedMotion();

  if (reduceMotion) {
    return <>{children}</>;
  }

  return (
    <motion.div initial={false} animate={{ opacity: 1, y: 0 }} transition={spring}>
      {children}
    </motion.div>
  );
}
