"use client";
import { motion, useReducedMotion } from "motion/react";

const spring = { type: "spring" as const, stiffness: 300, damping: 30 };

export default function Template({ children }: { children: React.ReactNode }) {
  const reduceMotion = useReducedMotion();
  return (
    <motion.div
      initial={{ opacity: 0, y: reduceMotion ? 0 : 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={reduceMotion ? { duration: 0 } : spring}
      style={{ willChange: "opacity, transform" }}
    >
      {children}
    </motion.div>
  );
}
