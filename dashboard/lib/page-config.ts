/** ISR window for pipeline dashboard pages (CSV exports change on re-run, not live).
 *  Next.js requires `export const revalidate = 30` as a literal in each page/route —
 *  do not import this constant into page config exports. */
export const PAGE_REVALIDATE_SECONDS = 30;
