/**
 * Renders the current numeric value through the supplied formatter.
 *
 * The original Phase-14 spec called for a Framer-style number roll-up; we
 * defer the actual tween to Phase 16 so the renderer is deterministic in
 * jsdom and React 18 strict-mode tests. The CSS `tabular-nums` keeps digit
 * widths stable when the value updates.
 */
export function AnimatedNumber({
  value, format,
}: {
  value: number;
  format: (n: number) => string;
}) {
  return (
    <span data-testid="animated-number" className="tabular-nums">
      {format(value)}
    </span>
  );
}
