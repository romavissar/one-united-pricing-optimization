/**
 * Non-dismissible banner when the demand model is not calibrated on real MLS.
 * AGENTS.md §4 / PROJECT_BRIEF Phase 7.
 */
export default function CalibrationBanner({ visible }) {
  if (!visible) return null;
  return (
    <div
      role="status"
      className="sticky top-0 z-50 border-b border-warn/40 bg-warn/15 px-4 py-3 text-sm text-ink md:px-8"
    >
      <p className="font-medium text-warn">
        Illustrative output — demand model fitted on synthetic data, not market
        data.
      </p>
    </div>
  );
}
