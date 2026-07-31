import { useMemo } from "react";
import { useProject } from "../../context/ProjectContext.jsx";
import { formatUsdPerSqft } from "../../utils/format.js";
import EmptyState from "../common/EmptyState.jsx";

/** Map a $/sqft into the signal→paper→warn continuum. */
function priceColor(value, min, max) {
  if (value == null || max <= min) return "var(--rule)";
  const t = (value - min) / (max - min);
  // Low = muted paper-adjacent, high = signal teal
  const r = Math.round(22 + (15 - 22) * t);
  const g = Math.round(32 + (110 - 32) * t);
  const b = Math.round(43 + (92 - 43) * t);
  return `rgb(${r},${g},${b})`;
}

export default function BuildingHeatmap() {
  const { planResult, inventory, selectedUnitId, setSelectedUnitId } =
    useProject();

  const buildings = useMemo(() => {
    if (!planResult?.plan?.length || !inventory.length) return [];
    const byId = new Map(inventory.map((u) => [String(u.unit_id), u]));
    const priceById = new Map(
      planResult.plan.map((r) => [String(r.unit_id), r.price_ppsf])
    );
    const groups = new Map();
    for (const unit of inventory) {
      const building = unit.building || "Building";
      if (!groups.has(building)) groups.set(building, []);
      groups.get(building).push({
        ...unit,
        price_ppsf: priceById.get(String(unit.unit_id)),
      });
    }
    return [...groups.entries()].map(([name, units]) => {
      const floors = [...new Set(units.map((u) => Number(u.floor)))].sort(
        (a, b) => b - a
      );
      // Position within floor: stable order by unit_id
      const byFloor = new Map();
      for (const u of units) {
        const f = Number(u.floor);
        if (!byFloor.has(f)) byFloor.set(f, []);
        byFloor.get(f).push(u);
      }
      for (const list of byFloor.values()) {
        list.sort((a, b) => String(a.unit_id).localeCompare(String(b.unit_id)));
      }
      const maxPos = Math.max(
        ...[...byFloor.values()].map((list) => list.length),
        1
      );
      const prices = units.map((u) => u.price_ppsf).filter((p) => p != null);
      const min = prices.length ? Math.min(...prices) : 0;
      const max = prices.length ? Math.max(...prices) : 1;
      return { name, floors, byFloor, maxPos, min, max };
    });
  }, [planResult, inventory]);

  if (!buildings.length) {
    return (
      <EmptyState
        title="Building cross-section"
        body="After Optimize, each cell is a unit — floor × stack — filled by recommended $/sqft."
      />
    );
  }

  return (
    <section aria-labelledby="heatmap-heading">
      <h2 id="heatmap-heading" className="font-display text-lg font-semibold">
        Building cross-section
      </h2>
      <p className="mt-1 text-sm text-muted">
        Recommended $/sqft by floor and stack. Click a cell to inspect demand.
      </p>

      <div className="mt-6 space-y-10">
        {buildings.map((b) => (
          <div key={b.name}>
            <p className="font-display text-sm font-semibold tracking-wide">
              {b.name}
            </p>
            <div className="mt-3 overflow-x-auto">
              <div
                className="inline-grid gap-1"
                style={{
                  gridTemplateColumns: `2.5rem repeat(${b.maxPos}, minmax(2.75rem, 1fr))`,
                }}
              >
                {b.floors.map((floor) => (
                  <div key={floor} className="contents">
                    <div className="flex items-center justify-end pr-2 font-mono text-xs tabular-nums text-muted">
                      {floor}
                    </div>
                    {Array.from({ length: b.maxPos }, (_, pos) => {
                      const unit = b.byFloor.get(floor)?.[pos];
                      if (!unit) {
                        return (
                          <div
                            key={`${floor}-${pos}`}
                            className="h-10 border border-transparent"
                          />
                        );
                      }
                      const selected =
                        String(unit.unit_id) === String(selectedUnitId);
                      const priced = unit.price_ppsf != null;
                      return (
                        <button
                          key={unit.unit_id}
                          type="button"
                          title={`${unit.unit_id} · ${formatUsdPerSqft(unit.price_ppsf)}`}
                          onClick={() => setSelectedUnitId(unit.unit_id)}
                          className={[
                            "flex h-10 flex-col items-center justify-center border text-[10px] leading-tight transition-transform",
                            "focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-ink",
                            selected ? "ring-2 ring-ink ring-offset-1" : "",
                            priced ? "text-paper" : "border-rule bg-rule/30 text-muted",
                          ].join(" ")}
                          style={
                            priced
                              ? {
                                  background: priceColor(
                                    unit.price_ppsf,
                                    b.min,
                                    b.max
                                  ),
                                  borderColor: "transparent",
                                }
                              : undefined
                          }
                        >
                          <span className="font-mono tabular-nums opacity-90">
                            {priced
                              ? Math.round(unit.price_ppsf)
                              : "—"}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                ))}
              </div>
            </div>
            <div className="mt-3 flex items-center gap-2 text-xs text-muted">
              <span>Lower $/sqft</span>
              <div
                className="h-2 w-24"
                style={{
                  background: `linear-gradient(to right, ${priceColor(
                    b.min,
                    b.min,
                    b.max
                  )}, ${priceColor(b.max, b.min, b.max)})`,
                }}
              />
              <span>Higher $/sqft</span>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
