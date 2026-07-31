import { useProject } from "../../context/ProjectContext.jsx";

export default function Constraints() {
  const {
    phases,
    setPhases,
    discountRate,
    setDiscountRate,
    presaleLeadMonths,
    setPresaleLeadMonths,
    projectStart,
    setProjectStart,
  } = useProject();

  const updatePhase = (index, field, value) => {
    setPhases((prev) =>
      prev.map((p, i) => (i === index ? { ...p, [field]: value } : p))
    );
  };

  const setPhaseCount = (count) => {
    const n = Math.max(1, Math.min(8, Number(count) || 1));
    setPhases((prev) => {
      const next = [];
      for (let i = 0; i < n; i += 1) {
        next.push(
          prev[i] || {
            name: `phase_${i + 1}`,
            start_month: i * 6,
            cash_flow_floor_usd: 0,
            max_units: 20,
            competing_listings: 30,
          }
        );
      }
      return next;
    });
  };

  return (
    <section aria-labelledby="constraints-heading">
      <h2 id="constraints-heading" className="font-display text-lg font-semibold">
        Constraints
      </h2>
      <p className="mt-1 text-sm text-muted">
        Phase timing, cash-flow floors, and release caps. Floors bind on expected
        revenue — simulate to see breach risk.
      </p>

      <div className="mt-4 grid gap-4 sm:grid-cols-3">
        <label className="block text-sm">
          <span className="text-muted">Project start</span>
          <input
            type="date"
            value={projectStart}
            onChange={(e) => setProjectStart(e.target.value)}
            className="mt-1 w-full border border-rule bg-paper px-3 py-2 font-mono tabular-nums focus-visible:outline focus-visible:outline-2 focus-visible:outline-signal"
          />
        </label>
        <label className="block text-sm">
          <span className="text-muted">Discount rate (annual)</span>
          <input
            type="number"
            step="0.01"
            min="0"
            max="0.5"
            value={discountRate}
            onChange={(e) => setDiscountRate(Number(e.target.value))}
            className="mt-1 w-full border border-rule bg-paper px-3 py-2 font-mono tabular-nums focus-visible:outline focus-visible:outline-2 focus-visible:outline-signal"
          />
        </label>
        <label className="block text-sm">
          <span className="text-muted">Presale lead (months)</span>
          <input
            type="number"
            min="0"
            max="60"
            value={presaleLeadMonths}
            onChange={(e) => setPresaleLeadMonths(Number(e.target.value))}
            className="mt-1 w-full border border-rule bg-paper px-3 py-2 font-mono tabular-nums focus-visible:outline focus-visible:outline-2 focus-visible:outline-signal"
          />
        </label>
      </div>

      <label className="mt-4 block text-sm">
        <span className="text-muted">Number of phases</span>
        <input
          type="number"
          min="1"
          max="8"
          value={phases.length}
          onChange={(e) => setPhaseCount(e.target.value)}
          className="mt-1 w-32 border border-rule bg-paper px-3 py-2 font-mono tabular-nums focus-visible:outline focus-visible:outline-2 focus-visible:outline-signal"
        />
      </label>

      <div className="mt-4 overflow-x-auto">
        <table className="w-full min-w-[36rem] border-collapse text-sm">
          <thead>
            <tr className="border-b border-rule text-left text-muted">
              <th className="py-2 pr-3 font-medium">Name</th>
              <th className="py-2 pr-3 font-medium">Start (mo)</th>
              <th className="py-2 pr-3 font-medium">CF min ($)</th>
              <th className="py-2 pr-3 font-medium">Max units</th>
              <th className="py-2 font-medium">Competing listings</th>
            </tr>
          </thead>
          <tbody>
            {phases.map((phase, i) => (
              <tr key={i} className="border-b border-rule/70">
                <td className="py-2 pr-3">
                  <input
                    value={phase.name}
                    onChange={(e) => updatePhase(i, "name", e.target.value)}
                    className="w-full border border-rule bg-paper px-2 py-1 focus-visible:outline focus-visible:outline-2 focus-visible:outline-signal"
                  />
                </td>
                <td className="py-2 pr-3">
                  <input
                    type="number"
                    value={phase.start_month}
                    onChange={(e) =>
                      updatePhase(i, "start_month", Number(e.target.value))
                    }
                    className="w-20 border border-rule bg-paper px-2 py-1 font-mono tabular-nums focus-visible:outline focus-visible:outline-2 focus-visible:outline-signal"
                  />
                </td>
                <td className="py-2 pr-3">
                  <input
                    type="number"
                    value={phase.cash_flow_floor_usd}
                    onChange={(e) =>
                      updatePhase(
                        i,
                        "cash_flow_floor_usd",
                        Number(e.target.value)
                      )
                    }
                    className="w-28 border border-rule bg-paper px-2 py-1 font-mono tabular-nums focus-visible:outline focus-visible:outline-2 focus-visible:outline-signal"
                  />
                </td>
                <td className="py-2 pr-3">
                  <input
                    type="number"
                    value={phase.max_units ?? ""}
                    onChange={(e) =>
                      updatePhase(
                        i,
                        "max_units",
                        e.target.value === "" ? null : Number(e.target.value)
                      )
                    }
                    className="w-20 border border-rule bg-paper px-2 py-1 font-mono tabular-nums focus-visible:outline focus-visible:outline-2 focus-visible:outline-signal"
                  />
                </td>
                <td className="py-2">
                  <input
                    type="number"
                    value={phase.competing_listings ?? ""}
                    onChange={(e) =>
                      updatePhase(
                        i,
                        "competing_listings",
                        e.target.value === "" ? null : Number(e.target.value)
                      )
                    }
                    className="w-24 border border-rule bg-paper px-2 py-1 font-mono tabular-nums focus-visible:outline focus-visible:outline-2 focus-visible:outline-signal"
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
