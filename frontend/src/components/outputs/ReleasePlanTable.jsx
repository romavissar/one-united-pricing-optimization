import { useMemo, useState } from "react";
import {
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getSortedRowModel,
  useReactTable,
} from "@tanstack/react-table";
import { useProject } from "../../context/ProjectContext.jsx";
import { formatPct, formatUsd, formatUsdPerSqft } from "../../utils/format.js";
import EmptyState from "../common/EmptyState.jsx";

function inventoryLookup(inventory) {
  const map = new Map();
  for (const row of inventory) map.set(String(row.unit_id), row);
  return map;
}

export default function ReleasePlanTable() {
  const {
    planResult,
    inventory,
    selectedUnitId,
    setSelectedUnitId,
  } = useProject();
  const [sorting, setSorting] = useState([]);
  const [globalFilter, setGlobalFilter] = useState("");

  const units = useMemo(() => inventoryLookup(inventory), [inventory]);

  const data = useMemo(() => {
    if (!planResult?.plan) return [];
    return planResult.plan.map((row) => {
      const inv = units.get(String(row.unit_id)) || {};
      return {
        ...row,
        unit_type: inv.unit_type,
        floor: inv.floor,
        living_area_sqft: inv.living_area_sqft,
        view: inv.view,
        building: inv.building,
      };
    });
  }, [planResult, units]);

  const columns = useMemo(
    () => [
      { accessorKey: "unit_id", header: "Unit" },
      { accessorKey: "unit_type", header: "Type" },
      {
        accessorKey: "floor",
        header: "Floor",
        cell: (info) => (
          <span className="font-mono tabular-nums">{info.getValue() ?? "—"}</span>
        ),
      },
      {
        accessorKey: "living_area_sqft",
        header: "Sqft",
        cell: (info) => (
          <span className="font-mono tabular-nums">
            {info.getValue()?.toLocaleString?.() ?? "—"}
          </span>
        ),
      },
      { accessorKey: "view", header: "View" },
      { accessorKey: "phase_name", header: "Phase" },
      {
        accessorKey: "price_ppsf",
        header: "$/sqft",
        cell: (info) => (
          <span className="font-mono tabular-nums text-signal">
            {formatUsdPerSqft(info.getValue())}
          </span>
        ),
      },
      {
        accessorKey: "total_price_usd",
        header: "Total",
        cell: (info) => (
          <span className="font-mono tabular-nums">
            {formatUsd(info.getValue())}
          </span>
        ),
      },
      {
        accessorKey: "sale_probability",
        header: "P(sell)",
        cell: (info) => (
          <span className="font-mono tabular-nums">
            {formatPct(info.getValue())}
          </span>
        ),
      },
      {
        accessorKey: "expected_revenue_usd",
        header: "E[rev]",
        cell: (info) => (
          <span className="font-mono tabular-nums">
            {formatUsd(info.getValue())}
          </span>
        ),
      },
    ],
    []
  );

  const table = useReactTable({
    data,
    columns,
    state: { sorting, globalFilter },
    onSortingChange: setSorting,
    onGlobalFilterChange: setGlobalFilter,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
  });

  // A cell that starts with = + - or @ is a formula to Excel, Sheets and
  // Numbers, and unit ids come from a spreadsheet the developer uploaded. A
  // prefixed apostrophe forces the cell to text; quotes are doubled and any
  // value containing a quote, comma, or newline is wrapped, which the previous
  // comma-only rule did not do.
  const csvCell = (value) => {
    if (value == null) return "";
    let s = String(value);
    if (/^[=+\-@\t\r]/.test(s)) s = `'${s}`;
    return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };

  const exportCsv = () => {
    const headers = columns.map((c) => c.header);
    const keys = columns.map((c) => c.accessorKey);
    const lines = [
      headers.map(csvCell).join(","),
      ...data.map((row) => keys.map((k) => csvCell(row[k])).join(",")),
    ];
    const blob = new Blob([lines.join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "release_plan.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  if (!planResult?.plan?.length) {
    return (
      <EmptyState
        title="No release plan yet"
        body="Set constraints, then Optimize. The table fills with unit, phase, and price."
      />
    );
  }

  return (
    <section aria-labelledby="plan-heading">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 id="plan-heading" className="font-display text-lg font-semibold">
            Release plan
          </h2>
          <p className="mt-1 text-sm text-muted">
            Click a row to see its demand curve.{" "}
            <span className="font-mono tabular-nums">
              {planResult.units_released}
            </span>{" "}
            units · objective{" "}
            <span className="font-mono tabular-nums text-signal">
              {formatUsd(planResult.objective_usd)}
            </span>
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <input
            value={globalFilter ?? ""}
            onChange={(e) => setGlobalFilter(e.target.value)}
            placeholder="Filter…"
            className="border border-rule bg-paper px-3 py-1.5 text-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-signal"
          />
          <button
            type="button"
            onClick={exportCsv}
            className="border border-rule px-3 py-1.5 text-sm hover:border-ink/40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-signal"
          >
            Export CSV
          </button>
        </div>
      </div>

      <div className="mt-4 max-h-[28rem] overflow-auto border border-rule">
        <table className="w-full min-w-[56rem] border-collapse text-sm">
          <thead className="sticky top-0 bg-paper">
            {table.getHeaderGroups().map((hg) => (
              <tr key={hg.id} className="border-b border-rule text-left text-muted">
                {hg.headers.map((header) => (
                  <th
                    key={header.id}
                    className="cursor-pointer px-2 py-2 font-medium"
                    onClick={header.column.getToggleSortingHandler()}
                  >
                    {flexRender(
                      header.column.columnDef.header,
                      header.getContext()
                    )}
                    {{ asc: " ↑", desc: " ↓" }[header.column.getIsSorted()] ||
                      ""}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.map((row) => {
              const selected = row.original.unit_id === selectedUnitId;
              return (
                <tr
                  key={row.id}
                  onClick={() => setSelectedUnitId(row.original.unit_id)}
                  className={[
                    "cursor-pointer border-b border-rule/60",
                    selected ? "bg-signal/10" : "hover:bg-rule/20",
                  ].join(" ")}
                >
                  {row.getVisibleCells().map((cell) => (
                    <td key={cell.id} className="px-2 py-1.5">
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext()
                      )}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
