import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import Papa from "papaparse";
import * as XLSX from "xlsx";
import { useProject } from "../../context/ProjectContext.jsx";
import EmptyState from "../common/EmptyState.jsx";

function parseFile(file) {
  const name = file.name.toLowerCase();
  if (name.endsWith(".csv") || name.endsWith(".txt")) {
    return new Promise((resolve, reject) => {
      Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        complete: (result) => {
          if (result.errors?.length) {
            reject(new Error(result.errors[0].message));
            return;
          }
          resolve(result.data);
        },
        error: (err) => reject(err),
      });
    });
  }
  if (name.endsWith(".xlsx") || name.endsWith(".xls")) {
    return file.arrayBuffer().then((buf) => {
      const workbook = XLSX.read(buf, { type: "array" });
      const sheet = workbook.Sheets[workbook.SheetNames[0]];
      return XLSX.utils.sheet_to_json(sheet, { defval: null });
    });
  }
  return Promise.reject(
    new Error("Use a CSV or XLSX inventory file.")
  );
}

export default function InventoryUpload() {
  const { runValidate, validation, busy, setError } = useProject();

  const onDrop = useCallback(
    async (accepted) => {
      const file = accepted[0];
      if (!file) return;
      try {
        const rows = await parseFile(file);
        await runValidate(rows);
      } catch (err) {
        setError(err.message);
      }
    },
    [runValidate, setError]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    multiple: false,
    accept: {
      "text/csv": [".csv", ".txt"],
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [
        ".xlsx",
      ],
      "application/vnd.ms-excel": [".xls"],
    },
    disabled: Boolean(busy),
  });

  return (
    <section aria-labelledby="inventory-heading">
      <h2 id="inventory-heading" className="font-display text-lg font-semibold">
        Inventory
      </h2>
      <p className="mt-1 text-sm text-muted">
        Upload the unit list. Bad rows come back named — nothing is silently
        fixed.
      </p>

      <div
        {...getRootProps()}
        className={[
          "mt-4 cursor-pointer border border-dashed px-5 py-10 text-center transition-colors",
          isDragActive ? "border-signal bg-signal/5" : "border-rule bg-paper",
          busy ? "opacity-60" : "",
        ].join(" ")}
      >
        <input {...getInputProps()} />
        <p className="text-sm text-ink">
          {isDragActive
            ? "Drop to validate"
            : "Drop a CSV or XLSX here, or click to browse"}
        </p>
        <p className="mt-2 text-xs text-muted">
          Needs unit_id, floor, living_area_sqft, beds, unit_type, submarket,
          completion_date, cost_basis_ppsf
        </p>
      </div>

      {!validation && (
        <div className="mt-4">
          <EmptyState
            title="No inventory yet"
            body="Start with backend/data/project_inputs/example_inventory.csv — sixty units across two buildings."
          />
        </div>
      )}

      {validation && (
        <div className="mt-4 border border-rule p-4">
          <dl className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
            <div>
              <dt className="text-muted">Rows in</dt>
              <dd className="font-mono tabular-nums">{validation.rows_in}</dd>
            </div>
            <div>
              <dt className="text-muted">Valid</dt>
              <dd className="font-mono tabular-nums text-signal">
                {validation.rows_valid}
              </dd>
            </div>
            <div>
              <dt className="text-muted">Errors</dt>
              <dd
                className={[
                  "font-mono tabular-nums",
                  validation.errors?.length ? "text-warn" : "",
                ].join(" ")}
              >
                {validation.errors?.length || 0}
              </dd>
            </div>
            <div>
              <dt className="text-muted">Status</dt>
              <dd>{validation.is_valid ? "Ready" : "Needs fixes"}</dd>
            </div>
          </dl>

          {validation.errors?.length > 0 && (
            <div className="mt-4 max-h-48 overflow-auto border-t border-rule pt-3">
              <p className="text-xs font-medium uppercase tracking-wide text-warn">
                Row errors
              </p>
              <ul className="mt-2 space-y-1 text-sm">
                {validation.errors.slice(0, 40).map((e, i) => (
                  <li key={`${e.row_number}-${e.field}-${i}`} className="text-muted">
                    <span className="font-mono tabular-nums text-ink">
                      row {e.row_number}
                    </span>
                    {e.unit_id ? ` (${e.unit_id})` : ""} — {e.field}: {e.message}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
