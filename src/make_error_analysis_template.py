from __future__ import annotations

from pathlib import Path


def main() -> int:
    try:
        from openpyxl import Workbook
        from openpyxl.worksheet.datavalidation import DataValidation
    except Exception as exc:
        print("[ERROR] openpyxl is required to generate the template.")
        print(f"Details: {exc}")
        return 1

    base_dir = Path(__file__).resolve().parent.parent
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "error_analysis_template.xlsx"

    wb = Workbook()
    ws = wb.active
    ws.title = "Error Analysis"

    headers = [
        "Seed",
        "Algorithm",
        "true_label",
        "pred_label",
        "prob_pos",
        "error_type",
        "Category",
        "Notes",
        "review_raw",
    ]
    ws.append(headers)

    ws.column_dimensions["A"].width = 10
    ws.column_dimensions["B"].width = 24
    ws.column_dimensions["C"].width = 12
    ws.column_dimensions["D"].width = 12
    ws.column_dimensions["E"].width = 12
    ws.column_dimensions["F"].width = 14
    ws.column_dimensions["G"].width = 20
    ws.column_dimensions["H"].width = 40
    ws.column_dimensions["I"].width = 120

    options = [
        "Negation",
        "Sarcasm",
        "Mixed sentiment",
        "Slang/Vocabulary",
        "Long review",
        "Other",
    ]
    validation = DataValidation(
        type="list",
        formula1='"' + ",".join(options) + '"',
        allow_blank=True,
        showDropDown=True,
    )
    ws.add_data_validation(validation)
    validation.add("G2:G5000")

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = "A1:I1"

    wb.save(output_path)
    print(f"[SAVED] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
