from __future__ import annotations

from io import StringIO

import pandas as pd
import streamlit as st

from gst_recon import (
    FIELD_LABELS,
    InputMappingError,
    MissingRequiredColumnsError,
    build_client_report,
    build_follow_up_sheet,
    build_itc_risk_summary,
    build_reconciliation_export,
    build_summary_metrics,
    build_vendor_summary,
    inspect_input_dataframe,
    load_input_file,
    reconcile_invoices,
    standardize_mapped_dataframe,
)


FILE_SIZE_WARNING_BYTES = 50 * 1024 * 1024
MANUAL_SELECT_PLACEHOLDER = "-- Select column --"


st.set_page_config(
    page_title="GST Reconciliation Assistant",
    layout="wide",
)


def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8-sig")


def _status_style(value: str) -> str:
    palette = {
        "MATCHED": "background-color: #d1fae5; color: #065f46;",
        "MISSING_IN_2B": "background-color: #fee2e2; color: #991b1b;",
        "ONLY_IN_2B": "background-color: #ede9fe; color: #5b21b6;",
        "VALUE_MISMATCH": "background-color: #fef3c7; color: #92400e;",
        "DATE_MISMATCH": "background-color: #dbeafe; color: #1d4ed8;",
        "INVALID DATA": "background-color: #fef2f2; color: #b91c1c;",
        "INVALID GSTIN": "background-color: #ffe4e6; color: #be123c;",
        "DATE ERROR": "background-color: #e0f2fe; color: #075985;",
        "AMBIGUOUS MATCH": "background-color: #fce7f3; color: #9d174d;",
        "REVIEW REQUIRED": "background-color: #ede9fe; color: #6d28d9;",
    }
    return palette.get(value, "")


def _row_highlight(row: pd.Series) -> list[str]:
    itc_status = row.get("ITC Status", "")
    if itc_status == "AT RISK":
        style = "background-color: #fee2e2; color: #991b1b;"
    elif itc_status == "NOT ELIGIBLE":
        style = "background-color: #7f1d1d; color: #fef2f2;"
    elif itc_status == "REVIEW REQUIRED":
        style = "background-color: #fef3c7; color: #92400e;"
    elif row.get("Status", "") == "MATCHED":
        style = "background-color: #dcfce7; color: #166534;"
    else:
        style = ""
    return [style] * len(row)


def _render_data_table(df: pd.DataFrame, height: int = 460) -> None:
    styled = df.style.apply(_row_highlight, axis=1)
    st.dataframe(styled, use_container_width=True, height=height)


def _warn_for_file_size(uploaded_file) -> None:
    if uploaded_file is not None and getattr(uploaded_file, "size", 0) > FILE_SIZE_WARNING_BYTES:
        st.warning(
            f"{uploaded_file.name} is larger than 50 MB. Processing may be slower, "
            "and fuzzy matching may be reduced for stability."
        )


def _render_mapping_controls(title: str, inspection_result, key_prefix: str) -> dict[str, str | None]:
    resolved_mapping: dict[str, str | None] = {}
    available_columns = inspection_result.dataframe.columns.tolist()
    st.markdown(f"#### {title} Column Mapping")

    for field, label in FIELD_LABELS.items():
        suggested_value = inspection_result.suggested_mapping.get(field)
        is_ambiguous = field in inspection_result.ambiguous_columns
        requires_manual_selection = is_ambiguous or suggested_value is None

        if requires_manual_selection:
            options = [MANUAL_SELECT_PLACEHOLDER] + available_columns
            default_index = options.index(suggested_value) if suggested_value in available_columns else 0
            help_text = (
                f"Multiple possible columns found: {inspection_result.ambiguous_columns[field]}"
                if is_ambiguous
                else f"No confident match found for {label}. Please select it manually."
            )
            selected_value = st.selectbox(
                f"{title}: {label}",
                options,
                index=default_index,
                key=f"{key_prefix}_{field}",
                help=help_text,
            )
            resolved_mapping[field] = None if selected_value == MANUAL_SELECT_PLACEHOLDER else selected_value
        else:
            resolved_mapping[field] = suggested_value
            st.caption(f"{label}: {suggested_value}")

    with st.expander(f"{title} Mapping Debug", expanded=False):
        st.write("Available columns:", available_columns)
        st.write("Suggested mapping:", inspection_result.suggested_mapping)
        st.write("Resolved mapping:", resolved_mapping)

    if resolved_mapping.get("invoice_date") is None:
        st.warning(f"{title}: Invoice Date was not mapped. Downstream checks may mark records as DATE ERROR.")

    return resolved_mapping


st.title("GST Reconciliation Assistant")
st.warning(
    "This tool provides automated insights. Please review results before making financial or compliance decisions."
)
st.caption("ITC eligibility depends on multiple compliance factors.")

with st.expander("Status Legend", expanded=False):
    st.markdown(
        "\n".join(
            [
                "- `MATCHED` = matched on GSTIN and invoice controls within configured tolerances.",
                "- `AT RISK` = currently used only for `MISSING_IN_2B` and `VALUE_MISMATCH`.",
                "- `NOT ELIGIBLE` = reserved for cases where business rules explicitly require it.",
                "- `REVIEW REQUIRED` = manual review needed before any financial conclusion.",
                "- `AMBIGUOUS MATCH` = multiple plausible matches were found, so no auto-match was made.",
            ]
        )
    )

with st.sidebar:
    st.header("Inputs")
    purchase_file = st.file_uploader("Upload Purchase Register", type=["csv", "xlsx", "xls"])
    gstr2b_file = st.file_uploader("Upload GSTR-2B", type=["csv", "xlsx", "xls"])

    _warn_for_file_size(purchase_file)
    _warn_for_file_size(gstr2b_file)

    purchase_skip_rows = st.number_input(
        "Skip top rows (Purchase Register)",
        min_value=0,
        max_value=10,
        value=0,
        step=1,
        help="Use this if the file has extra title rows or merged header rows.",
    )
    gstr2b_skip_rows = st.number_input(
        "Skip top rows (GSTR-2B)",
        min_value=0,
        max_value=10,
        value=0,
        step=1,
        help="Use this if the file has extra title rows or merged header rows.",
    )

    st.header("Tolerance Controls")
    value_tolerance = st.slider("Value tolerance (Rs.)", min_value=1, max_value=20, value=2, step=1)
    date_tolerance = st.slider("Date tolerance (days)", min_value=0, max_value=30, value=5, step=1)
    fuzzy_threshold = st.slider("Fuzzy invoice similarity", min_value=0.85, max_value=0.99, value=0.92, step=0.01)

    st.header("Filters")
    mismatches_only = st.checkbox("Show only mismatches")
    only_itc_risk = st.checkbox("Show only ITC risk")

    run_reconciliation = st.button(
        "Run Reconciliation",
        type="primary",
        use_container_width=True,
        disabled=(purchase_file is None or gstr2b_file is None),
    )


if purchase_file is None or gstr2b_file is None:
    st.info("Upload both files to enable reconciliation.")
    st.stop()

try:
    with st.spinner("Reading uploaded files..."):
        purchase_raw_df = load_input_file(purchase_file, skiprows=int(purchase_skip_rows))
        gstr2b_raw_df = load_input_file(gstr2b_file, skiprows=int(gstr2b_skip_rows))
        purchase_inspection = inspect_input_dataframe(purchase_raw_df)
        gstr2b_inspection = inspect_input_dataframe(gstr2b_raw_df)
except InputMappingError as exc:
    st.error(str(exc))
    st.stop()
except Exception:
    st.error("Error processing file. Please check format.")
    st.stop()

mapping_left, mapping_right = st.columns(2)
with mapping_left:
    purchase_mapping = _render_mapping_controls("Purchase Register", purchase_inspection, "purchase_mapping")
with mapping_right:
    gstr2b_mapping = _render_mapping_controls("GSTR-2B", gstr2b_inspection, "gstr2b_mapping")

if not run_reconciliation:
    st.info("Review the detected mappings, adjust if needed, then click Run Reconciliation.")
    st.stop()

try:
    purchase_df = standardize_mapped_dataframe(purchase_inspection.dataframe, purchase_mapping)
    gstr2b_df = standardize_mapped_dataframe(gstr2b_inspection.dataframe, gstr2b_mapping)
except InputMappingError as exc:
    st.error(str(exc))
    st.stop()
except Exception:
    st.error("Error processing file. Please check format.")
    st.stop()

if len(purchase_df) < 5 or len(gstr2b_df) < 5:
    st.warning("Sample size too small for meaningful analysis.")

try:
    with st.spinner("Running reconciliation..."):
        reconciliation_df = reconcile_invoices(
            purchase_df,
            gstr2b_df,
            value_tolerance=value_tolerance,
            date_tolerance_days=date_tolerance,
            fuzzy_threshold=fuzzy_threshold,
        )
except MissingRequiredColumnsError as exc:
    st.error("Uploaded file is missing required columns.")
    with st.expander("Column details"):
        st.write({"missing_columns": exc.missing_columns, "uploaded_columns": exc.available_columns})
    st.stop()
except Exception:
    st.error("Error processing file. Please check format.")
    st.stop()

if reconciliation_df.empty:
    st.warning("No records were produced. Please verify the uploaded files.")
    st.stop()

for warning_message in reconciliation_df.attrs.get("warnings", []):
    st.warning(warning_message)

reconciliation_export_df = build_reconciliation_export(reconciliation_df)
client_report_df = build_client_report(reconciliation_df)
summary_metrics = build_summary_metrics(reconciliation_df)
risk_summary = build_itc_risk_summary(reconciliation_df)
vendor_summary_df = build_vendor_summary(reconciliation_df)
follow_up_df = build_follow_up_sheet(reconciliation_df)
processing_log = reconciliation_df.attrs.get("processing_log", {})

exception_mask = reconciliation_df["Status"] != "MATCHED"
if mismatches_only:
    reconciliation_view = reconciliation_export_df[exception_mask.values].copy()
else:
    reconciliation_view = reconciliation_export_df.copy()

if only_itc_risk:
    reconciliation_view = reconciliation_view[reconciliation_view["ITC Status"] == "AT RISK"].copy()

st.markdown(
    f"### ₹{summary_metrics['itc_at_risk_amount']:,.2f} ITC is at risk across "
    f"{risk_summary['num_risky_invoices']:,} invoices from {risk_summary['num_vendors']:,} vendors."
)
st.caption("⚠️ This is an automated estimate. Please review before making compliance decisions.")

metric_columns = st.columns(4)
metric_columns[0].metric("Total Invoices", f"{summary_metrics['total_invoices']:,}")
metric_columns[1].metric("Matched %", f"{summary_metrics['matched_percentage']:.2f}%")
metric_columns[2].metric("Potential ITC Risk", f"Rs. {summary_metrics['itc_at_risk_amount']:,.2f}")
metric_columns[3].metric("Missing Invoices", f"{summary_metrics['missing_invoices_count']:,}")

with st.expander("Processing Log", expanded=False):
    st.write(
        {
            "purchase_rows_processed": processing_log.get("purchase_rows_processed", 0),
            "gstr2b_rows_processed": processing_log.get("gstr2b_rows_processed", 0),
            "matches": processing_log.get("matches", 0),
            "exceptions": processing_log.get("exceptions", 0),
        }
    )

st.markdown("### Invoice Issues Breakdown")
download_columns = st.columns(2)
with download_columns[0]:
    st.download_button(
        "Download Reconciliation CSV",
        data=_to_csv_bytes(reconciliation_export_df),
        file_name="gst_reconciliation_output.csv",
        mime="text/csv",
    )
with download_columns[1]:
    st.download_button(
        "Download Client Report",
        data=_to_csv_bytes(client_report_df),
        file_name="gst_client_report.csv",
        mime="text/csv",
    )
_render_data_table(reconciliation_view, height=460)

dashboard_left, dashboard_right = st.columns([1.1, 1])

with dashboard_left:
    st.markdown("### Vendor Summary")
    st.download_button(
        "Download Vendor Summary CSV",
        data=_to_csv_bytes(vendor_summary_df),
        file_name="vendor_summary.csv",
        mime="text/csv",
    )
    st.dataframe(vendor_summary_df, use_container_width=True, height=320)

with dashboard_right:
    st.markdown("### Vendor Follow-up Sheet")
    st.download_button(
        "Download Follow-up CSV",
        data=_to_csv_bytes(follow_up_df),
        file_name="vendor_follow_up.csv",
        mime="text/csv",
    )
    st.dataframe(follow_up_df, use_container_width=True, height=320)

st.markdown("### Exception Snapshot")
exception_counts = (
    reconciliation_df.loc[reconciliation_df["Status"] != "MATCHED", "Status"]
    .value_counts()
    .rename_axis("Issue")
    .reset_index(name="Invoice Count")
)

if exception_counts.empty:
    st.success("No exceptions found in the processed dataset.")
else:
    st.bar_chart(exception_counts.set_index("Issue"))
