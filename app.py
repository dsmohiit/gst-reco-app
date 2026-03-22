from __future__ import annotations

import traceback
from io import StringIO

import pandas as pd
import streamlit as st

from gst_recon import (
    detect_file_header,
    InputMappingError,
    MissingRequiredColumnsError,
    build_client_report,
    build_follow_up_sheet,
    build_itc_risk_summary,
    build_reconciliation_export,
    build_summary_metrics,
    build_vendor_summary,
    generate_vendor_summary,
    inspect_input_dataframe,
    load_input_file,
    prepare_data,
    reconcile_invoices,
    standardize_mapped_dataframe,
)


FILE_SIZE_WARNING_BYTES = 50 * 1024 * 1024
MANUAL_SELECT_PLACEHOLDER = "-- Select column --"
DISPLAY_FIELD_LABELS = {
    "gstin": "GST Number",
    "invoice_number": "Invoice Number",
    "invoice_date": "Invoice Date",
    "taxable_value": "Amount",
}
REQUIRED_MAPPING_FIELDS = {"gstin", "invoice_number", "taxable_value"}


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


def _vendor_scorecard_highlight(row: pd.Series) -> list[str]:
    if float(row.get("Compliance Score", 0) or 0) < 80:
        style = "background-color: #fef3c7; color: #92400e;"
    else:
        style = ""
    return [style] * len(row)


def _warn_for_file_size(uploaded_file) -> None:
    if uploaded_file is not None and getattr(uploaded_file, "size", 0) > FILE_SIZE_WARNING_BYTES:
        st.warning(
            f"{uploaded_file.name} is larger than 50 MB. Processing may be slower, "
            "and fuzzy matching may be reduced for stability."
        )


def _build_file_token(uploaded_file, skip_rows: int) -> str:
    return f"{uploaded_file.name}:{getattr(uploaded_file, 'size', 0)}:{skip_rows}"


def _build_files_token(uploaded_files) -> str:
    return "|".join(
        f"{uploaded_file.name}:{getattr(uploaded_file, 'size', 0)}"
        for uploaded_file in uploaded_files
    )


def _make_unique_source_file_name(file_name: str, seen_names: dict[str, int], file_index: int) -> str:
    current_count = seen_names.get(file_name, 0) + 1
    seen_names[file_name] = current_count
    if current_count == 1:
        return file_name
    return f"{file_name} ({file_index})"


def _load_gstr2b_files(uploaded_files) -> tuple[pd.DataFrame, list[dict[str, object]], int]:
    loaded_frames: list[pd.DataFrame] = []
    file_summaries: list[dict[str, object]] = []
    seen_names: dict[str, int] = {}
    total_rows = 0

    for file_index, uploaded_file in enumerate(uploaded_files, start=1):
        uploaded_file.seek(0)
        raw_df = load_input_file(uploaded_file, skiprows=None)
        source_file_name = _make_unique_source_file_name(uploaded_file.name, seen_names, file_index)
        raw_df = raw_df.copy()
        raw_df["source_file"] = source_file_name
        loaded_frames.append(raw_df)
        total_rows += len(raw_df)
        file_summaries.append(
            {
                "display_name": source_file_name,
                "rows": int(len(raw_df)),
                "header_row": int(raw_df.attrs.get("detected_header_row", 0)),
                "auto_detected": bool(raw_df.attrs.get("header_auto_detected", False)),
            }
        )

    if not loaded_frames:
        raise InputMappingError("No GSTR-2B files could be loaded.")

    combined_df = pd.concat(loaded_frames, axis=0, ignore_index=True, sort=False)
    if combined_df.empty:
        raise InputMappingError("Combined GSTR-2B data is empty. Please check the uploaded files.")

    return combined_df, file_summaries, total_rows


def _initialize_header_detection_state(dataset_key: str, uploaded_file) -> None:
    st.session_state.setdefault("header_detection_tokens", {})
    st.session_state.setdefault("header_detection_results", {})

    if uploaded_file is None:
        return

    file_token = f"{uploaded_file.name}:{getattr(uploaded_file, 'size', 0)}"
    existing_token = st.session_state["header_detection_tokens"].get(dataset_key)
    if existing_token == file_token:
        return

    detected_row, detected_score = detect_file_header(uploaded_file)
    st.session_state["header_detection_tokens"][dataset_key] = file_token
    st.session_state["header_detection_results"][dataset_key] = {
        "row": int(detected_row),
        "score": int(detected_score),
    }
    st.session_state[f"{dataset_key}_skip_rows"] = int(detected_row if detected_score > 0 else 0)


def _initialize_mapping_state(dataset_key: str, inspection_result, file_token: str) -> None:
    st.session_state.setdefault("column_mapping", {})
    st.session_state.setdefault("mapping_tokens", {})
    st.session_state.setdefault("show_mapping_ui", False)

    existing_token = st.session_state["mapping_tokens"].get(dataset_key)
    if existing_token != file_token:
        st.session_state["column_mapping"][dataset_key] = inspection_result.suggested_mapping.copy()
        st.session_state["mapping_tokens"][dataset_key] = file_token
        st.session_state["show_mapping_ui"] = False


def _get_mapping_state(dataset_key: str) -> dict[str, str | None]:
    return st.session_state["column_mapping"].get(dataset_key, {}).copy()


def _mapping_requires_attention(inspection_result, mapping: dict[str, str | None]) -> bool:
    for field in DISPLAY_FIELD_LABELS:
        if inspection_result.confidence.get(field) == "LOW":
            return True
        if mapping.get(field) is None:
            return True
    return False


def _render_mapping_status(purchase_inspection, purchase_mapping, gstr2b_inspection, gstr2b_mapping) -> None:
    purchase_needs_review = _mapping_requires_attention(purchase_inspection, purchase_mapping)
    gstr2b_needs_review = _mapping_requires_attention(gstr2b_inspection, gstr2b_mapping)
    any_medium = any(
        confidence == "MEDIUM"
        for confidence in list(purchase_inspection.confidence.values()) + list(gstr2b_inspection.confidence.values())
    )

    if purchase_needs_review or gstr2b_needs_review:
        st.warning("Some columns could not be detected. Please review mapping.")
        st.session_state["show_mapping_ui"] = True
    elif any_medium:
        st.info("Columns detected. Please review if needed.")
    else:
        st.success("Columns detected successfully")


def _render_mapping_editor(title: str, dataset_key: str, inspection_result) -> dict[str, str | None]:
    mapping = _get_mapping_state(dataset_key)
    available_columns = [
        column for column in inspection_result.dataframe.columns.tolist()
        if not str(column).startswith("_") and str(column) != "source_file"
    ]
    st.markdown(f"#### {title}")

    for field, label in DISPLAY_FIELD_LABELS.items():
        options = [MANUAL_SELECT_PLACEHOLDER] + available_columns
        current_value = mapping.get(field)
        default_index = options.index(current_value) if current_value in options else 0

        help_text = None
        if inspection_result.confidence.get(field) == "LOW":
            help_text = "Multiple possible columns were found. Please choose the correct one."
        elif inspection_result.confidence.get(field) == "MEDIUM":
            help_text = "A likely match was found. Please confirm if needed."
        elif inspection_result.confidence.get(field) == "MISSING":
            help_text = "No column was detected for this field. Select it manually if available."

        selected_value = st.selectbox(
            f"{title}: {label}",
            options,
            index=default_index,
            key=f"mapping_editor_{dataset_key}_{field}",
            help=help_text,
        )
        mapping[field] = None if selected_value == MANUAL_SELECT_PLACEHOLDER else selected_value

    st.session_state["column_mapping"][dataset_key] = mapping
    return mapping


def _render_mapping_debug(dataset_title: str, inspection_result, mapping: dict[str, str | None]) -> None:
    with st.expander(f"Advanced Debug: {dataset_title}", expanded=False):
        st.write("Resolved mapping:", mapping)
        st.write("Confidence:", inspection_result.confidence)
        st.write("Available columns:", inspection_result.dataframe.columns.tolist())


def _render_preview(title: str, standardized_df: pd.DataFrame) -> None:
    preview_columns = [column for column in ["GSTIN", "Invoice Number", "Invoice Date", "Purchase Value"] if column in standardized_df.columns]
    st.markdown(f"#### {title} Preview")
    st.dataframe(standardized_df[preview_columns].head(), use_container_width=True, height=220)


def _process_mapped_dataframe(df, mapping):
    standardized_df = prepare_data(df, mapping)
    if standardized_df.empty:
        raise InputMappingError("The mapped columns resulted in an empty dataset. Please check your mapping.")
    standardized_df["Purchase Value"] = pd.to_numeric(standardized_df["Purchase Value"], errors="coerce").fillna(0)
    standardized_df["Invoice Number"] = standardized_df["Invoice Number"].fillna("").astype(str)
    standardized_df["GSTIN"] = standardized_df["GSTIN"].fillna("").astype(str)
    if "source_file" in standardized_df.columns:
        standardized_df["source_file"] = standardized_df["source_file"].fillna("").astype(str)
    return standardized_df


def _prepare_gstr2b_dataframe(df: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    standardized_df = _process_mapped_dataframe(df, mapping)
    dedupe_subset = ["GSTIN", "Invoice Number", "Purchase Value"]
    standardized_df = standardized_df.drop_duplicates(subset=dedupe_subset, keep="first").reset_index(drop=True)
    if standardized_df.empty:
        raise InputMappingError("Combined GSTR-2B data is empty after deduplication. Please check the uploaded files.")
    return standardized_df


def _run_reconciliation(purchase_df: pd.DataFrame, gstr2b_df: pd.DataFrame, value_tolerance: int, date_tolerance: int, fuzzy_threshold: float):
    return reconcile_invoices(
        purchase_df,
        gstr2b_df,
        value_tolerance=value_tolerance,
        date_tolerance_days=date_tolerance,
        fuzzy_threshold=fuzzy_threshold,
    )


def _show_processing_message(message: str) -> None:
    if message == "The mapped columns resulted in an empty dataset. Please check your mapping.":
        st.warning(message)
    else:
        st.error(f"Error: {message}")


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
    gstr2b_files = st.file_uploader(
        "Upload GSTR-2B Files",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
    )

    _warn_for_file_size(purchase_file)
    for gstr2b_file in gstr2b_files:
        _warn_for_file_size(gstr2b_file)

    try:
        _initialize_header_detection_state("purchase", purchase_file)
    except InputMappingError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"Error: {str(exc)}")
        st.stop()

    purchase_skip_rows = st.number_input(
        "Skip top rows (Purchase Register)",
        min_value=0,
        max_value=25,
        step=1,
        key="purchase_skip_rows",
        help="Use this if the file has extra title rows or merged header rows.",
    )
    st.caption("GSTR-2B header rows are auto-detected separately for each uploaded file.")

    st.header("Tolerance Controls")
    value_tolerance = st.slider("Value tolerance (Rs.)", min_value=1, max_value=20, value=2, step=1)
    date_tolerance = st.slider("Date tolerance (days)", min_value=0, max_value=30, value=5, step=1)
    fuzzy_threshold = st.slider("Fuzzy invoice similarity", min_value=0.85, max_value=0.99, value=0.92, step=0.01)

    st.header("Filters")
    mismatches_only = st.checkbox("Show only mismatches")
    only_itc_risk = st.checkbox("Show only ITC risk")
    only_timing_differences = st.checkbox("Show only Timing Differences")

    run_reconciliation = st.button(
        "Run Reconciliation",
        type="primary",
        use_container_width=True,
        disabled=(purchase_file is None or not gstr2b_files),
    )


if purchase_file is None or not gstr2b_files:
    st.info("Upload both files to enable reconciliation.")
    st.stop()

purchase_header_info = st.session_state.get("header_detection_results", {}).get("purchase", {"row": 0, "score": 0})

if purchase_header_info.get("score", 0) == 0:
    st.warning("Could not auto-detect header for Purchase Register. Please select manually.")

try:
    with st.spinner("Reading uploaded files..."):
        purchase_skip_param = (
            None
            if purchase_header_info.get("score", 0) > 0 and int(purchase_skip_rows) == int(purchase_header_info["row"])
            else int(purchase_skip_rows)
        )

        purchase_raw_df = load_input_file(purchase_file, skiprows=purchase_skip_param)
        gstr2b_raw_df, gstr2b_file_summaries, gstr2b_total_rows = _load_gstr2b_files(gstr2b_files)
        if purchase_raw_df.attrs.get("header_auto_detected"):
            st.toast(f"🎯 Auto-detected header at row {purchase_raw_df.attrs.get('detected_header_row', 0)}", icon="✅")
        for file_summary in gstr2b_file_summaries:
            if file_summary["auto_detected"]:
                st.toast(
                    f"🎯 Auto-detected header at row {file_summary['header_row']} for {file_summary['display_name']}",
                    icon="✅",
                )
        st.toast(f"✅ {len(gstr2b_files)} GSTR-2B files merged successfully.", icon="✅")
        purchase_inspection = inspect_input_dataframe(purchase_raw_df)
        gstr2b_inspection = inspect_input_dataframe(gstr2b_raw_df)
except InputMappingError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:
    st.error(f"Error: {str(exc)}")
    st.stop()

with st.sidebar:
    st.success(f"✅ {len(gstr2b_file_summaries)} files loaded (Total {gstr2b_total_rows:,} rows).")
    for file_summary in gstr2b_file_summaries:
        st.caption(f"{file_summary['display_name']} - {file_summary['rows']:,} rows")
    if gstr2b_total_rows > 50000:
        st.warning("Large dataset detected. Processing may take a moment.")

mapping_left, mapping_right = st.columns(2)
purchase_token = _build_file_token(purchase_file, int(purchase_skip_rows))
gstr2b_token = _build_files_token(gstr2b_files)
_initialize_mapping_state("purchase", purchase_inspection, purchase_token)
_initialize_mapping_state("gstr2b", gstr2b_inspection, gstr2b_token)
purchase_mapping = _get_mapping_state("purchase")
gstr2b_mapping = _get_mapping_state("gstr2b")

_render_mapping_status(purchase_inspection, purchase_mapping, gstr2b_inspection, gstr2b_mapping)

if st.button("Review / Edit Mapping"):
    st.session_state["show_mapping_ui"] = True

if st.session_state.get("show_mapping_ui", False):
    with mapping_left:
        purchase_mapping = _render_mapping_editor("Purchase Register Mapping", "purchase", purchase_inspection)
    with mapping_right:
        gstr2b_mapping = _render_mapping_editor("GSTR-2B Mapping", "gstr2b", gstr2b_inspection)

_render_mapping_debug("Purchase Register", purchase_inspection, purchase_mapping)
_render_mapping_debug("GSTR-2B", gstr2b_inspection, gstr2b_mapping)

if not run_reconciliation:
    preview_error = None
    preview_purchase_df = None
    preview_gstr2b_df = None
    try:
        preview_purchase_df = _process_mapped_dataframe(purchase_inspection.dataframe, purchase_mapping)
        preview_gstr2b_df = _prepare_gstr2b_dataframe(gstr2b_inspection.dataframe, gstr2b_mapping)
    except InputMappingError as exc:
        preview_error = str(exc)
    except Exception:
        preview_error = "Error processing file. Please check format."

    if preview_error:
        _show_processing_message(preview_error)
    else:
        preview_left, preview_right = st.columns(2)
        with preview_left:
            _render_preview("Purchase Register", preview_purchase_df)
        with preview_right:
            _render_preview("GSTR-2B", preview_gstr2b_df)
        if len(preview_purchase_df) < 5 or len(preview_gstr2b_df) < 5:
            st.warning("Sample size too small for meaningful analysis.")

    st.info("Click Run Reconciliation after reviewing the detected columns.")
    st.stop()

try:
    purchase_df = _process_mapped_dataframe(purchase_inspection.dataframe, purchase_mapping)
    gstr2b_df = _prepare_gstr2b_dataframe(gstr2b_inspection.dataframe, gstr2b_mapping)
except InputMappingError as exc:
    _show_processing_message(str(exc))
    st.stop()
except Exception as exc:
    st.error(f"Error: {str(exc)}")
    st.stop()

st.write(f"Rows in Purchase: {len(purchase_df)}, Rows in 2B: {len(gstr2b_df)}")
if len(purchase_df) == 0 or len(gstr2b_df) == 0:
    st.warning("The mapped columns resulted in an empty dataset. Please check your Skip Rows setting or Sheet Selection.")
    st.stop()

if len(purchase_df) < 5 or len(gstr2b_df) < 5:
    st.warning("Sample size too small for meaningful analysis.")

try:
    with st.spinner("Running reconciliation..."):
        reconciliation_df = _run_reconciliation(
            purchase_df,
            gstr2b_df,
            value_tolerance,
            date_tolerance,
            fuzzy_threshold,
        )
except MissingRequiredColumnsError as exc:
    st.error("Uploaded file is missing required columns.")
    with st.expander("Column details"):
        st.write({"missing_columns": exc.missing_columns, "uploaded_columns": exc.available_columns})
    st.stop()
except Exception as exc:
    st.error(traceback.format_exc())
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
vendor_scorecard_df = generate_vendor_summary(reconciliation_df)
follow_up_df = build_follow_up_sheet(reconciliation_df)
processing_log = reconciliation_df.attrs.get("processing_log", {})

exception_mask = reconciliation_df["Status"] != "MATCHED"
if only_timing_differences:
    reconciliation_view = reconciliation_export_df[
        reconciliation_export_df["Reason"] == "Matched (Timing Difference)"
    ].copy()
elif mismatches_only:
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

st.subheader("📊 Vendor Risk Scorecard")
high_risk_vendors = vendor_scorecard_df[vendor_scorecard_df["Compliance Score"] < 80].copy()
if not high_risk_vendors.empty:
    st.warning("Vendors with compliance below 80% are marked as High Risk and should be reviewed first.")
st.dataframe(
    vendor_scorecard_df.style.apply(_vendor_scorecard_highlight, axis=1),
    use_container_width=True,
    height=320,
)

metric_columns = st.columns(5)
metric_columns[0].metric("Total Invoices", f"{summary_metrics['total_invoices']:,}")
metric_columns[1].metric("Matched %", f"{summary_metrics['matched_percentage']:.2f}%")
metric_columns[2].metric("Potential ITC Risk", f"Rs. {summary_metrics['itc_at_risk_amount']:,.2f}")
metric_columns[3].metric("Timing Differences", f"{summary_metrics['timing_differences_count']:,}")
metric_columns[4].metric("Missing Invoices", f"{summary_metrics['missing_invoices_count']:,}")

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
