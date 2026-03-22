from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from io import BytesIO
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd


class MissingRequiredColumnsError(ValueError):
    def __init__(self, missing_columns: Sequence[str], available_columns: Sequence[str]) -> None:
        super().__init__("Uploaded file is missing required columns.")
        self.missing_columns = list(missing_columns)
        self.available_columns = [str(column) for column in available_columns]


REQUIRED_CANONICAL_COLUMNS = {
    "gstin": "GSTIN",
    "invoice_number": "Invoice Number",
    "invoice_date": "Invoice Date",
    "taxable_value": "Taxable Value",
}

OPTIONAL_CANONICAL_COLUMNS = {
    "supplier_name": "Supplier Name",
    "igst": "IGST",
    "cgst": "CGST",
    "sgst": "SGST",
}

COLUMN_ALIASES = {
    "gstin": "gstin",
    "gstno": "gstin",
    "gst_no": "gstin",
    "suppliergstin": "gstin",
    "vendorgstin": "gstin",
    "suppliername": "supplier_name",
    "supplier_name": "supplier_name",
    "legalname": "supplier_name",
    "tradename": "supplier_name",
    "vendorname": "supplier_name",
    "vendor_name": "supplier_name",
    "invoicenumber": "invoice_number",
    "invno": "invoice_number",
    "inv_no": "invoice_number",
    "invoiceno": "invoice_number",
    "invoiceno.": "invoice_number",
    "invoicenum": "invoice_number",
    "invoice_no": "invoice_number",
    "invoice_no.": "invoice_number",
    "invoice_num": "invoice_number",
    "billno": "invoice_number",
    "billnumber": "invoice_number",
    "invoicedate": "invoice_date",
    "invdate": "invoice_date",
    "invoice_dt": "invoice_date",
    "billdate": "invoice_date",
    "taxablevalue": "taxable_value",
    "purchasevalue": "taxable_value",
    "purchase_value": "taxable_value",
    "purchaseamount": "taxable_value",
    "taxable_amount": "taxable_value",
    "taxableamt": "taxable_value",
    "invoicevalue": "taxable_value",
    "assessablevalue": "taxable_value",
    "igst": "igst",
    "cgst": "cgst",
    "sgst": "sgst",
}

STATUS_ORDER = [
    "INVALID DATA",
    "INVALID GSTIN",
    "DATE ERROR",
    "REVIEW REQUIRED",
    "AMBIGUOUS MATCH",
    "VALUE_MISMATCH",
    "DATE_MISMATCH",
    "MISSING_IN_2B",
    "ONLY_IN_2B",
    "MATCHED",
]

EXPORT_COLUMNS = [
    "GSTIN",
    "Supplier Name",
    "Invoice Number",
    "Purchase Date",
    "2B Date",
    "Status",
    "ITC Status",
    "Purchase Value",
    "2B Value",
    "Difference",
    "Reason",
    "Explanation",
    "Recommended Action",
    "Duplicate Flag",
    "Match Stage",
]

CLIENT_REPORT_COLUMNS = [
    "GSTIN",
    "Invoice Number",
    "Issue",
    "ITC Status",
    "Amount at Risk",
    "Recommended Action",
]


@dataclass(frozen=True)
class ReconciliationConfig:
    value_tolerance: float = 2.0
    date_tolerance_days: int = 5
    fuzzy_threshold: float = 0.92
    large_dataset_row_threshold: int = 100000


def load_invoice_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    if name.endswith(".csv"):
        return pd.read_csv(BytesIO(file_bytes))

    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(BytesIO(file_bytes))

    raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")


def _normalize_column_name(column_name: str) -> str:
    cleaned = "".join(char for char in str(column_name).strip().lower() if char.isalnum() or char == "_")
    return COLUMN_ALIASES.get(cleaned, cleaned)


def _clean_text_series(series: pd.Series) -> pd.Series:
    return series.where(series.notna(), "").astype(str).str.strip()


def _normalize_gstin(series: pd.Series) -> pd.Series:
    return _clean_text_series(series).str.upper()


def _normalize_invoice_number(series: pd.Series) -> pd.Series:
    return (
        _clean_text_series(series)
        .str.lower()
        .str.replace(r"[/\-\s]+", "", regex=True)
        .str.replace(r"[^a-z0-9]", "", regex=True)
        .str.lstrip("0")
    )


def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {column: _normalize_column_name(column) for column in df.columns}
    standardized = df.rename(columns=rename_map).copy()

    missing = [display for key, display in REQUIRED_CANONICAL_COLUMNS.items() if key not in standardized.columns]
    if missing:
        raise MissingRequiredColumnsError(missing, df.columns)

    for key in OPTIONAL_CANONICAL_COLUMNS:
        if key not in standardized.columns:
            standardized[key] = 0 if key in {"igst", "cgst", "sgst"} else ""

    return standardized


def clean_invoice_data(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    standardized = _ensure_required_columns(df)
    cleaned = standardized.copy()

    for column in cleaned.columns:
        if pd.api.types.is_object_dtype(cleaned[column]) or pd.api.types.is_string_dtype(cleaned[column]):
            cleaned[column] = _clean_text_series(cleaned[column])

    for numeric_column in ["taxable_value", "igst", "cgst", "sgst"]:
        cleaned[numeric_column] = pd.to_numeric(cleaned[numeric_column], errors="coerce").fillna(0.0)

    raw_invoice_date = cleaned["invoice_date"].copy()
    cleaned["invoice_date"] = pd.to_datetime(cleaned["invoice_date"], errors="coerce", dayfirst=True)
    cleaned["gstin"] = _normalize_gstin(cleaned["gstin"])
    cleaned["supplier_name"] = _clean_text_series(cleaned["supplier_name"])
    cleaned["invoice_number"] = _clean_text_series(cleaned["invoice_number"])
    cleaned["invoice_number_normalized"] = _normalize_invoice_number(cleaned["invoice_number"])
    cleaned["source"] = source_name
    cleaned["source_row_id"] = np.arange(len(cleaned), dtype=int)
    cleaned["key"] = cleaned["gstin"] + "|" + cleaned["invoice_number_normalized"]

    cleaned["invoice_missing_flag"] = cleaned["invoice_number_normalized"].eq("")
    cleaned["invalid_gstin_flag"] = cleaned["gstin"].str.len().ne(15)
    cleaned["date_error_flag"] = cleaned["invoice_date"].isna()
    cleaned["tax_total"] = cleaned[["igst", "cgst", "sgst"]].sum(axis=1)

    has_key = cleaned["gstin"].ne("") & cleaned["invoice_number_normalized"].ne("")
    cleaned["duplicate_group_size"] = np.where(has_key, cleaned.groupby("key")["key"].transform("size"), 0)
    cleaned["duplicate_flag"] = cleaned["duplicate_group_size"].gt(1)

    cleaned["precheck_status"] = ""
    cleaned["precheck_reason"] = ""

    cleaned.loc[cleaned["invoice_missing_flag"], ["precheck_status", "precheck_reason"]] = [
        "INVALID DATA",
        "Missing invoice number",
    ]
    cleaned.loc[
        cleaned["precheck_status"].eq("") & cleaned["invalid_gstin_flag"],
        ["precheck_status", "precheck_reason"],
    ] = ["INVALID GSTIN", "Invalid GSTIN format"]
    cleaned.loc[
        cleaned["precheck_status"].eq("") & cleaned["date_error_flag"],
        ["precheck_status", "precheck_reason"],
    ] = ["DATE ERROR", "Invoice date could not be parsed"]
    cleaned.loc[
        cleaned["precheck_status"].eq("") & cleaned["duplicate_flag"],
        ["precheck_status", "precheck_reason"],
    ] = ["REVIEW REQUIRED", "Duplicate invoice"]

    cleaned["eligible_for_matching"] = cleaned["precheck_status"].eq("")
    return cleaned


def _date_diff_days(left_dates: pd.Series, right_dates: pd.Series) -> pd.Series:
    return (left_dates - right_dates).abs().dt.days.fillna(np.inf)


def _string_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left, right).ratio()


def _status_rank(status: str) -> int:
    try:
        return STATUS_ORDER.index(status)
    except ValueError:
        return len(STATUS_ORDER)


def assign_itc_status(row: pd.Series) -> str:
    if row["Status"] == "MATCHED":
        return "ELIGIBLE"
    elif row["Status"] in ["MISSING_IN_2B", "VALUE_MISMATCH"]:
        return "AT RISK"
    elif row["Status"] in ["DATE ERROR", "INVALID DATA", "AMBIGUOUS MATCH"]:
        return "REVIEW REQUIRED"
    else:
        return "REVIEW REQUIRED"


def generate_explanation(row: pd.Series) -> str:
    reason = str(row.get("Reason", "")).strip()
    status = str(row.get("Status", "")).strip()
    purchase_value = float(row.get("Purchase Value", 0.0) or 0.0)
    gstr2b_value = float(row.get("2B Value", 0.0) or 0.0)

    if reason == "Supplier not filed":
        return "Invoice is missing in GSTR-2B. Supplier has likely not filed or uploaded this invoice."
    if reason == "Value mismatch":
        return f"Purchase value ({purchase_value:,.2f}) does not match GSTR-2B value ({gstr2b_value:,.2f})."
    if reason == "Present only in GSTR-2B":
        return "Invoice exists in GSTR-2B but not in purchase register."
    if reason == "GSTIN mismatch" or status == "GSTIN_MISMATCH":
        return "Invoice number matches but GSTIN is different between records."
    if reason == "Duplicate invoice":
        return "Duplicate invoice detected in dataset."
    if status == "INVALID DATA":
        return "Invoice has missing or incorrect required fields."
    if status == "INVALID GSTIN":
        return "GSTIN appears invalid and should be corrected before relying on this record."
    if status == "DATE ERROR":
        return "Invoice date could not be parsed safely from the uploaded data."
    if status == "AMBIGUOUS MATCH":
        return "More than one possible match was found, so this invoice needs manual review."
    if status == "REVIEW REQUIRED":
        return "This invoice needs manual review before any compliance action is taken."
    if reason == "Invoice uploaded in different period":
        return "Invoice dates differ beyond the configured tolerance and may belong to a different filing period."
    return "No issue detected"


def generate_action(row: pd.Series) -> str:
    reason = str(row.get("Reason", "")).strip()
    status = str(row.get("Status", "")).strip()

    if reason == "Supplier not filed":
        return "Follow up with vendor to file GSTR-1."
    if reason == "Value mismatch":
        return "Verify invoice values in books and vendor filing."
    if reason == "Present only in GSTR-2B":
        return "Check for missing entry in purchase register."
    if reason == "GSTIN mismatch" or status == "GSTIN_MISMATCH":
        return "Verify GSTIN correctness in invoice."
    if reason == "Duplicate invoice":
        return "Remove or validate duplicate entries."
    if status == "INVALID DATA":
        return "Review and correct missing or invalid fields."
    if status == "INVALID GSTIN":
        return "Validate the GSTIN and correct the source record before relying on it."
    if status == "DATE ERROR":
        return "Review the invoice date format and correct the source file."
    if status == "AMBIGUOUS MATCH":
        return "Review all possible matches manually before concluding."
    if status == "REVIEW REQUIRED":
        return "Review this invoice manually before taking any action."
    if reason == "Invoice uploaded in different period":
        return "Review return periods and confirm whether the invoice was uploaded in a different month."
    return "No action needed"


def _is_exception(df: pd.DataFrame) -> pd.Series:
    return df["Status"].ne("MATCHED") | df["Duplicate Flag"].fillna(False)


def _select_best_matches(
    candidates: pd.DataFrame,
    left_id_column: str,
    right_id_column: str,
    priority_columns: Iterable[str],
    ascending: Sequence[bool] | None = None,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    sort_columns = list(priority_columns)
    sort_ascending = list(ascending) if ascending is not None else [True] * len(sort_columns)
    ordered = candidates.sort_values(sort_columns, ascending=sort_ascending).copy()
    ordered = ordered.drop_duplicates(subset=[left_id_column], keep="first")
    ordered = ordered.drop_duplicates(subset=[right_id_column], keep="first")
    return ordered


def _extract_ambiguous_ids(
    candidates: pd.DataFrame,
    left_id_column: str,
    right_id_column: str,
) -> Tuple[Set[int], Set[int]]:
    if candidates.empty:
        return set(), set()

    left_ambiguous = {
        int(index)
        for index, count in candidates[left_id_column].value_counts().items()
        if int(count) > 1
    }
    right_ambiguous = {
        int(index)
        for index, count in candidates[right_id_column].value_counts().items()
        if int(count) > 1
    }
    return left_ambiguous, right_ambiguous


def _build_exact_candidates(
    purchase_df: pd.DataFrame,
    gstr2b_df: pd.DataFrame,
    config: ReconciliationConfig,
) -> pd.DataFrame:
    candidates = purchase_df.merge(gstr2b_df, on="key", suffixes=("_pr", "_2b"))
    if candidates.empty:
        return candidates

    candidates["value_diff_abs"] = (candidates["taxable_value_pr"] - candidates["taxable_value_2b"]).abs()
    candidates["date_diff_abs"] = _date_diff_days(candidates["invoice_date_pr"], candidates["invoice_date_2b"])
    exact = candidates[
        (candidates["value_diff_abs"] <= config.value_tolerance)
        & (candidates["date_diff_abs"] <= config.date_tolerance_days)
    ].copy()
    exact["match_stage"] = "EXACT"
    return exact


def _build_key_candidates(purchase_df: pd.DataFrame, gstr2b_df: pd.DataFrame) -> pd.DataFrame:
    candidates = purchase_df.merge(gstr2b_df, on="key", suffixes=("_pr", "_2b"))
    if candidates.empty:
        return candidates

    candidates["value_diff_abs"] = (candidates["taxable_value_pr"] - candidates["taxable_value_2b"]).abs()
    candidates["date_diff_abs"] = _date_diff_days(candidates["invoice_date_pr"], candidates["invoice_date_2b"])
    candidates["match_stage"] = "KEY"
    return candidates


def _build_value_buckets(df: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    safe_tolerance = max(float(tolerance), 1.0)
    bucket = (df["taxable_value"] / safe_tolerance).round().astype("Int64")

    blocked = df.copy()
    blocked["candidate_bucket"] = pd.DataFrame(
        {
            "minus_one": bucket - 1,
            "center": bucket,
            "plus_one": bucket + 1,
        }
    ).values.tolist()
    return blocked.explode("candidate_bucket")


def _build_fuzzy_candidates(
    purchase_df: pd.DataFrame,
    gstr2b_df: pd.DataFrame,
    config: ReconciliationConfig,
) -> pd.DataFrame:
    if purchase_df.empty or gstr2b_df.empty:
        return pd.DataFrame()

    purchase_blocked = _build_value_buckets(purchase_df, config.value_tolerance)
    gstr2b_blocked = gstr2b_df.copy()
    safe_tolerance = max(float(config.value_tolerance), 1.0)
    gstr2b_blocked["candidate_bucket"] = (gstr2b_blocked["taxable_value"] / safe_tolerance).round().astype("Int64")

    candidates = purchase_blocked.merge(
        gstr2b_blocked,
        on=["gstin", "candidate_bucket"],
        suffixes=("_pr", "_2b"),
    )
    if candidates.empty:
        return candidates

    candidates["value_diff_abs"] = (candidates["taxable_value_pr"] - candidates["taxable_value_2b"]).abs()
    candidates["date_diff_abs"] = _date_diff_days(candidates["invoice_date_pr"], candidates["invoice_date_2b"])
    candidates = candidates[
        (candidates["value_diff_abs"] <= config.value_tolerance)
        & (candidates["date_diff_abs"] <= config.date_tolerance_days)
    ].copy()
    if candidates.empty:
        return candidates

    candidates["invoice_similarity"] = candidates.apply(
        lambda row: _string_similarity(
            row["invoice_number_normalized_pr"],
            row["invoice_number_normalized_2b"],
        ),
        axis=1,
    )
    candidates = candidates[candidates["invoice_similarity"] >= config.fuzzy_threshold].copy()
    if candidates.empty:
        return candidates

    candidates["match_stage"] = "FUZZY"
    return candidates


def _classify_match(row: pd.Series, config: ReconciliationConfig) -> Tuple[str, str]:
    if row["value_diff_abs"] > config.value_tolerance:
        return "VALUE_MISMATCH", "Value mismatch"

    if row["date_diff_abs"] > config.date_tolerance_days:
        return "DATE_MISMATCH", "Invoice uploaded in different period"

    return "MATCHED", "Matched within configured tolerances"


def _build_source_output_row(
    row: pd.Series,
    status: str,
    reason: str,
    match_stage: str,
) -> Dict[str, object]:
    is_purchase = row["source"] == "Purchase Register"
    purchase_value = float(row["taxable_value"]) if is_purchase else 0.0
    gstr2b_value = float(row["taxable_value"]) if not is_purchase else 0.0

    return {
        "GSTIN": row["gstin"],
        "Purchase GSTIN": row["gstin"] if is_purchase else "",
        "2B GSTIN": row["gstin"] if not is_purchase else "",
        "Supplier Name": row["supplier_name"],
        "Invoice Number": row["invoice_number"],
        "Purchase Invoice Number": row["invoice_number"] if is_purchase else "",
        "2B Invoice Number": row["invoice_number"] if not is_purchase else "",
        "Purchase Date": row["invoice_date"] if is_purchase else pd.NaT,
        "2B Date": row["invoice_date"] if not is_purchase else pd.NaT,
        "Status": status,
        "ITC Status": "",
        "Purchase Value": purchase_value,
        "2B Value": gstr2b_value,
        "Difference": purchase_value - gstr2b_value,
        "Reason": reason,
        "Explanation": "",
        "Recommended Action": "",
        "Duplicate Flag": bool(row.get("duplicate_flag", False)),
        "Match Stage": match_stage,
        "Source": row["source"],
    }


def _build_match_output_row(row: pd.Series, config: ReconciliationConfig) -> Dict[str, object]:
    status, reason = _classify_match(row, config)
    purchase_value = float(row["taxable_value_pr"])
    gstr2b_value = float(row["taxable_value_2b"])

    return {
        "GSTIN": row["gstin_pr"],
        "Purchase GSTIN": row["gstin_pr"],
        "2B GSTIN": row["gstin_2b"],
        "Supplier Name": row["supplier_name_pr"] or row["supplier_name_2b"],
        "Invoice Number": row["invoice_number_pr"] or row["invoice_number_2b"],
        "Purchase Invoice Number": row["invoice_number_pr"],
        "2B Invoice Number": row["invoice_number_2b"],
        "Purchase Date": row["invoice_date_pr"],
        "2B Date": row["invoice_date_2b"],
        "Status": status,
        "ITC Status": "",
        "Purchase Value": purchase_value,
        "2B Value": gstr2b_value,
        "Difference": purchase_value - gstr2b_value,
        "Reason": reason,
        "Explanation": "",
        "Recommended Action": "",
        "Duplicate Flag": False,
        "Match Stage": row["match_stage"],
        "Source": "Reconciled",
    }


def _append_flagged_rows(
    output_rows: List[Dict[str, object]],
    df: pd.DataFrame,
    row_ids: Set[int],
    status: str,
    reason: str,
    match_stage: str,
) -> None:
    if not row_ids:
        return

    flagged_rows = df[df["source_row_id"].isin(row_ids)]
    for _, row in flagged_rows.iterrows():
        output_rows.append(_build_source_output_row(row, status, reason, match_stage))


def build_reconciliation_export(reconciliation_df: pd.DataFrame) -> pd.DataFrame:
    if reconciliation_df.empty:
        return pd.DataFrame(columns=EXPORT_COLUMNS)
    return reconciliation_df[EXPORT_COLUMNS].copy()


def build_itc_risk_view(reconciliation_df: pd.DataFrame) -> pd.DataFrame:
    if reconciliation_df.empty:
        return pd.DataFrame(columns=list(reconciliation_df.columns) + ["Risk Amount"])

    df = reconciliation_df.copy()
    df["Difference"] = pd.to_numeric(df["Difference"], errors="coerce").fillna(0)
    df["Purchase Value"] = pd.to_numeric(df["Purchase Value"], errors="coerce")

    df = df[df["Purchase Value"].notna()].copy()
    df_risk = df[df["Status"].isin(["MISSING_IN_2B", "VALUE_MISMATCH"])].copy()
    if df_risk.empty:
        df_risk["Risk Amount"] = pd.Series(dtype="float64")
        return df_risk

    df_risk["Risk Amount"] = 0.0
    df_risk.loc[df_risk["Status"] == "MISSING_IN_2B", "Risk Amount"] = df_risk["Purchase Value"]
    df_risk.loc[df_risk["Status"] == "VALUE_MISMATCH", "Risk Amount"] = df_risk["Difference"].clip(lower=0)
    df_risk = df_risk[df_risk["Risk Amount"] > 0].copy()

    if "Duplicate Flag" in df_risk.columns:
        df_risk = df_risk[df_risk["Duplicate Flag"].fillna(False) == False].copy()

    df_risk["Risk Amount"] = pd.to_numeric(df_risk["Risk Amount"], errors="coerce").fillna(0).round(2)
    return df_risk.reset_index(drop=True)


def build_itc_risk_summary(reconciliation_df: pd.DataFrame) -> Dict[str, float]:
    df_risk = build_itc_risk_view(reconciliation_df)
    if df_risk.empty:
        return {
            "itc_at_risk_amount": 0.0,
            "num_risky_invoices": 0,
            "num_vendors": 0,
        }

    return {
        "itc_at_risk_amount": round(float(df_risk["Risk Amount"].sum()), 2),
        "num_risky_invoices": int(len(df_risk)),
        "num_vendors": int(df_risk.loc[df_risk["GSTIN"].ne(""), "GSTIN"].nunique()),
    }


def build_client_report(reconciliation_df: pd.DataFrame) -> pd.DataFrame:
    if reconciliation_df.empty:
        return pd.DataFrame(columns=CLIENT_REPORT_COLUMNS)

    client_report = reconciliation_df.copy()
    client_report = client_report[client_report["Status"] != "MATCHED"].copy()
    client_report["Amount at Risk"] = client_report["Difference"].abs().round(2)
    client_report = client_report.rename(columns={"Reason": "Issue"})
    return client_report[
        [
            "GSTIN",
            "Invoice Number",
            "Issue",
            "ITC Status",
            "Amount at Risk",
            "Recommended Action",
        ]
    ].reset_index(drop=True)


def reconcile_invoices(
    purchase_raw: pd.DataFrame,
    gstr2b_raw: pd.DataFrame,
    value_tolerance: float = 2.0,
    date_tolerance_days: int = 5,
    fuzzy_threshold: float = 0.92,
) -> pd.DataFrame:
    config = ReconciliationConfig(
        value_tolerance=float(value_tolerance),
        date_tolerance_days=int(date_tolerance_days),
        fuzzy_threshold=float(fuzzy_threshold),
    )

    purchase_all = clean_invoice_data(purchase_raw, "Purchase Register")
    gstr2b_all = clean_invoice_data(gstr2b_raw, "GSTR-2B")

    output_rows: List[Dict[str, object]] = []
    matched_purchase_ids: Set[int] = set()
    matched_gstr2b_ids: Set[int] = set()
    warnings: List[str] = []

    for dataset in (purchase_all, gstr2b_all):
        precheck_rows = dataset[~dataset["eligible_for_matching"]].copy()
        for _, row in precheck_rows.iterrows():
            output_rows.append(
                _build_source_output_row(
                    row,
                    row["precheck_status"],
                    row["precheck_reason"],
                    "PRECHECK",
                )
            )

    purchase_working = purchase_all.loc[purchase_all["eligible_for_matching"]].copy()
    gstr2b_working = gstr2b_all.loc[gstr2b_all["eligible_for_matching"]].copy()

    def current_frames() -> Tuple[pd.DataFrame, pd.DataFrame]:
        purchase_remaining = purchase_working.loc[~purchase_working["source_row_id"].isin(matched_purchase_ids)].copy()
        gstr2b_remaining = gstr2b_working.loc[~gstr2b_working["source_row_id"].isin(matched_gstr2b_ids)].copy()
        return purchase_remaining, gstr2b_remaining

    def register_ambiguous(
        candidates: pd.DataFrame,
        purchase_remaining: pd.DataFrame,
        gstr2b_remaining: pd.DataFrame,
    ) -> pd.DataFrame:
        ambiguous_purchase_ids, ambiguous_gstr2b_ids = _extract_ambiguous_ids(
            candidates,
            "source_row_id_pr",
            "source_row_id_2b",
        )
        if ambiguous_purchase_ids or ambiguous_gstr2b_ids:
            _append_flagged_rows(
                output_rows,
                purchase_remaining,
                ambiguous_purchase_ids,
                "AMBIGUOUS MATCH",
                "Multiple candidate matches found",
                "AMBIGUOUS",
            )
            _append_flagged_rows(
                output_rows,
                gstr2b_remaining,
                ambiguous_gstr2b_ids,
                "AMBIGUOUS MATCH",
                "Multiple candidate matches found",
                "AMBIGUOUS",
            )
            matched_purchase_ids.update(ambiguous_purchase_ids)
            matched_gstr2b_ids.update(ambiguous_gstr2b_ids)
            candidates = candidates[
                ~candidates["source_row_id_pr"].isin(ambiguous_purchase_ids)
                & ~candidates["source_row_id_2b"].isin(ambiguous_gstr2b_ids)
            ].copy()
        return candidates

    purchase_remaining, gstr2b_remaining = current_frames()
    exact_matches = _build_exact_candidates(purchase_remaining, gstr2b_remaining, config)
    exact_matches = register_ambiguous(exact_matches, purchase_remaining, gstr2b_remaining)
    exact_matches = _select_best_matches(
        exact_matches,
        "source_row_id_pr",
        "source_row_id_2b",
        ["key", "value_diff_abs", "date_diff_abs"],
    )
    for _, row in exact_matches.iterrows():
        matched_purchase_ids.add(int(row["source_row_id_pr"]))
        matched_gstr2b_ids.add(int(row["source_row_id_2b"]))
        output_rows.append(_build_match_output_row(row, config))

    purchase_remaining, gstr2b_remaining = current_frames()
    key_matches = _build_key_candidates(purchase_remaining, gstr2b_remaining)
    key_matches = register_ambiguous(key_matches, purchase_remaining, gstr2b_remaining)
    key_matches = _select_best_matches(
        key_matches,
        "source_row_id_pr",
        "source_row_id_2b",
        ["key", "value_diff_abs", "date_diff_abs"],
    )
    for _, row in key_matches.iterrows():
        matched_purchase_ids.add(int(row["source_row_id_pr"]))
        matched_gstr2b_ids.add(int(row["source_row_id_2b"]))
        output_rows.append(_build_match_output_row(row, config))

    purchase_remaining, gstr2b_remaining = current_frames()
    large_dataset_mode = (
        len(purchase_remaining) > config.large_dataset_row_threshold
        or len(gstr2b_remaining) > config.large_dataset_row_threshold
    )
    if large_dataset_mode:
        warnings.append("Large dataset mode enabled. Fuzzy matching was skipped to preserve performance.")
    else:
        fuzzy_matches = _build_fuzzy_candidates(purchase_remaining, gstr2b_remaining, config)
        fuzzy_matches = register_ambiguous(fuzzy_matches, purchase_remaining, gstr2b_remaining)
        fuzzy_matches = _select_best_matches(
            fuzzy_matches,
            "source_row_id_pr",
            "source_row_id_2b",
            ["invoice_similarity", "value_diff_abs", "date_diff_abs"],
            ascending=[False, True, True],
        )
        for _, row in fuzzy_matches.iterrows():
            matched_purchase_ids.add(int(row["source_row_id_pr"]))
            matched_gstr2b_ids.add(int(row["source_row_id_2b"]))
            output_rows.append(_build_match_output_row(row, config))

    purchase_remaining, gstr2b_remaining = current_frames()
    for _, row in purchase_remaining.iterrows():
        output_rows.append(
            _build_source_output_row(
                row,
                "MISSING_IN_2B",
                "Supplier not filed",
                "UNMATCHED",
            )
        )
    for _, row in gstr2b_remaining.iterrows():
        output_rows.append(
            _build_source_output_row(
                row,
                "ONLY_IN_2B",
                "Present only in GSTR-2B",
                "UNMATCHED",
            )
        )

    reconciliation_df = pd.DataFrame(output_rows)
    if reconciliation_df.empty:
        reconciliation_df.attrs["processing_log"] = {
            "purchase_rows_processed": 0,
            "gstr2b_rows_processed": 0,
            "matches": 0,
            "exceptions": 0,
        }
        reconciliation_df.attrs["warnings"] = warnings
        return reconciliation_df

    for column in ["Purchase Value", "2B Value", "Difference"]:
        reconciliation_df[column] = reconciliation_df[column].fillna(0.0).round(2)

    reconciliation_df["ITC Status"] = reconciliation_df.apply(assign_itc_status, axis=1)
    invalid_rows = reconciliation_df[
        (reconciliation_df["Status"] == "DATE ERROR")
        & (reconciliation_df["ITC Status"] == "AT RISK")
    ]
    print("DATE ERROR rows check:")
    print(reconciliation_df[reconciliation_df["Status"] == "DATE ERROR"][["Invoice Number", "ITC Status"]])
    assert len(invalid_rows) == 0, "DATE ERROR incorrectly classified as AT RISK"

    reconciliation_df["Explanation"] = reconciliation_df.apply(generate_explanation, axis=1)
    reconciliation_df["Recommended Action"] = reconciliation_df.apply(generate_action, axis=1)

    reconciliation_df["Status Rank"] = reconciliation_df["Status"].map(_status_rank)
    reconciliation_df = reconciliation_df.sort_values(
        ["Status Rank", "GSTIN", "Invoice Number", "Source"],
        ascending=[True, True, True, True],
    ).drop(columns=["Status Rank"])

    matches = int((reconciliation_df["Status"] == "MATCHED").sum())
    exceptions = int(_is_exception(reconciliation_df).sum())
    reconciliation_df.attrs["processing_log"] = {
        "purchase_rows_processed": int(len(purchase_all)),
        "gstr2b_rows_processed": int(len(gstr2b_all)),
        "matches": matches,
        "exceptions": exceptions,
    }
    reconciliation_df.attrs["warnings"] = warnings
    return reconciliation_df.reset_index(drop=True)


def build_summary_metrics(reconciliation_df: pd.DataFrame) -> Dict[str, float]:
    if reconciliation_df.empty:
        return {
            "total_invoices": 0,
            "matched_percentage": 0.0,
            "itc_at_risk_amount": 0.0,
            "missing_invoices_count": 0,
        }

    base_scope = reconciliation_df[reconciliation_df["Status"] != "ONLY_IN_2B"].copy()
    total_invoices = int(len(base_scope))
    matched_percentage = float((base_scope["Status"] == "MATCHED").mean() * 100) if total_invoices else 0.0
    itc_risk_summary = build_itc_risk_summary(reconciliation_df)
    missing_invoices_count = int((reconciliation_df["Status"] == "MISSING_IN_2B").sum())

    return {
        "total_invoices": total_invoices,
        "matched_percentage": round(matched_percentage, 2),
        "itc_at_risk_amount": itc_risk_summary["itc_at_risk_amount"],
        "missing_invoices_count": missing_invoices_count,
    }


def build_vendor_summary(reconciliation_df: pd.DataFrame) -> pd.DataFrame:
    if reconciliation_df.empty:
        return pd.DataFrame(columns=["GSTIN", "Total Mismatches", "Total ITC Risk", "Missing Invoices"])

    df_risk = build_itc_risk_view(reconciliation_df)
    vendor_risk = (
        df_risk.groupby("GSTIN", dropna=False)["Risk Amount"].sum().reset_index(name="Total ITC Risk")
        if not df_risk.empty
        else pd.DataFrame(columns=["GSTIN", "Total ITC Risk"])
    )

    vendor_summary = (
        reconciliation_df.assign(
            mismatch_flag=lambda df: _is_exception(df).astype(int),
            missing_flag=lambda df: df["Status"].eq("MISSING_IN_2B").astype(int),
        )
        .groupby("GSTIN", dropna=False)
        .agg(
            Total_Mismatches=("mismatch_flag", "sum"),
            Missing_Invoices=("missing_flag", "sum"),
        )
        .reset_index()
        .rename(columns={"Total_Mismatches": "Total Mismatches", "Missing_Invoices": "Missing Invoices"})
    )
    vendor_summary = vendor_summary.merge(vendor_risk, on="GSTIN", how="left")
    vendor_summary["Total ITC Risk"] = vendor_summary["Total ITC Risk"].fillna(0.0)
    vendor_summary = vendor_summary.sort_values(["Total Mismatches", "Total ITC Risk"], ascending=[False, False])
    vendor_summary["Total ITC Risk"] = vendor_summary["Total ITC Risk"].round(2)
    return vendor_summary.reset_index(drop=True)


def generate_vendor_summary(reconciliation_df: pd.DataFrame) -> pd.DataFrame:
    if reconciliation_df.empty:
        return pd.DataFrame(
            columns=[
                "GSTIN",
                "Vendor Name",
                "Total Invoices",
                "Matched Invoices",
                "ITC At Risk",
                "Compliance Score",
                "Risk Level",
            ]
        )

    working_df = reconciliation_df.copy()
    working_df["GSTIN"] = working_df["GSTIN"].fillna("").astype(str)
    working_df["Supplier Name"] = working_df["Supplier Name"].fillna("").astype(str).str.strip()
    risk_view = build_itc_risk_view(working_df)
    vendor_risk = (
        risk_view.groupby(["GSTIN", "Supplier Name"], dropna=False)["Risk Amount"]
        .sum()
        .reset_index(name="ITC At Risk")
        if not risk_view.empty
        else pd.DataFrame(columns=["GSTIN", "Supplier Name", "ITC At Risk"])
    )

    summary = (
        working_df.groupby(["GSTIN", "Supplier Name"], dropna=False)
        .agg(
            total_invoices=("Invoice Number", "count"),
            matched_invoices=("Status", lambda x: (x == "MATCHED").sum()),
        )
        .reset_index()
        .rename(
            columns={
                "Supplier Name": "Vendor Name",
                "total_invoices": "Total Invoices",
                "matched_invoices": "Matched Invoices",
            }
        )
    )

    vendor_risk = vendor_risk.rename(columns={"Supplier Name": "Vendor Name"})
    summary = summary.merge(vendor_risk, on=["GSTIN", "Vendor Name"], how="left")
    summary["ITC At Risk"] = summary["ITC At Risk"].fillna(0.0).round(2)
    summary["Compliance Score"] = (
        (summary["Matched Invoices"] / summary["Total Invoices"].replace(0, np.nan)) * 100
    ).fillna(0.0).round(1)
    summary["Risk Level"] = np.where(summary["Compliance Score"] < 80, "High Risk", "Normal")
    return summary.sort_values(by="ITC At Risk", ascending=False).reset_index(drop=True)


def build_follow_up_sheet(reconciliation_df: pd.DataFrame) -> pd.DataFrame:
    if reconciliation_df.empty:
        return pd.DataFrame(columns=["GSTIN", "Issue Type", "Invoice Count", "Suggested Action"])

    follow_up = (
        reconciliation_df[_is_exception(reconciliation_df)]
        .groupby(["GSTIN", "Reason"], dropna=False)
        .size()
        .reset_index(name="Invoice Count")
        .rename(columns={"Reason": "Issue Type"})
    )
    follow_up["Status"] = follow_up["Issue Type"].map(
        {
            "Supplier not filed": "MISSING_IN_2B",
            "Value mismatch": "VALUE_MISMATCH",
            "Invoice uploaded in different period": "DATE_MISMATCH",
            "Present only in GSTR-2B": "ONLY_IN_2B",
            "Duplicate invoice": "REVIEW REQUIRED",
            "Multiple candidate matches found": "AMBIGUOUS MATCH",
            "Missing invoice number": "INVALID DATA",
            "Invalid GSTIN format": "INVALID GSTIN",
            "Invoice date could not be parsed": "DATE ERROR",
        }
    ).fillna("AT_RISK")
    follow_up["Suggested Action"] = follow_up.apply(
        lambda row: generate_action(pd.Series({"Reason": row["Issue Type"], "Status": row["Status"]})),
        axis=1,
    )
    follow_up = follow_up.drop(columns=["Status"])
    return follow_up.sort_values(["Invoice Count", "GSTIN"], ascending=[False, True]).reset_index(drop=True)
