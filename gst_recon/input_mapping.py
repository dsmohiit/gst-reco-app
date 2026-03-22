from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import re
from typing import Dict, List, Sequence, Tuple

import pandas as pd


COLUMN_SYNONYMS = {
    "gstin": ["gstin", "gst number", "gst no", "supplier gstin"],
    "invoice_number": ["invoice number", "invoice no", "inv no", "bill no", "document number"],
    "invoice_date": ["invoice date", "inv date", "bill date", "date"],
    "taxable_value": ["taxable value", "taxable amount", "assessable value", "value"],
    "vendor_name": ["tradelegal name", "supplier name", "vendor name", "party name", "name of supplier"],
}

FIELD_LABELS = {
    "gstin": "GSTIN",
    "invoice_number": "Invoice Number",
    "invoice_date": "Invoice Date",
    "taxable_value": "Purchase Value",
    "vendor_name": "Supplier Name",
}

REQUIRED_MAPPING_FIELDS = ["gstin", "invoice_number", "taxable_value"]
HEADER_SIGNPOST_KEYWORDS = [
    "gstin",
    "invoice",
    "inv no",
    "inv date",
    "date",
    "taxable",
    "taxable value",
    "total",
    "amount",
    "supplier",
]


class InputMappingError(ValueError):
    pass


@dataclass(frozen=True)
class InputMappingResult:
    dataframe: pd.DataFrame
    suggested_mapping: Dict[str, str | None]
    ambiguous_columns: Dict[str, List[str]]
    confidence: Dict[str, str]
    review_required: Dict[str, bool]


def _sanitize_column_names(columns: Sequence[object]) -> List[str]:
    sanitized: List[str] = []
    for col in columns:
        column_name = str(col).strip().lower()
        compact_name = re.sub(r"[^a-z0-9_]", "", column_name)
        if column_name.startswith("_") or compact_name in {"source_file", "sourcefile"}:
            if compact_name in {"source_file", "sourcefile"}:
                sanitized.append("source_file")
            else:
                sanitized.append(re.sub(r"\s+", " ", column_name).strip())
            continue
        cleaned = re.sub(r"[^a-z0-9\s]", "", column_name)
        sanitized.append(re.sub(r"\s+", " ", cleaned).strip())
    return sanitized


def _read_uploaded_file(uploaded_file, header=0, skiprows=None) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    if name.endswith(".csv"):
        return pd.read_csv(BytesIO(file_bytes), header=header, skiprows=skiprows)

    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(BytesIO(file_bytes), header=header, skiprows=skiprows)

    raise InputMappingError("Unsupported file format.")


def detect_header_row(df_raw: pd.DataFrame, max_rows: int = 15) -> Tuple[int, int]:
    if df_raw.empty:
        return 0, 0

    best_row_index = 0
    best_score = 0

    rows_to_scan = min(max_rows, len(df_raw))
    for row_index in range(rows_to_scan):
        row_series = pd.Series(df_raw.iloc[row_index], dtype="object").fillna("").astype(str).str.strip().str.lower()
        row_score = 0
        for keyword in HEADER_SIGNPOST_KEYWORDS:
            if row_series.str.contains(keyword, case=False, regex=False, na=False).any():
                row_score += 1

        if row_score > best_score:
            best_score = row_score
            best_row_index = row_index

    return best_row_index, best_score


def detect_file_header(uploaded_file, max_rows: int = 15) -> Tuple[int, int]:
    raw_df = _read_uploaded_file(uploaded_file, header=None)
    if raw_df.empty:
        raise InputMappingError("Uploaded file is empty or corrupted.")
    return detect_header_row(raw_df, max_rows=max_rows)


def load_input_file(uploaded_file, skiprows: int | None = None) -> pd.DataFrame:
    detected_header_row = 0
    detected_score = 0
    auto_detected = False

    if skiprows is None:
        raw_df = _read_uploaded_file(uploaded_file, header=None)
        if raw_df.empty:
            raise InputMappingError("Uploaded file is empty or corrupted.")
        detected_header_row, detected_score = detect_header_row(raw_df)
        uploaded_file.seek(0)
        effective_skiprows = int(detected_header_row) if detected_score > 0 else 0
        auto_detected = detected_score > 0
    else:
        effective_skiprows = int(skiprows) if skiprows else 0

    df = _read_uploaded_file(uploaded_file, header=0, skiprows=effective_skiprows)

    if df.empty:
        raise InputMappingError("Uploaded file is empty or corrupted.")

    df.columns = _sanitize_column_names(df.columns)
    df.attrs["detected_header_row"] = int(detected_header_row)
    df.attrs["header_detection_score"] = int(detected_score)
    df.attrs["header_auto_detected"] = auto_detected
    df.attrs["header_detection_warning"] = bool(auto_detected and detected_score == 0)
    return df


def _make_unique(columns: Sequence[str]) -> List[str]:
    counts: Dict[str, int] = {}
    unique_columns: List[str] = []

    for column in columns:
        count = counts.get(column, 0) + 1
        counts[column] = count
        if count == 1:
            unique_columns.append(column)
        else:
            unique_columns.append(f"{column} {count}")

    return unique_columns


def preprocess_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise InputMappingError("Uploaded file is empty or corrupted.")

    cleaned = df.copy()
    cleaned = cleaned.dropna(how="all").copy()
    cleaned = cleaned.dropna(axis=1, how="all").copy()
    if cleaned.empty or cleaned.shape[1] == 0:
        raise InputMappingError("Uploaded file is empty or corrupted.")

    cleaned.columns = _sanitize_column_names(cleaned.columns)
    cleaned.columns = _make_unique(cleaned.columns.tolist())

    for column in cleaned.columns:
        if pd.api.types.is_object_dtype(cleaned[column]) or pd.api.types.is_string_dtype(cleaned[column]):
            cleaned[column] = cleaned[column].where(cleaned[column].notna(), "").astype(str).str.strip()

    return cleaned


def detect_columns(df_columns: Sequence[str], synonyms: Sequence[str]) -> List[str]:
    matches = []
    for col in df_columns:
        for syn in synonyms:
            if syn in col:
                matches.append(col)
                break
    return list(dict.fromkeys(matches))


def _normalize_synonyms(synonyms: Sequence[str]) -> List[str]:
    normalized = (
        pd.Index(synonyms, dtype="string")
        .str.lower()
        .str.strip()
        .str.replace(r"[^a-z0-9 ]", "", regex=True)
        .tolist()
    )
    return [value for value in normalized if value]


def build_column_mapping(
    df_columns: Sequence[str],
) -> Tuple[Dict[str, str | None], Dict[str, List[str]], Dict[str, str], Dict[str, bool]]:
    column_mapping: Dict[str, str | None] = {}
    ambiguous_columns: Dict[str, List[str]] = {}
    confidence: Dict[str, str] = {}
    review_required: Dict[str, bool] = {}

    for key, synonyms in COLUMN_SYNONYMS.items():
        normalized_synonyms = _normalize_synonyms(synonyms)
        exact_matches = [column for column in df_columns if column in normalized_synonyms]
        partial_matches = [
            column
            for column in detect_columns(df_columns, normalized_synonyms)
            if column not in exact_matches
        ]

        if len(exact_matches) == 1:
            column_mapping[key] = exact_matches[0]
            confidence[key] = "HIGH"
            review_required[key] = False
        elif len(exact_matches) > 1:
            ambiguous_columns[key] = exact_matches
            column_mapping[key] = None
            confidence[key] = "LOW"
            review_required[key] = True
        elif len(partial_matches) == 1:
            column_mapping[key] = partial_matches[0]
            confidence[key] = "MEDIUM"
            review_required[key] = False
        elif len(partial_matches) > 1:
            ambiguous_columns[key] = partial_matches
            column_mapping[key] = None
            confidence[key] = "LOW"
            review_required[key] = True
        else:
            column_mapping[key] = None
            confidence[key] = "MISSING"
            review_required[key] = True

    return column_mapping, ambiguous_columns, confidence, review_required


def inspect_input_dataframe(df: pd.DataFrame) -> InputMappingResult:
    cleaned = preprocess_input_dataframe(df)
    candidate_columns = [
        column for column in cleaned.columns.tolist()
        if not str(column).startswith("_") and str(column) != "source_file"
    ]
    mapping, ambiguous, confidence, review_required = build_column_mapping(candidate_columns)
    return InputMappingResult(
        dataframe=cleaned,
        suggested_mapping=mapping,
        ambiguous_columns=ambiguous,
        confidence=confidence,
        review_required=review_required,
    )


def validate_column_mapping(column_mapping: Dict[str, str | None], df_columns: Sequence[str]) -> None:
    available_columns = set(df_columns)

    for field in REQUIRED_MAPPING_FIELDS:
        selected = column_mapping.get(field)
        if not selected or selected not in available_columns:
            raise InputMappingError("Uploaded file is missing required columns.")

    selected_columns = [value for value in column_mapping.values() if value]
    if len(selected_columns) != len(set(selected_columns)):
        raise InputMappingError("Each logical field must map to a different source column.")


def standardize_mapped_dataframe(df: pd.DataFrame, column_mapping: Dict[str, str | None]) -> pd.DataFrame:
    return prepare_data(df, column_mapping)


def prepare_data(df: pd.DataFrame, mapping: Dict[str, str | None]) -> pd.DataFrame:
    working_df = df.copy()
    working_df.columns = _sanitize_column_names(working_df.columns)

    validate_column_mapping(mapping, working_df.columns.tolist())

    rename_map: Dict[str, str] = {}

    gstin_col = mapping.get("gstin")
    invoice_col = mapping.get("invoice_number")
    date_col = mapping.get("invoice_date")
    amount_col = mapping.get("taxable_value")
    vendor_col = mapping.get("vendor_name")

    if gstin_col in working_df.columns:
        rename_map[gstin_col] = "gstin_internal"
    if invoice_col in working_df.columns:
        rename_map[invoice_col] = "inv_no_internal"
    if date_col and date_col in working_df.columns:
        rename_map[date_col] = "date_internal"
    if amount_col in working_df.columns:
        rename_map[amount_col] = "amount_internal"
    if vendor_col and vendor_col in working_df.columns:
        rename_map[vendor_col] = "vendor_name_internal"

    working_df = working_df.rename(columns=rename_map).copy()

    required_internal_columns = ["gstin_internal", "inv_no_internal", "amount_internal"]
    missing_internal = [column for column in required_internal_columns if column not in working_df.columns]
    if missing_internal:
        raise InputMappingError("Uploaded file is missing required columns.")

    working_df["amount_internal"] = pd.to_numeric(working_df["amount_internal"], errors="coerce").fillna(0)
    working_df["gstin_internal"] = (
        working_df["gstin_internal"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )
    working_df["inv_no_internal"] = (
        working_df["inv_no_internal"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    working_df["norm_inv_no"] = (
        working_df["inv_no_internal"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]", "", regex=True)
        .str.lstrip("0")
    )

    if "date_internal" in working_df.columns:
        working_df["date_internal"] = pd.to_datetime(working_df["date_internal"], errors="coerce")
    else:
        working_df["date_internal"] = pd.NaT

    if "vendor_name_internal" in working_df.columns:
        working_df["vendor_name_internal"] = (
            working_df["vendor_name_internal"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )
    else:
        working_df["vendor_name_internal"] = ""

    standardized = pd.DataFrame(
        {
            "GSTIN": working_df["gstin_internal"],
            "Invoice Number": working_df["inv_no_internal"],
            "Invoice Date": working_df["date_internal"],
            "Purchase Value": working_df["amount_internal"],
            "Supplier Name": working_df["vendor_name_internal"],
            "Normalized Invoice Number": working_df["norm_inv_no"],
        }
    )

    if "source_file" in working_df.columns:
        standardized["source_file"] = (
            working_df["source_file"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    if standardized.empty:
        return standardized.copy()
    if standardized["Purchase Value"].notna().sum() == 0:
        raise InputMappingError("Purchase Value column could not be parsed. Please check mapping and data format.")
    return standardized.copy()
