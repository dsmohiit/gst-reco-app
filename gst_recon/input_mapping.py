from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Sequence, Tuple

import pandas as pd


COLUMN_SYNONYMS = {
    "gstin": ["gstin", "gst number", "gst no", "supplier gstin"],
    "invoice_number": ["invoice number", "invoice no", "inv no", "bill no", "document number"],
    "invoice_date": ["invoice date", "inv date", "bill date", "date"],
    "taxable_value": ["taxable value", "purchase value", "invoice value", "taxable amount"],
}

FIELD_LABELS = {
    "gstin": "GSTIN",
    "invoice_number": "Invoice Number",
    "invoice_date": "Invoice Date",
    "taxable_value": "Purchase Value",
}

REQUIRED_MAPPING_FIELDS = ["gstin", "invoice_number", "taxable_value"]


class InputMappingError(ValueError):
    pass


@dataclass(frozen=True)
class InputMappingResult:
    dataframe: pd.DataFrame
    suggested_mapping: Dict[str, str | None]
    ambiguous_columns: Dict[str, List[str]]


def load_input_file(uploaded_file, skiprows: int = 0) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()
    effective_skiprows = int(skiprows) if skiprows else None

    if name.endswith(".csv"):
        return pd.read_csv(BytesIO(file_bytes), skiprows=effective_skiprows)

    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(BytesIO(file_bytes), skiprows=effective_skiprows)

    raise InputMappingError("Unsupported file format.")


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

    cleaned.columns = (
        cleaned.columns.astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"[^a-z0-9 ]", "", regex=True)
    )
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


def build_column_mapping(df_columns: Sequence[str]) -> Tuple[Dict[str, str | None], Dict[str, List[str]]]:
    column_mapping: Dict[str, str | None] = {}
    ambiguous_columns: Dict[str, List[str]] = {}

    for key, synonyms in COLUMN_SYNONYMS.items():
        matches = detect_columns(df_columns, synonyms)
        if len(matches) == 1:
            column_mapping[key] = matches[0]
        elif len(matches) > 1:
            ambiguous_columns[key] = matches
            column_mapping[key] = None
        else:
            column_mapping[key] = None

    return column_mapping, ambiguous_columns


def inspect_input_dataframe(df: pd.DataFrame) -> InputMappingResult:
    cleaned = preprocess_input_dataframe(df)
    mapping, ambiguous = build_column_mapping(cleaned.columns.tolist())
    return InputMappingResult(
        dataframe=cleaned,
        suggested_mapping=mapping,
        ambiguous_columns=ambiguous,
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
    validate_column_mapping(column_mapping, df.columns.tolist())

    rename_map = {
        column_mapping["gstin"]: "GSTIN",
        column_mapping["invoice_number"]: "Invoice Number",
        column_mapping["taxable_value"]: "Purchase Value",
    }
    if column_mapping.get("invoice_date"):
        rename_map[column_mapping["invoice_date"]] = "Invoice Date"

    standardized = df.rename(columns=rename_map).copy()
    if "Invoice Date" not in standardized.columns:
        standardized["Invoice Date"] = pd.NaT

    standardized["Purchase Value"] = pd.to_numeric(standardized["Purchase Value"], errors="coerce")
    standardized["Invoice Date"] = pd.to_datetime(standardized["Invoice Date"], errors="coerce")
    if standardized["Purchase Value"].notna().sum() == 0:
        raise InputMappingError("Purchase Value column could not be parsed. Please check mapping and data format.")
    return standardized
