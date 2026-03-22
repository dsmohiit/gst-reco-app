# GST Reconciliation Assistant

Streamlit MVP for reconciling GSTR-2B against a purchase register and generating:

- invoice-level reconciliation output
- exception reporting
- ITC risk insights
- vendor follow-up views

## Run locally

```powershell
pip install -r requirements.txt
streamlit run app.py
```

## Expected input columns

- GSTIN
- Supplier Name
- Invoice Number
- Invoice Date
- Taxable Value
- IGST
- CGST
- SGST

The app accepts `.csv`, `.xlsx`, and `.xls` files.
