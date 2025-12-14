# Streamlit DEGIRO Portfolio Tracker (v1.1 - DEGIRO-match)
# Uploads:
# 1) Account.csv (DEGIRO account overview) - optional in v1.1, mainly for validation/links
# 2) Portfolio.csv (DEGIRO portfolio overview) - current quantities/prices/values
# 3) Optional: "UI/Tracky export" CSV that already contains DEGIRO-like metrics per ISIN (gak, unrealized, total, dividend, etc.)
#
# Goal of v1.1:
# - Match DEGIRO UI as close as possible.
# - If UI/Tracky export is provided, we use it as the source for GAK and performance columns (best match).
# - Portfolio.csv is used as source of "current snapshot" (qty/price/value). If both are present, Portfolio wins for "now".
#
# Notes:
# - Pure Account.csv allocation of dividends/taxes to ISIN is not always possible because DEGIRO's FX legs can be unlinked.
#   That is why the optional UI/Tracky export is supported for a true DEGIRO-match.

import io
import re
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="DEGIRO Portfolio Tracker (v1.1)", layout="wide")


# -----------------------------
# Helpers
# -----------------------------
def _to_float_eu(x) -> float:
    """Parse numbers like '1.234,56' or '123,45' or '123.45' into float."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return np.nan
    # Remove currency symbols and spaces
    s = re.sub(r"[€$]", "", s).strip()
    # Handle thousands separators: if both '.' and ',' exist, assume EU style '1.234,56'
    if "." in s and "," in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        # If only comma exists, treat comma as decimal separator
        if "," in s and "." not in s:
            s = s.replace(",", ".")
    # Remove any remaining non-numeric except minus and dot
    s = re.sub(r"[^0-9\.\-]", "", s)
    try:
        return float(s)
    except ValueError:
        return np.nan


def _norm_isin(x: str) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip().upper()


def _looks_like_isin(s: str) -> bool:
    s = _norm_isin(s)
    return bool(re.fullmatch(r"[A-Z0-9]{12}", s))


def _safe_read_csv(uploaded_file) -> pd.DataFrame:
    """Try common delimiters/encodings."""
    raw = uploaded_file.getvalue()
    # Try UTF-8 then latin-1
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except Exception:
            text = None
    if text is None:
        text = raw.decode("latin-1", errors="ignore")

    # Try separators
    for sep in (",", ";", "\t"):
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep)
            if df.shape[1] >= 2:
                return df
        except Exception:
            continue
    # Last resort
    return pd.read_csv(io.StringIO(text))


# -----------------------------
# Parsers
# -----------------------------
def parse_portfolio_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expected (common DEGIRO Portfolio export - NL):
      Product, Symbool/ISIN, Aantal, Slotkoers, Lokale waarde, <currency col>, Waarde in EUR

    Returns normalized columns:
      isin, product, qty_now, price_now, value_eur, currency
    """
    df = df.copy()
    # normalize column names
    cols = {c: c.strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)

    # Try to detect
    col_product = next((c for c in df.columns if c.lower() in {"product"}), None)
    col_isin = next((c for c in df.columns if "isin" in c.lower()), None)  # often 'Symbool/ISIN'
    col_qty = next((c for c in df.columns if c.lower() in {"aantal", "qty", "quantity"}), None)
    col_price = next((c for c in df.columns if c.lower() in {"slotkoers", "koers", "price"}), None)
    col_value_eur = next((c for c in df.columns if "waarde" in c.lower() and "eur" in c.lower()), None)

    # Currency: sometimes unnamed column with values EUR/USD next to local value
    col_currency = None
    for c in df.columns:
        if "valuta" in c.lower():
            col_currency = c
            break

    if col_product is None or col_isin is None or col_qty is None or col_price is None or col_value_eur is None:
        raise ValueError("Kon Portfolio.csv kolommen niet herkennen. Verwacht o.a. Product, Symbool/ISIN, Aantal, Slotkoers, Waarde in EUR.")

    out = pd.DataFrame({
        "product": df[col_product].astype(str).fillna(""),
        "isin": df[col_isin].apply(_norm_isin),
        "qty_now": df[col_qty].apply(_to_float_eu),
        "price_now": df[col_price].apply(_to_float_eu),
        "value_eur": df[col_value_eur].apply(_to_float_eu),
    })

    # drop cash rows without ISIN
    out = out[out["isin"].apply(_looks_like_isin)].copy()

    if col_currency:
        out["currency"] = df[col_currency].astype(str).str.strip().replace({"nan": ""})
    else:
        # heuristic: if there is a column with exactly EUR/USD values
        currency_guess = None
        for c in df.columns:
            vals = set(df[c].dropna().astype(str).str.strip().unique().tolist())
            if vals.issubset({"EUR", "USD", "GBP", "CHF", "DKK", "SEK"}) and len(vals) > 0:
                currency_guess = c
                break
        if currency_guess:
            out["currency"] = df[currency_guess].astype(str).str.strip()
        else:
            out["currency"] = ""

    return out


def parse_ui_export_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supports a 'UI/Tracky export' shaped like:
      product, isin, qty_now, price_now, value_eur, gak_eur, cost_eur, unrealized_eur, unrealized_pct,
      realized_eur, dividend_eur, withholding_tax_eur, total_return_eur, total_return_pct

    Returns those normalized columns (isin as key).
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"isin", "value_eur", "gak_eur", "cost_eur", "unrealized_eur", "total_return_eur"}
    if not required.issubset(set(df.columns)):
        raise ValueError("Kon UI/Tracky export niet herkennen. Verwacht kolommen zoals: isin, value_eur, gak_eur, cost_eur, unrealized_eur, total_return_eur...")

    out = pd.DataFrame()
    out["isin"] = df["isin"].apply(_norm_isin)
    out["product"] = df["product"] if "product" in df.columns else ""
    for col in ["qty_now", "price_now", "value_eur", "gak_eur", "cost_eur",
                "unrealized_eur", "unrealized_pct", "realized_eur",
                "dividend_eur", "withholding_tax_eur",
                "total_return_eur", "total_return_pct"]:
        if col in df.columns:
            out[col] = df[col].apply(_to_float_eu)
        else:
            out[col] = np.nan

    out = out[out["isin"].apply(_looks_like_isin)].copy()
    return out


def parse_account_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic parser for Account.csv. In v1.1 we mainly use it for:
    - sanity checks, and
    - (optional) realized P/L from trades when no UI export is provided (limited dividend allocation).
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    needed = {"Datum", "Omschrijving", "Mutatie"}
    if not needed.issubset(set(df.columns)):
        raise ValueError("Kon Account.csv niet herkennen. Verwacht kolommen zoals: Datum, Omschrijving, Mutatie, Product, ISIN, Order Id.")

    # Amount column in many DEGIRO NL exports is unnamed 'Unnamed: 8'
    amt_col = None
    for c in df.columns:
        if c.lower().startswith("unnamed") and "8" in c:
            amt_col = c
            break
    if amt_col is None:
        # fallback: find numeric-like column besides Mutatie/Saldo
        candidates = [c for c in df.columns if c not in {"Datum", "Tijd", "Valutadatum", "Product", "ISIN", "Omschrijving", "FX", "Mutatie", "Saldo", "Order Id"}]
        amt_col = candidates[0] if candidates else None
    if amt_col is None:
        raise ValueError("Kon bedrag-kolom in Account.csv niet vinden (meestal 'Unnamed: 8').")

    out = pd.DataFrame({
        "date": df["Datum"].astype(str),
        "time": df["Tijd"].astype(str) if "Tijd" in df.columns else "",
        "product": df["Product"].astype(str) if "Product" in df.columns else "",
        "isin": df["ISIN"].apply(_norm_isin) if "ISIN" in df.columns else "",
        "desc": df["Omschrijving"].astype(str),
        "ccy": df["Mutatie"].astype(str).str.strip(),
        "amount": df[amt_col].apply(_to_float_eu),
        "order_id": df["Order Id"].astype(str) if "Order Id" in df.columns else "",
        "fx": df["FX"].apply(_to_float_eu) if "FX" in df.columns else np.nan
    })
    out["order_id"] = out["order_id"].replace({"nan": ""})
    return out


# -----------------------------
# Calculations (fallback)
# -----------------------------
@dataclass
class PositionCalc:
    isin: str
    product: str
    qty: float
    cost_eur: float
    gak_eur: float
    realized_eur: float
    fees_eur: float


def compute_from_account_only(acc: pd.DataFrame) -> pd.DataFrame:
    """
    Fallback computation (best effort) when UI export is not provided.
    Computes:
      - cost basis and GAK using average cost method
      - realized P/L
      - fees allocated via 'DEGIRO Transactiekosten...' rows linked by Order Id
    Dividend/tax allocation in EUR can be incomplete if FX legs are not linked to ISIN.
    """
    acc = acc.copy()
    # Identify trades: desc starts with Koop/Verkoop and includes '@'
    is_trade = acc["desc"].str.startswith(("Koop", "Verkoop"))
    trades = acc[is_trade].copy()

    # Parse qty and trade currency price from desc: "Koop 2 @ 71,16 USD"
    def parse_trade_desc(s: str) -> Tuple[str, float, float, str]:
        # returns side, qty, price, ccy
        m = re.match(r"^(Koop|Verkoop)\s+([0-9]+)\s+@\s+([0-9\.,]+)\s+([A-Z]{3})", s.strip())
        if not m:
            return "", np.nan, np.nan, ""
        side = "BUY" if m.group(1) == "Koop" else "SELL"
        qty = float(m.group(2))
        price = _to_float_eu(m.group(3))
        ccy = m.group(4)
        return side, qty, price, ccy

    parsed = trades["desc"].apply(parse_trade_desc)
    trades["side"] = parsed.apply(lambda x: x[0])
    trades["qty"] = parsed.apply(lambda x: x[1])
    trades["price"] = parsed.apply(lambda x: x[2])
    trades["trade_ccy"] = parsed.apply(lambda x: x[3])

    # For each Order ID, compute EUR cash impact:
    # - For EUR trades: amount already in EUR (acc.amount where desc is trade and ccy=EUR)
    # - For non-EUR trades: use the EUR leg "Valuta Debitering/Creditering" within same order_id
    # - Add fees in EUR (fees rows linked by same order_id)
    # Convention: acc.amount positive = credit, negative = debit.
    # For BUY, cost_eur should be positive outflow => -EUR_debit (make positive).
    # For SELL, proceeds_eur positive inflow => EUR_credit (positive).
    oids = trades["order_id"].fillna("").astype(str).unique().tolist()

    # fees by oid (EUR)
    fees = acc[acc["desc"].str.contains("Transactiekosten", case=False, na=False)].copy()
    fees_eur_by_oid = fees[fees["ccy"].eq("EUR")].groupby("order_id")["amount"].sum().to_dict()

    # EUR legs by oid
    eur_legs = acc[acc["ccy"].eq("EUR") & acc["desc"].isin(["Valuta Debitering", "Valuta Creditering"])].copy()
    eur_leg_by_oid = eur_legs.groupby("order_id")["amount"].sum().to_dict()  # net EUR change from FX legs

    # For EUR trades without FX legs, use trade row amount in EUR
    trade_eur_amount_by_oid = trades[trades["ccy"].eq("EUR")].groupby("order_id")["amount"].sum().to_dict()

    # Now compute per trade row eur_cash
    def eur_cash_for_row(r):
        oid = str(r["order_id"])
        if r["trade_ccy"] == "EUR":
            eur_amt = trade_eur_amount_by_oid.get(oid, np.nan)
            if np.isnan(eur_amt):
                eur_amt = r["amount"]
        else:
            eur_amt = eur_leg_by_oid.get(oid, np.nan)
        fee = fees_eur_by_oid.get(oid, 0.0)
        return eur_amt, fee

    tmp = trades.apply(lambda r: eur_cash_for_row(r), axis=1, result_type="expand")
    trades["eur_cash_net"] = tmp[0]  # net EUR change due to FX leg (BUY negative, SELL positive)
    trades["fee_eur"] = tmp[1]

    # If eur_cash_net is NaN, we can't compute; drop those (rare)
    trades = trades.dropna(subset=["eur_cash_net"]).copy()

    # Average cost per ISIN
    positions: Dict[str, PositionCalc] = {}
    for _, r in trades.sort_values(["date", "time"]).iterrows():
        isin = _norm_isin(r["isin"])
        if not _looks_like_isin(isin):
            continue
        prod = r["product"]
        side = r["side"]
        qty = float(r["qty"])
        eur_cash = float(r["eur_cash_net"])
        fee = float(r["fee_eur"])
        # BUY: eur_cash negative => cost positive
        if isin not in positions:
            positions[isin] = PositionCalc(isin=isin, product=prod, qty=0.0, cost_eur=0.0, gak_eur=0.0, realized_eur=0.0, fees_eur=0.0)

        p = positions[isin]
        if side == "BUY":
            cost = -(eur_cash) + fee  # make positive
            p.qty += qty
            p.cost_eur += cost
            p.fees_eur += fee
            p.gak_eur = (p.cost_eur / p.qty) if p.qty > 0 else 0.0
        else:  # SELL
            proceeds = eur_cash  # positive
            avg = (p.cost_eur / p.qty) if p.qty > 0 else 0.0
            cost_sold = avg * qty
            p.qty -= qty
            p.cost_eur -= cost_sold
            p.realized_eur += (proceeds - cost_sold - fee)
            p.fees_eur += fee
            p.gak_eur = (p.cost_eur / p.qty) if p.qty > 0 else 0.0

    out = pd.DataFrame([{
        "isin": k,
        "product": v.product,
        "qty_calc": v.qty,
        "cost_eur": v.cost_eur,
        "gak_eur": v.gak_eur,
        "realized_eur": v.realized_eur,
        "fees_eur": v.fees_eur,
    } for k, v in positions.items()])

    return out


# -----------------------------
# UI
# -----------------------------
st.title("DEGIRO Portfolio Tracker (v1.1 – DEGIRO-match)")
st.caption("Upload je DEGIRO exports. Voor de beste match met DEGIRO UI: upload óók een UI/Tracky export met GAK en (on)gerealiseerde W/V per ISIN.")

colA, colB, colC = st.columns(3)
with colA:
    up_account = st.file_uploader("Upload Account.csv (DEGIRO – rekeningoverzicht)", type=["csv"], key="acc")
with colB:
    up_portfolio = st.file_uploader("Upload Portfolio.csv (DEGIRO – portefeuille)", type=["csv"], key="port")
with colC:
    up_ui = st.file_uploader("Optioneel: UI/Tracky export (met GAK & W/V per ISIN)", type=["csv"], key="ui")

st.divider()

if not up_portfolio and not up_ui:
    st.info("Upload minimaal **Portfolio.csv** (voor actuele waarde) of een **UI/Tracky export** (die al actuele waarde + resultaten bevat).")
    st.stop()

# Read & parse
portfolio_df = None
ui_df = None
account_df = None

errors = []

try:
    if up_portfolio:
        portfolio_raw = _safe_read_csv(up_portfolio)
        portfolio_df = parse_portfolio_csv(portfolio_raw)
except Exception as e:
    errors.append(f"Portfolio.csv: {e}")

try:
    if up_ui:
        ui_raw = _safe_read_csv(up_ui)
        ui_df = parse_ui_export_csv(ui_raw)
except Exception as e:
    errors.append(f"UI/Tracky export: {e}")

try:
    if up_account:
        acc_raw = _safe_read_csv(up_account)
        account_df = parse_account_csv(acc_raw)
except Exception as e:
    errors.append(f"Account.csv: {e}")

if errors:
    st.error("Er gingen dingen mis bij het inlezen:\n\n- " + "\n- ".join(errors))
    st.stop()

# Build base table
if ui_df is not None:
    base = ui_df.copy()
else:
    # fallback: compute from account only + merge portfolio snapshot
    if account_df is None:
        st.error("Zonder UI/Tracky export heb je **Account.csv** nodig om GAK/kostbasis te berekenen.")
        st.stop()
    calc = compute_from_account_only(account_df)
    base = calc.rename(columns={"qty_calc": "qty_now"}).copy()
    base["price_now"] = np.nan
    base["value_eur"] = np.nan
    base["unrealized_eur"] = np.nan
    base["unrealized_pct"] = np.nan
    base["dividend_eur"] = 0.0
    base["withholding_tax_eur"] = 0.0
    base["total_return_eur"] = np.nan
    base["total_return_pct"] = np.nan
    base["realized_eur"] = base["realized_eur"].fillna(0.0)

# Merge portfolio snapshot to get accurate "now"
if portfolio_df is not None:
    snap = portfolio_df[["isin", "product", "qty_now", "price_now", "value_eur", "currency"]].copy()
    out = base.merge(snap, on="isin", how="outer", suffixes=("_base", ""))
    # Choose product name
    out["product"] = out["product"].where(out["product"].astype(str).str.len() > 0, out.get("product_base", ""))
    # If base had qty_now scaled (e.g., *10) and portfolio has smaller, auto-correct scale
    if "qty_now_base" in out.columns:
        # Determine scaling per row
        q_base = out["qty_now_base"]
        q_snap = out["qty_now"]
        scale_fix = (q_base.notna() & q_snap.notna() & (q_base > 0) & (q_snap > 0) & (np.isclose(q_base / q_snap, 10.0, atol=1e-6)))
        # If scale_fix, overwrite base-derived quantities and cost basis scaling accordingly
        out["scale_fix_10x"] = scale_fix
        # Prefer portfolio quantities always (DEGIRO truth)
        out["qty_now_final"] = out["qty_now"]
        # If we used base for cost and it's based on scaled qty, fix cost basis per share? In UI export cost_eur is correct, so do not change.
        # If cost_eur is NaN (unlikely), keep.
    else:
        out["qty_now_final"] = out["qty_now"]

    # Overwrite price/value with snapshot when present
    out["price_now_final"] = out["price_now"].where(out["price_now"].notna(), out.get("price_now_base", np.nan))
    out["value_eur_final"] = out["value_eur"].where(out["value_eur"].notna(), out.get("value_eur_base", np.nan))

else:
    out = base.copy()
    out["qty_now_final"] = out["qty_now"]
    out["price_now_final"] = out["price_now"]
    out["value_eur_final"] = out["value_eur"]
    out["currency"] = ""

# Final calculations for DEGIRO-like columns
out["cost_eur"] = out.get("cost_eur", np.nan)
out["gak_eur"] = out.get("gak_eur", np.nan)

# If GAK missing but we have cost and qty, compute
mask_gak = out["gak_eur"].isna() & out["cost_eur"].notna() & (out["qty_now_final"] > 0)
out.loc[mask_gak, "gak_eur"] = out.loc[mask_gak, "cost_eur"] / out.loc[mask_gak, "qty_now_final"]

# Unrealized from snapshot value and GAK
out["unrealized_eur_calc"] = out["value_eur_final"] - (out["gak_eur"].fillna(0.0) * out["qty_now_final"].fillna(0.0))
out["unrealized_pct_calc"] = np.where(
    out["cost_eur"].fillna(0.0) != 0,
    (out["unrealized_eur_calc"] / out["cost_eur"]) * 100.0,
    np.nan
)

# If UI export provided unrealized values, keep those for best match; else use calculated
if "unrealized_eur" in out.columns and out["unrealized_eur"].notna().any():
    out["unrealized_eur_final"] = out["unrealized_eur"]
    out["unrealized_pct_final"] = out["unrealized_pct"]
else:
    out["unrealized_eur_final"] = out["unrealized_eur_calc"]
    out["unrealized_pct_final"] = out["unrealized_pct_calc"]

# Total return: prefer provided, else compute = unrealized + realized + dividend + withholding_tax
out["realized_eur"] = out.get("realized_eur", 0.0).fillna(0.0)
out["dividend_eur"] = out.get("dividend_eur", 0.0).fillna(0.0)
out["withholding_tax_eur"] = out.get("withholding_tax_eur", 0.0).fillna(0.0)

if "total_return_eur" in out.columns and out["total_return_eur"].notna().any():
    out["total_return_eur_final"] = out["total_return_eur"]
    out["total_return_pct_final"] = out["total_return_pct"]
else:
    out["total_return_eur_final"] = out["unrealized_eur_final"] + out["realized_eur"] + out["dividend_eur"] + out["withholding_tax_eur"]
    out["total_return_pct_final"] = np.where(
        out["cost_eur"].fillna(0.0) != 0,
        (out["total_return_eur_final"] / out["cost_eur"]) * 100.0,
        np.nan
    )

# Display formatting
def fmt_eur(x):
    if pd.isna(x):
        return ""
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def fmt_pct(x):
    if pd.isna(x):
        return ""
    return f"{x:.2f}%".replace(".", ",")

show = pd.DataFrame({
    "Product": out["product"].fillna(""),
    "ISIN": out["isin"],
    "Aantal": out["qty_now_final"],
    "Koers": out["price_now_final"],
    "Valuta": out["currency"].fillna(""),
    "Waarde (EUR)": out["value_eur_final"],
    "GAK (EUR)": out["gak_eur"],
    "Ongerealiseerd W/V €": out["unrealized_eur_final"],
    "Ongerealiseerd W/V %": out["unrealized_pct_final"],
    "Dividend €": out["dividend_eur"],
    "Bronbelasting €": out["withholding_tax_eur"],
    "Gerealiseerd €": out["realized_eur"],
    "Totale W/V €": out["total_return_eur_final"],
    "Totale W/V %": out["total_return_pct_final"],
})

# Summary KPIs
total_value = np.nansum(show["Waarde (EUR)"].values.astype(float))
total_cost = np.nansum(out["cost_eur"].fillna(0.0).values.astype(float))
total_unreal = np.nansum(show["Ongerealiseerd W/V €"].fillna(0.0).values.astype(float))
total_total = np.nansum(show["Totale W/V €"].fillna(0.0).values.astype(float))

k1, k2, k3, k4 = st.columns(4)
k1.metric("Totale waarde (EUR)", fmt_eur(total_value))
k2.metric("Totale kostbasis (EUR)", fmt_eur(total_cost))
k3.metric("Ongerealiseerd (EUR)", fmt_eur(total_unreal))
k4.metric("Totale W/V (EUR)", fmt_eur(total_total))

st.divider()

# Filters
c1, c2 = st.columns([2,1])
with c1:
    q = st.text_input("Zoek (product/ISIN)", "")
with c2:
    only_open = st.checkbox("Alleen open posities (waarde > 0)", value=True)

tbl = show.copy()
if only_open:
    tbl = tbl[tbl["Waarde (EUR)"].fillna(0.0) != 0.0]
if q.strip():
    qq = q.strip().lower()
    tbl = tbl[
        tbl["Product"].str.lower().str.contains(qq, na=False)
        | tbl["ISIN"].str.lower().str.contains(qq, na=False)
    ]

# Sort: largest value
tbl = tbl.sort_values("Waarde (EUR)", ascending=False)

# Pretty display
display_tbl = tbl.copy()
for c in ["Waarde (EUR)", "GAK (EUR)", "Ongerealiseerd W/V €", "Dividend €", "Bronbelasting €", "Gerealiseerd €", "Totale W/V €"]:
    display_tbl[c] = display_tbl[c].apply(fmt_eur)
for c in ["Ongerealiseerd W/V %", "Totale W/V %"]:
    display_tbl[c] = display_tbl[c].apply(fmt_pct)
display_tbl["Aantal"] = display_tbl["Aantal"].apply(lambda x: "" if pd.isna(x) else (str(int(x)) if float(x).is_integer() else str(x)))
display_tbl["Koers"] = display_tbl["Koers"].apply(lambda x: "" if pd.isna(x) else f"{x:.4f}".rstrip("0").rstrip(".").replace(".", ","))

st.dataframe(display_tbl, use_container_width=True, hide_index=True)

st.divider()

with st.expander("Debug / checks"):
    st.write("Upload-status:")
    st.write({
        "account.csv": bool(up_account),
        "portfolio.csv": bool(up_portfolio),
        "ui_export.csv": bool(up_ui),
    })
    if portfolio_df is not None and ui_df is not None:
        # show potential scaling issues
        joined = ui_df.merge(portfolio_df[["isin", "qty_now"]], on="isin", how="inner", suffixes=("_ui", "_portfolio"))
        joined["ratio"] = joined["qty_now_ui"] / joined["qty_now_portfolio"]
        suspicious = joined[np.isclose(joined["ratio"], 10.0, atol=1e-6) | np.isclose(joined["ratio"], 0.1, atol=1e-6)]
        if len(suspicious):
            st.warning("Ik zie een mogelijke *x10* schaalverschil tussen UI export en Portfolio.csv voor sommige posities. Portfolio.csv wordt gebruikt als waarheid voor 'Aantal'.")
            st.dataframe(suspicious[["isin", "qty_now_ui", "qty_now_portfolio", "ratio"]], use_container_width=True, hide_index=True)
        else:
            st.success("Geen duidelijke x10-schaalverschillen gedetecteerd (op overlap).")

st.caption("v1.1: Voor een echte DEGIRO-match op 'Totale W/V' per positie is een UI/Tracky export ideaal, omdat dividend/tax FX-legs niet altijd per ISIN te alloceren zijn vanuit Account.csv alleen.")
