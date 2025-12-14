
import re
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import pandas as pd
import streamlit as st

st.set_page_config(page_title="DEGIRO Portfolio Tracker (v1)", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
ISIN_RE = re.compile(r"\b[A-Z]{2}[A-Z0-9]{10}\b")

def parse_eu_number(x) -> float:
    """Parse numbers like '1.234,56' or '-3,00' or '0,00' or NaN into float."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return 0.0
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return 0.0
    # Remove thousands separators and convert comma decimal
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return 0.0

def extract_isin(val: Optional[str]) -> Optional[str]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    m = ISIN_RE.search(str(val))
    return m.group(0) if m else None

def parse_trade_desc(desc: str) -> Optional[Tuple[str, float, Optional[float], Optional[str]]]:
    """
    Parse 'Koop 3 @ 111,72 EUR' or 'Verkoop 150 @ 2,08 USD'
    Returns (side, qty, price, ccy)
    """
    if not isinstance(desc, str):
        return None
    desc = desc.strip()
    m = re.match(r"^(Koop|Verkoop)\s+([0-9]+(?:[.,][0-9]+)?)\s+@\s+([0-9]+(?:[.,][0-9]+)?)\s+([A-Z]{3})", desc)
    if not m:
        return None
    side = "BUY" if m.group(1) == "Koop" else "SELL"
    qty = parse_eu_number(m.group(2))
    price = parse_eu_number(m.group(3))
    ccy = m.group(4)
    return side, qty, price, ccy

def safe_to_datetime(d: str, t: str) -> pd.Timestamp:
    # DEGIRO export: 'DD-MM-YYYY' + 'HH:MM'
    try:
        return pd.to_datetime(f"{d} {t}", format="%d-%m-%Y %H:%M")
    except Exception:
        try:
            return pd.to_datetime(d, dayfirst=True)
        except Exception:
            return pd.NaT

# -----------------------------
# Parsing DEGIRO exports
# -----------------------------
def load_account_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)

    # Normalize expected columns (DEGIRO NL exports often have these):
    expected = {"Datum","Tijd","Product","ISIN","Omschrijving","Mutatie","Order Id"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Account.csv mist kolommen: {', '.join(sorted(missing))}")

    # Amount and currency columns are weird in DEGIRO exports:
    # 'Mutatie' usually contains the currency code, and the amount is in the next unnamed column.
    # We'll detect the first 'Unnamed' column after Mutatie; in your sample it's 'Unnamed: 8'.
    unnamed_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    # Prefer the first unnamed column as amount column
    amount_col = unnamed_cols[0] if unnamed_cols else None
    if amount_col is None:
        raise ValueError("Kan mutatie-bedragen niet vinden (geen 'Unnamed' kolom).")

    df = df.rename(columns={
        "Mutatie": "mut_ccy",
        amount_col: "mut_amount_raw",
        "Order Id": "order_id",
        "Omschrijving": "description",
        "Product": "product",
        "ISIN": "isin",
        "Datum": "date",
        "Tijd": "time"
    })

    df["mut_amount"] = df["mut_amount_raw"].apply(parse_eu_number)
    df["isin"] = df["isin"].apply(extract_isin)
    df["ts"] = [safe_to_datetime(d, t) for d, t in zip(df["date"].astype(str), df["time"].astype(str))]
    return df

def load_portfolio_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)

    # Expected columns from your sample:
    # Product, Symbool/ISIN, Aantal, Slotkoers, Waarde in EUR
    required_any = {"Product", "Symbool/ISIN", "Aantal"}
    if not required_any.issubset(df.columns):
        raise ValueError("Portfolio.csv mist minimaal: Product, Symbool/ISIN, Aantal")

    # Identify EUR value column
    eur_value_col = None
    for c in df.columns:
        if str(c).strip().lower() in ["waarde in eur", "waarde in EUR".lower()]:
            eur_value_col = c
            break
    if eur_value_col is None and "Waarde in EUR" in df.columns:
        eur_value_col = "Waarde in EUR"
    if eur_value_col is None:
        # Try fuzzy
        for c in df.columns:
            if "eur" in str(c).lower() and "waarde" in str(c).lower():
                eur_value_col = c
                break
    if eur_value_col is None:
        raise ValueError("Portfolio.csv: kan 'Waarde in EUR' kolom niet vinden.")

    # Identify price column (optional)
    price_col = None
    for c in df.columns:
        if str(c).strip().lower() in ["slotkoers", "laatste koers", "koers", "price", "last price"]:
            price_col = c
            break

    df = df.rename(columns={
        "Product": "product",
        "Symbool/ISIN": "symbol_isin",
        "Aantal": "qty_now_raw",
        eur_value_col: "value_eur_raw",
    })
    if price_col:
        df = df.rename(columns={price_col: "price_now_raw"})
    else:
        df["price_now_raw"] = None

    df["isin"] = df["symbol_isin"].apply(extract_isin)
    df["qty_now"] = df["qty_now_raw"].apply(parse_eu_number)
    df["value_eur"] = df["value_eur_raw"].apply(parse_eu_number)
    df["price_now"] = df["price_now_raw"].apply(parse_eu_number) if "price_now_raw" in df.columns else 0.0

    # Keep only rows that look like securities (exclude CASH lines without ISIN)
    df = df[df["isin"].notna()].copy()
    return df[["isin","product","qty_now","price_now","value_eur"]]

# -----------------------------
# Core calculations
# -----------------------------
@dataclass
class PositionState:
    qty: float = 0.0
    cost_eur: float = 0.0  # open cost basis in EUR
    realized_eur: float = 0.0

def build_orders(account_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build order-level table:
    - side BUY/SELL
    - qty
    - eur_cash (net cash movement in EUR within the same order_id)
    """
    df = account_df.copy()
    df = df[df["order_id"].notna()].copy()

    if df.empty:
        return pd.DataFrame(columns=["ts","isin","product","side","qty","eur_cash"])

    orders = []
    for oid, g in df.groupby("order_id", dropna=True):
        # Identify instrument
        isin = g["isin"].dropna().astype(str).iloc[0] if g["isin"].notna().any() else None
        product = g["product"].dropna().astype(str).iloc[0] if g["product"].notna().any() else None

        # Identify trade rows (Koop/Verkoop) - sometimes multiple fills under same order id
        trades = []
        for desc in g["description"].dropna().astype(str).tolist():
            parsed = parse_trade_desc(desc)
            if parsed:
                trades.append(parsed)
        if not trades:
            continue

        # Ensure consistent side
        sides = {t[0] for t in trades}
        if len(sides) != 1:
            # Mixed side in one order id is unusual; skip safely
            continue
        side = list(sides)[0]
        qty = sum(t[1] for t in trades)

        # Net EUR cash movement within order
        eur_cash = g.loc[g["mut_ccy"].astype(str).str.upper() == "EUR", "mut_amount"].sum()

        # Timestamp: earliest timestamp in group
        ts = g["ts"].dropna().min() if g["ts"].notna().any() else pd.NaT

        orders.append({
            "order_id": oid,
            "ts": ts,
            "isin": isin,
            "product": product,
            "side": side,
            "qty": float(qty),
            "eur_cash": float(eur_cash)
        })

    odf = pd.DataFrame(orders)
    if not odf.empty:
        odf = odf.sort_values("ts")
    return odf

def compute_positions_from_orders(order_df: pd.DataFrame) -> pd.DataFrame:
    """
    Average-cost method:
    - BUY: increase qty; increase cost by -eur_cash (eur_cash is negative outflow)
    - SELL: decrease qty; realized += proceeds - cost_removed; cost -= cost_removed
      proceeds = eur_cash (positive inflow, net of fees if fees are in EUR within order)
    """
    states: Dict[str, PositionState] = {}
    for _, r in order_df.iterrows():
        isin = r["isin"]
        if not isin or pd.isna(isin):
            continue
        side = r["side"]
        qty = float(r["qty"] or 0.0)
        eur_cash = float(r["eur_cash"] or 0.0)

        stt = states.get(isin, PositionState())

        if side == "BUY":
            # Cash outflow (usually negative). Cost increase is positive.
            cost_increase = -eur_cash
            if cost_increase < 0:
                # If EUR cash is unexpectedly positive, ignore rather than corrupt.
                cost_increase = abs(cost_increase)
            stt.qty += qty
            stt.cost_eur += cost_increase

        elif side == "SELL":
            if stt.qty <= 0:
                # Nothing to sell against (data issue); skip
                continue
            sell_qty = min(qty, stt.qty)
            avg_cost = stt.cost_eur / stt.qty if stt.qty != 0 else 0.0
            cost_removed = avg_cost * sell_qty
            proceeds = eur_cash  # already net in EUR within this order
            stt.qty -= sell_qty
            stt.cost_eur -= cost_removed
            stt.realized_eur += (proceeds - cost_removed)

        states[isin] = stt

    rows = []
    for isin, stt in states.items():
        gak = stt.cost_eur / stt.qty if stt.qty else 0.0
        rows.append({
            "isin": isin,
            "qty_calc": stt.qty,
            "cost_eur": stt.cost_eur,
            "gak_eur": gak,
            "realized_eur": stt.realized_eur
        })
    return pd.DataFrame(rows)

def compute_dividends_eur(account_df: pd.DataFrame) -> pd.DataFrame:
    """
    v1: dividend EUR only (reliable without FX matching).
    Includes:
    - Dividend
    - Kapitaalsuitkering
    Excludes dividendbelasting (withholding) as separate column.
    """
    df = account_df.copy()
    df["desc"] = df["description"].astype(str)

    dividend_mask = df["desc"].isin(["Dividend", "Kapitaalsuitkering"])
    tax_mask = df["desc"].isin(["Dividendbelasting"])

    div = df[dividend_mask & (df["mut_ccy"].astype(str).str.upper() == "EUR") & df["isin"].notna()]
    tax = df[tax_mask & (df["mut_ccy"].astype(str).str.upper() == "EUR") & df["isin"].notna()]

    div_agg = div.groupby("isin")["mut_amount"].sum().rename("dividend_eur").reset_index()
    tax_agg = tax.groupby("isin")["mut_amount"].sum().rename("withholding_tax_eur").reset_index()

    out = pd.merge(div_agg, tax_agg, on="isin", how="outer").fillna(0.0)
    return out

# -----------------------------
# UI
# -----------------------------
st.title("DEGIRO Portfolio Tracker (v1)")
st.caption("Upload je **Account.csv** (transacties) + **Portfolio.csv** (actuele posities/waarde). Geen API nodig.")

with st.sidebar:
    st.header("Uploads")
    account_file = st.file_uploader("Account.csv (transacties)", type=["csv"])
    portfolio_file = st.file_uploader("Portfolio.csv (actuele posities)", type=["csv"])

    st.divider()
    st.subheader("Opties")
    show_debug = st.checkbox("Toon debug / checks", value=False)

if not account_file or not portfolio_file:
    st.info("Upload beide bestanden om te starten.")
    st.stop()

# Load
try:
    acc = load_account_csv(account_file)
    port = load_portfolio_csv(portfolio_file)
except Exception as e:
    st.error(f"Kan bestanden niet inlezen: {e}")
    st.stop()

# Build orders and compute
orders = build_orders(acc)
pos_calc = compute_positions_from_orders(orders)
divs = compute_dividends_eur(acc)

# Merge with portfolio (current snapshot is leading for 'now')
df = port.merge(pos_calc, on="isin", how="left").merge(divs, on="isin", how="left")
df[["cost_eur","gak_eur","realized_eur","dividend_eur","withholding_tax_eur"]] = df[["cost_eur","gak_eur","realized_eur","dividend_eur","withholding_tax_eur"]].fillna(0.0)

# Business rule: current qty/value come from Portfolio.csv
df["unrealized_eur"] = df["value_eur"] - df["cost_eur"]
df["unrealized_pct"] = df.apply(lambda r: (r["unrealized_eur"]/r["cost_eur"]*100.0) if r["cost_eur"] else 0.0, axis=1)

df["total_return_eur"] = df["unrealized_eur"] + df["realized_eur"] + df["dividend_eur"] + df["withholding_tax_eur"]
df["total_return_pct"] = df.apply(lambda r: (r["total_return_eur"]/r["cost_eur"]*100.0) if r["cost_eur"] else 0.0, axis=1)

# Summary metrics
total_value = float(df["value_eur"].sum())
total_cost = float(df["cost_eur"].sum())
total_unreal = float(df["unrealized_eur"].sum())
total_realized = float(df["realized_eur"].sum())
total_div = float(df["dividend_eur"].sum() + df["withholding_tax_eur"].sum())
total_return = float(total_unreal + total_realized + total_div)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Totale waarde (EUR)", f"€ {total_value:,.2f}")
col2.metric("Kostbasis open posities (EUR)", f"€ {total_cost:,.2f}")
col3.metric("Ongerealiseerd resultaat (EUR)", f"€ {total_unreal:,.2f}", delta=f"{(total_unreal/total_cost*100.0 if total_cost else 0.0):.2f}%")
col4.metric("Totaal rendement (EUR)", f"€ {total_return:,.2f}")

# Table
st.subheader("Posities")
display_cols = [
    "product","isin","qty_now","price_now","value_eur",
    "gak_eur","cost_eur","unrealized_eur","unrealized_pct",
    "realized_eur","dividend_eur","withholding_tax_eur",
    "total_return_eur","total_return_pct"
]
pretty = df.copy()
pretty = pretty[display_cols].sort_values("value_eur", ascending=False)

# Format
st.dataframe(pretty, use_container_width=True, hide_index=True)

# Downloads
csv_out = pretty.to_csv(index=False).encode("utf-8")
st.download_button("Download posities (CSV)", data=csv_out, file_name="positions_calculated.csv", mime="text/csv")

if show_debug:
    st.subheader("Debug")
    st.write("Account.csv rows:", len(acc), "Portfolio.csv rows:", len(port))
    st.write("Order rows parsed:", len(orders))
    st.dataframe(orders.head(50), use_container_width=True)
    # Show instruments where qty mismatch between calc and portfolio
    merged = df.copy()
    merged["qty_diff"] = merged["qty_now"] - merged["qty_calc"].fillna(0.0)
    mism = merged[merged["qty_diff"].abs() > 1e-9][["product","isin","qty_now","qty_calc","qty_diff"]].sort_values("qty_diff", key=lambda s: s.abs(), ascending=False)
    st.write("Qty verschillen (Portfolio vs berekend uit orders):")
    st.dataframe(mism, use_container_width=True, hide_index=True)

st.caption("v1: gebruikt **Portfolio.csv** als waarheid voor actuele waarde/aantal. Kostbasis & GAK worden berekend uit orderregels in Account.csv via netto EUR cashflow per Order Id.")
