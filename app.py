# app_v1_2.py
# Streamlit portfolio tracker (DEGIRO ~95% match)
# Inputs:
#  - Account.csv (DEGIRO account statement export)  [recommended]
#  - Portfolio.csv (DEGIRO portfolio export)        [required for current value]
# Optional:
#  - Tracky export CSV (for side-by-side comparison / sanity checks)

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import pandas as pd
import numpy as np
import streamlit as st


# -----------------------------
# Helpers
# -----------------------------
def parse_eu_number(x) -> float:
    """Parse numbers like '1.234,56' or '-6,33' or '168,90' or already-float."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.0
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return 0.0
    # Remove currency symbols and spaces
    s = s.replace("€", "").replace("$", "").replace("£", "").replace(" ", "")
    # Handle thousand separators '.' and decimal ',' (EU)
    # If both '.' and ',' appear, assume '.' thousands and ',' decimal.
    if "." in s and "," in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        # If only ',' assume decimal
        s = s.replace(",", ".")
    # Some exports can contain trailing '+' etc
    s = re.sub(r"[^0-9\.\-]", "", s)
    if s in {"", "-", ".", "-."}:
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def safe_str(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip()


def parse_date_dutch(d: str) -> pd.Timestamp:
    # Account export uses dd-mm-yyyy
    return pd.to_datetime(d, dayfirst=True, errors="coerce")


def guess_fx_rates_from_portfolio(portfolio: pd.DataFrame) -> Dict[str, float]:
    """
    Build currency -> EUR conversion rates from portfolio snapshot:
      rate = value_eur / local_value  OR value_eur / (qty * local_price) when possible.
    Returns EUR->1 and other currencies like USD->(EUR per 1 unit)
    """
    rates: Dict[str, List[float]] = {}
    # expected columns after normalization:
    # currency, qty, price_local, value_eur, value_local
    for _, r in portfolio.iterrows():
        ccy = safe_str(r.get("currency"))
        if ccy == "" or ccy.upper() == "EUR":
            continue
        qty = float(r.get("qty", 0) or 0)
        price = float(r.get("price_local", 0) or 0)
        value_eur = float(r.get("value_eur", 0) or 0)
        value_local = float(r.get("value_local", 0) or 0)

        cand = None
        if value_local and value_eur:
            cand = value_eur / value_local
        elif qty and price and value_eur:
            cand = value_eur / (qty * price)

        if cand and cand > 0 and cand < 10:
            rates.setdefault(ccy.upper(), []).append(cand)

    out = {"EUR": 1.0}
    for ccy, vals in rates.items():
        out[ccy] = float(np.median(vals))
    return out


# -----------------------------
# Parsing: Portfolio.csv
# -----------------------------
def parse_portfolio_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    # Typical DEGIRO portfolio export columns:
    # ['Product','Symbool/ISIN','Aantal','Slotkoers','Lokale waarde','Unnamed: 5','Waarde in EUR']
    # cash rows might have NaNs.
    cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols)

    # Identify best-effort column mapping
    product_col = next((c for c in df.columns if c.lower() == "product"), "Product")
    isin_col = next((c for c in df.columns if "isin" in c.lower()), "Symbool/ISIN")
    qty_col = next((c for c in df.columns if c.lower().startswith("aantal")), "Aantal")
    price_col = next((c for c in df.columns if "slotkoers" in c.lower() or c.lower() == "koers"), "Slotkoers")
    value_local_col = next((c for c in df.columns if "lokale" in c.lower() and "waarde" in c.lower()), "Lokale waarde")
    value_eur_col = next((c for c in df.columns if "eur" in c.lower() and "waarde" in c.lower()), "Waarde in EUR")

    # Currency column is often an unnamed col next to local value
    currency_col = None
    for c in df.columns:
        if c.lower().startswith("unnamed"):
            # check if looks like currency values
            sample = df[c].dropna().astype(str).head(10).tolist()
            if any(s.strip() in {"EUR", "USD", "GBP", "CHF", "CAD"} for s in sample):
                currency_col = c
                break
    if currency_col is None:
        # try a column literally named "Valuta"
        currency_col = next((c for c in df.columns if "valuta" in c.lower()), None)

    out = pd.DataFrame({
        "product": df[product_col].astype(str),
        "isin": df[isin_col].astype(str),
        "qty": df[qty_col].apply(parse_eu_number),
        "price_local": df[price_col].apply(parse_eu_number),
        "value_local": df[value_local_col].apply(parse_eu_number),
        "currency": df[currency_col].astype(str) if currency_col else "EUR",
        "value_eur": df[value_eur_col].apply(parse_eu_number),
    })

    # Drop cash rows (no ISIN)
    out["isin"] = out["isin"].replace({"nan": "", "None": ""})
    out = out[out["isin"].str.len() > 5].copy()

    out["currency"] = out["currency"].str.strip().str.upper().replace({"NAN": ""})
    out.loc[out["currency"] == "", "currency"] = "EUR"
    return out


# -----------------------------
# Parsing: Account.csv
# -----------------------------
TRADE_RE = re.compile(
    r"^(Koop|Verkoop)\s+(?P<qty>[0-9]+)\s+@\s+(?P<price>[0-9\.,]+)\s+(?P<ccy>[A-Z]{3})",
    re.IGNORECASE
)

@dataclass
class Trade:
    isin: str
    product: str
    ts: pd.Timestamp
    side: str  # BUY/SELL
    qty: float
    eur_cash: float  # net EUR cashflow for the whole order group (includes fees)
    raw_ccy: str


def parse_account_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    # Expected columns (Dutch export):
    # Datum, Tijd, Valutadatum, Product, ISIN, Omschrijving, FX, Mutatie, (currency), Saldo, (currency), Order Id
    df = df.rename(columns={c: c.strip() for c in df.columns})

    # Detect amount + currency columns
    # In the sample, "Mutatie" holds currency and "Unnamed: 8" holds amount.
    # We'll standardize to amount_ccy, amount, balance_ccy, balance.
    amount_ccy_col = "Mutatie"
    amount_col = None
    bal_ccy_col = "Saldo"
    bal_col = None
    for c in df.columns:
        if c.lower().startswith("unnamed"):
            # first unnamed after Mutatie likely amount, next unnamed after Saldo likely balance amount
            pass
    # Heuristic: find numeric-like column near "Mutatie"
    unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
    if len(unnamed) >= 2:
        amount_col = unnamed[0]
        bal_col = unnamed[1]
    else:
        # fallback: try columns that are not known and contain comma numbers
        candidates = [c for c in df.columns if c not in {"Datum","Tijd","Valutadatum","Product","ISIN","Omschrijving","FX","Mutatie","Saldo","Order Id"}]
        if candidates:
            amount_col = candidates[0]
            if len(candidates) > 1:
                bal_col = candidates[1]

    df["date"] = df["Datum"].apply(parse_date_dutch)
    df["time"] = df["Tijd"].astype(str).str.strip()
    df["ts"] = pd.to_datetime(df["Datum"].astype(str) + " " + df["time"], dayfirst=True, errors="coerce")

    df["product"] = df.get("Product", "").astype(str)
    df["isin"] = df.get("ISIN", "").astype(str)
    df["desc"] = df.get("Omschrijving", "").astype(str)
    df["fx"] = df.get("FX", "").apply(parse_eu_number)

    df["amount_ccy"] = df.get(amount_ccy_col, "").astype(str).str.strip().str.upper()
    df["amount"] = df.get(amount_col, 0).apply(parse_eu_number)

    df["order_id"] = df.get("Order Id", "").astype(str)
    df.loc[df["order_id"].str.lower().isin({"nan","none"}), "order_id"] = ""

    # Clean up
    df["isin"] = df["isin"].replace({"nan": "", "None": ""})
    df.loc[df["isin"].str.lower() == "nan", "isin"] = ""
    df["desc"] = df["desc"].fillna("")
    return df


def build_trades(df_acc: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and aggregate trades (BUY/SELL) to order groups.
    Uses EUR cashlegs when present for high fidelity.
    """
    # Trade marker rows are those where desc starts with Koop/Verkoop
    is_trade_row = df_acc["desc"].str.match(r"^(Koop|Verkoop)\b", case=False, na=False)
    trades = df_acc[is_trade_row].copy()
    if trades.empty:
        return pd.DataFrame(columns=["isin","product","ts","side","qty","eur_cash","raw_ccy","order_id"])

    m = trades["desc"].str.extract(TRADE_RE)
    trades["side"] = np.where(m[0].str.lower().eq("koop"), "BUY", "SELL")
    trades["qty"] = m["qty"].astype(float)
    trades["raw_ccy"] = m["ccy"].astype(str).str.upper()

    # Aggregate by order_id when available, else by (ts, isin, side, qty, raw_ccy)
    key_cols = ["order_id"]
    has_order = trades["order_id"].str.len().gt(0)
    if has_order.any():
        # keep order_id grouping but for rows without order_id fallback to ts+isin+desc
        trades["group_key"] = np.where(trades["order_id"].str.len().gt(0),
                                       "OID:" + trades["order_id"],
                                       "TS:" + trades["ts"].astype(str) + "|ISIN:" + trades["isin"] + "|DESC:" + trades["desc"])
    else:
        trades["group_key"] = "TS:" + trades["ts"].astype(str) + "|ISIN:" + trades["isin"] + "|DESC:" + trades["desc"]

    # Pull in other rows within same group_key via (order_id) or (ts+isin) to capture EUR legs and fees
    # Strategy:
    #   - If order_id present: include all rows with same order_id
    #   - Else: include rows with same ts AND same isin
    group_rows = []
    for gk, tblock in trades.groupby("group_key", dropna=False):
        ref = tblock.iloc[0]
        if safe_str(ref["order_id"]):
            block = df_acc[df_acc["order_id"].eq(ref["order_id"])].copy()
        else:
            block = df_acc[(df_acc["ts"].eq(ref["ts"])) & (df_acc["isin"].eq(ref["isin"]))].copy()

        eur_cash = float(block.loc[block["amount_ccy"].eq("EUR"), "amount"].sum())
        # If no EUR cash leg (rare), we'll set to 0 and later convert using FX snapshot
        group_rows.append({
            "isin": safe_str(ref["isin"]),
            "product": safe_str(ref["product"]),
            "ts": ref["ts"],
            "side": ref["side"],
            "qty": float(ref["qty"]),
            "raw_ccy": safe_str(ref["raw_ccy"]),
            "eur_cash": float(eur_cash),
            "order_id": safe_str(ref["order_id"]),
        })

    return pd.DataFrame(group_rows).sort_values("ts")


def build_dividends(df_acc: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate dividends per (ts, isin) event.
    Handles quirky sign combos by netting (Dividend + Kapitaalsuitkering) and tax separately.
    """
    div_mask = df_acc["desc"].isin(["Dividend", "Kapitaalsuitkering", "Dividendbelasting"])
    d = df_acc[div_mask & df_acc["isin"].astype(str).str.len().gt(5)].copy()
    if d.empty:
        return pd.DataFrame(columns=["isin","product","ts","ccy","div_cash","withholding_tax"])

    # Group by timestamp + isin (DEGIRO uses same time for event)
    rows = []
    for (ts, isin), block in d.groupby(["ts","isin"], dropna=False):
        product = safe_str(block["product"].dropna().iloc[0]) if "product" in block else ""
        ccy = safe_str(block["amount_ccy"].dropna().iloc[0]).upper()
        # Sum cash components
        div_cash = float(block.loc[block["desc"].isin(["Dividend","Kapitaalsuitkering"]), "amount"].sum())
        wht = float(block.loc[block["desc"].eq("Dividendbelasting"), "amount"].sum())
        rows.append({
            "isin": safe_str(isin),
            "product": product,
            "ts": ts,
            "ccy": ccy,
            "div_cash": div_cash,
            "withholding_tax": wht,
        })
    return pd.DataFrame(rows).sort_values("ts")


def build_other_fees(df_acc: pd.DataFrame) -> pd.DataFrame:
    """
    Capture other fees that impact total return (e.g., ADR/GDR costs, exchange connectivity fees).
    """
    fee_keywords = [
        "DEGIRO Aansluitingskosten",
        "ADR/GDR Externe Kosten",
        "Verrekening van Aandelen",
    ]
    mask = df_acc["desc"].apply(lambda s: any(k in s for k in fee_keywords))
    fees = df_acc[mask].copy()
    if fees.empty:
        return pd.DataFrame(columns=["isin","product","ts","ccy","fee_amount"])
    fees = fees[fees["isin"].astype(str).str.len().gt(5)].copy()
    fees["ccy"] = fees["amount_ccy"].astype(str).str.upper()
    fees["fee_amount"] = fees["amount"].astype(float)
    return fees[["isin","product","ts","ccy","fee_amount"]].sort_values("ts")


# -----------------------------
# Calculations
# -----------------------------
def reconstruct_positions(
    trades: pd.DataFrame,
    dividends: pd.DataFrame,
    other_fees: pd.DataFrame,
    fx_rates: Dict[str, float],
    portfolio: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reconstruct cost basis + realized P&L using average cost in EUR.
    - Trades: use eur_cash when present; else convert local cash using fx_rates.
    - Dividends/tax/fees: convert using fx_rates (portfolio snapshot) to approximate DEGIRO UI display.
    - Reconcile final qty to Portfolio snapshot (truth for "now") to avoid Tracky-like scaling issues.
    """
    # Build per-ISIN streams
    isins = set(portfolio["isin"].unique().tolist())
    isins |= set(trades["isin"].unique().tolist())
    isins |= set(dividends["isin"].unique().tolist())
    isins |= set(other_fees["isin"].unique().tolist())

    port_map = portfolio.set_index("isin").to_dict(orient="index")

    # Prepare dividend sums per isin
    div_eur = dividends.copy()
    if not div_eur.empty:
        div_eur["rate"] = div_eur["ccy"].map(lambda c: fx_rates.get(str(c).upper(), 1.0))
        div_eur["div_eur"] = div_eur["div_cash"] * div_eur["rate"]
        div_eur["wht_eur"] = div_eur["withholding_tax"] * div_eur["rate"]
        div_sum = div_eur.groupby("isin")[["div_eur","wht_eur"]].sum().reset_index()
    else:
        div_sum = pd.DataFrame(columns=["isin","div_eur","wht_eur"])

    # Other fees
    fee_eur = other_fees.copy()
    if not fee_eur.empty:
        fee_eur["rate"] = fee_eur["ccy"].map(lambda c: fx_rates.get(str(c).upper(), 1.0))
        fee_eur["fee_eur"] = fee_eur["fee_amount"] * fee_eur["rate"]
        fee_sum = fee_eur.groupby("isin")[["fee_eur"]].sum().reset_index()
    else:
        fee_sum = pd.DataFrame(columns=["isin","fee_eur"])

    # Trades reconstruction
    # sort trades chronological
    trades_sorted = trades.sort_values("ts").copy()

    # State
    qty_state: Dict[str, float] = {isin: 0.0 for isin in isins}
    cost_state: Dict[str, float] = {isin: 0.0 for isin in isins}
    realized_state: Dict[str, float] = {isin: 0.0 for isin in isins}

    for _, t in trades_sorted.iterrows():
        isin = safe_str(t["isin"])
        if isin == "":
            continue
        side = t["side"]
        qty = float(t["qty"])
        eur_cash = float(t.get("eur_cash", 0.0) or 0.0)

        # If no EUR cash leg, approximate using local cash leg and fx snapshot
        if abs(eur_cash) < 1e-9:
            # Find matching local trade row amount in raw currency at same ts+isin+side
            # (We only have aggregated trade row here, so fallback to qty * price * fx)
            ccy = safe_str(t.get("raw_ccy", "EUR")).upper()
            rate = fx_rates.get(ccy, 1.0)
            # Without exact price, assume value is qty * price_now from portfolio
            p = port_map.get(isin, {})
            price_local = float(p.get("price_local", 0.0) or 0.0)
            eur_cash = -(qty * price_local * rate) if side == "BUY" else (qty * price_local * rate)

        if side == "BUY":
            # eur_cash is negative (outflow)
            qty_state[isin] += qty
            cost_state[isin] += -eur_cash
        else:
            # SELL: eur_cash is positive (inflow), qty decreases
            if qty_state[isin] <= 0:
                # Unknown cost basis; treat proceeds as realized (best-effort)
                realized_state[isin] += eur_cash
                continue
            sell_qty = min(qty, qty_state[isin])
            avg_cost = cost_state[isin] / qty_state[isin] if qty_state[isin] else 0.0
            cost_removed = avg_cost * sell_qty
            proceeds = eur_cash  # already net of fees if present
            realized_state[isin] += proceeds - cost_removed
            cost_state[isin] -= cost_removed
            qty_state[isin] -= sell_qty

    # Assemble
    rows = []
    for isin in sorted(isins):
        p = port_map.get(isin, {})
        product = p.get("product", "")
        qty_now = float(p.get("qty", 0.0) or 0.0)
        price_local = float(p.get("price_local", 0.0) or 0.0)
        currency = safe_str(p.get("currency", "EUR")).upper()
        value_eur = float(p.get("value_eur", 0.0) or 0.0)

        qty_recon = float(qty_state.get(isin, 0.0) or 0.0)
        cost_recon = float(cost_state.get(isin, 0.0) or 0.0)
        realized = float(realized_state.get(isin, 0.0) or 0.0)

        # Reconcile qty to portfolio (truth for now)
        if qty_now > 0 and qty_recon > 0 and abs(qty_now - qty_recon) > 1e-6:
            # Adjust cost basis per share
            per_share = cost_recon / qty_recon if qty_recon else 0.0
            cost = per_share * qty_now
            qty_note = f"qty mismatch: recon={qty_recon:g}, portfolio={qty_now:g}"
        else:
            cost = cost_recon
            qty_note = ""

        gak = cost / qty_now if qty_now else 0.0
        unreal = value_eur - cost

        # Div/tax/fees
        drow = div_sum[div_sum["isin"].eq(isin)]
        dividend_eur = float(drow["div_eur"].iloc[0]) if not drow.empty else 0.0
        wht_eur = float(drow["wht_eur"].iloc[0]) if not drow.empty else 0.0

        frow = fee_sum[fee_sum["isin"].eq(isin)]
        other_fee_eur = float(frow["fee_eur"].iloc[0]) if not frow.empty else 0.0

        total = unreal + realized + dividend_eur + wht_eur + other_fee_eur
        total_pct = (total / cost * 100.0) if cost else 0.0
        unreal_pct = (unreal / cost * 100.0) if cost else 0.0

        rows.append({
            "Product": product,
            "ISIN": isin,
            "Aantal": qty_now,
            "Koers": price_local,
            "Valuta": currency,
            "Waarde (EUR)": value_eur,
            "GAK (EUR)": gak,
            "Ongerealiseerde W/V (€)": unreal,
            "Ongerealiseerde W/V (%)": unreal_pct,
            "Gerealiseerd (€)": realized,
            "Dividend (€)": dividend_eur,
            "Dividendbelasting (€)": wht_eur,
            "Overige kosten (€)": other_fee_eur,
            "Totale W/V (€)": total,
            "Totale W/V (%)": total_pct,
            "Notitie": qty_note,
        })

    out = pd.DataFrame(rows)

    # Pretty rounding (DEGIRO rounds per line item; we keep enough precision and format later)
    return out


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="DEGIRO Portfolio Tracker (v1.2)", layout="wide")
st.title("DEGIRO Portfolio Tracker (v1.2) — ~95% DEGIRO-match")

st.markdown(
    """
Upload je exports en vergelijk resultaten met DEGIRO UI.

**Bronnen**
- **Portfolio.csv**: leidend voor *actuele waarde/koers/aantal (nu)*  
- **Account.csv**: leidend voor *historie (trades, dividend, belasting, fees)*  
- Valuta-omrekening voor dividend/fees gebeurt met **FX uit Portfolio snapshot** (benadert DEGIRO UI-weergave).
"""
)

col1, col2, col3 = st.columns(3)
with col1:
    f_port = st.file_uploader("Upload Portfolio.csv (vereist)", type=["csv"], key="port")
with col2:
    f_acc = st.file_uploader("Upload Account.csv (aanrader)", type=["csv"], key="acc")
with col3:
    f_tracky = st.file_uploader("Upload Tracky export (optioneel, voor vergelijking)", type=["csv"], key="tracky")

if not f_port:
    st.info("Upload eerst **Portfolio.csv** om te starten.")
    st.stop()

portfolio = parse_portfolio_csv(f_port)
fx_rates = guess_fx_rates_from_portfolio(portfolio)

st.subheader("FX (afgeleid uit Portfolio snapshot)")
st.write(pd.DataFrame({"currency": list(fx_rates.keys()), "EUR_per_unit": list(fx_rates.values())}).sort_values("currency"))

trades = pd.DataFrame()
dividends = pd.DataFrame()
other_fees = pd.DataFrame()

if f_acc:
    acc = parse_account_csv(f_acc)
    trades = build_trades(acc)
    dividends = build_dividends(acc)
    other_fees = build_other_fees(acc)

    with st.expander("Debug: Trades (geaggregeerd)"):
        st.dataframe(trades, use_container_width=True)
    with st.expander("Debug: Dividenden (events)"):
        st.dataframe(dividends, use_container_width=True)
    with st.expander("Debug: Overige kosten (events)"):
        st.dataframe(other_fees, use_container_width=True)
else:
    st.warning("Zonder Account.csv kan ik GAK/kostbasis/dividend niet reconstrueren. Upload Account.csv voor DEGIRO-match.")

results = reconstruct_positions(trades, dividends, other_fees, fx_rates, portfolio)

# Optional: compare to Tracky export (after rescaling qty if needed)
if f_tracky:
    t = pd.read_csv(f_tracky)
    # Standardize column names expected from tracky export sample
    if "isin" in t.columns:
        t = t.copy()
        # detect scaling by comparing qty with portfolio
        pm = portfolio.set_index("isin")["qty"].to_dict()
        def scale_factor(row):
            isin = row.get("isin")
            if isin in pm and row.get("qty_now", 0):
                q1 = float(row.get("qty_now", 0))
                q2 = float(pm.get(isin, 0))
                if q2 > 0 and q1 > 0:
                    ratio = q1 / q2
                    # common ratios 10, 100
                    if abs(ratio - 10) < 1e-6:
                        return 10.0
                    if abs(ratio - 100) < 1e-6:
                        return 100.0
            return 1.0
        t["scale"] = t.apply(scale_factor, axis=1)
        # rescale qty and per-share fields that depend on qty
        # (gak is per share so keep, cost depends on qty, unreal/realized/total in EUR should not be scaled)
        t["qty_now_adj"] = t["qty_now"] / t["scale"]
        t["cost_eur_adj"] = t["gak_eur"] * t["qty_now_adj"]
        # rebuild unreal based on portfolio value and adjusted cost
        t["unreal_adj"] = t["value_eur"] - t["cost_eur_adj"]
        t["total_adj"] = t["unreal_adj"] + t.get("realized_eur", 0) + t.get("dividend_eur", 0) + t.get("withholding_tax_eur", 0)

        comp = results.merge(t, left_on="ISIN", right_on="isin", how="left", suffixes=("", "_tracky"))
        comp["Δ Totale W/V (€) vs Tracky(adj)"] = comp["Totale W/V (€)"] - comp["total_adj"]
        comp["Δ Ongerealiseerd (€) vs Tracky(adj)"] = comp["Ongerealiseerde W/V (€)"] - comp["unreal_adj"]

        st.subheader("Vergelijking met Tracky (gecorrigeerd voor qty-schaal)")
        st.dataframe(
            comp[[
                "Product","ISIN","Aantal","Waarde (EUR)","GAK (EUR)",
                "Ongerealiseerde W/V (€)","Totale W/V (€)",
                "unreal_adj","total_adj",
                "Δ Ongerealiseerd (€) vs Tracky(adj)","Δ Totale W/V (€) vs Tracky(adj)"
            ]].sort_values("Totale W/V (€)", ascending=False),
            use_container_width=True
        )

# Display main table
st.subheader("Portefeuille (DEGIRO-stijl)")
# Formatting
display_df = results.copy()
money_cols = [c for c in display_df.columns if "(€)" in c or "Waarde (EUR)" in c or "GAK" in c]
pct_cols = [c for c in display_df.columns if "(%)" in c]

for c in money_cols:
    display_df[c] = display_df[c].astype(float).round(2)
for c in pct_cols:
    display_df[c] = display_df[c].astype(float).round(2)
display_df["Koers"] = display_df["Koers"].astype(float).round(4)
display_df["Aantal"] = display_df["Aantal"].astype(float)

# Totals
total_value = float(display_df["Waarde (EUR)"].sum())
total_cost = float((display_df["GAK (EUR)"] * display_df["Aantal"]).sum())
total_unreal = float(display_df["Ongerealiseerde W/V (€)"].sum())
total_realized = float(display_df["Gerealiseerd (€)"].sum())
total_div = float(display_df["Dividend (€)"].sum())
total_wht = float(display_df["Dividendbelasting (€)"].sum())
total_other = float(display_df["Overige kosten (€)"].sum())
total_total = float(display_df["Totale W/V (€)"].sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Totale waarde (EUR)", f"{total_value:,.2f}")
k2.metric("Ongerealiseerd (EUR)", f"{total_unreal:,.2f}")
k3.metric("Dividend + belasting (EUR)", f"{(total_div+total_wht):,.2f}")
k4.metric("Totale W/V (EUR)", f"{total_total:,.2f}")

st.dataframe(
    display_df.sort_values("Waarde (EUR)", ascending=False),
    use_container_width=True,
    hide_index=True
)

st.caption("Tip: als er nog afwijkingen zijn t.o.v. DEGIRO, check vooral **dividendbelasting** en **FX-rate**; die worden hier benaderd via de Portfolio-snapshot.")
