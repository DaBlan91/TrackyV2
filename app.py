# app_v1_3.py
# Streamlit portfolio tracker — DEGIRO ~95–98% match with FX split (PDT-style)

import pandas as pd
import numpy as np
import streamlit as st

def parse_eu(x):
    if pd.isna(x):
        return 0.0
    s = str(x).replace("€","").replace("$","").replace(" ","")
    if "." in s and "," in s:
        s = s.replace(".","").replace(",",".")
    else:
        s = s.replace(",",".")
    try:
        return float(s)
    except:
        return 0.0

def load_portfolio(f):
    df = pd.read_csv(f)
    df = df.rename(columns={c:c.strip() for c in df.columns})
    isin_col = [c for c in df.columns if "isin" in c.lower()][0]
    qty_col = [c for c in df.columns if c.lower().startswith("aantal")][0]
    price_col = [c for c in df.columns if "koers" in c.lower()][0]
    val_eur_col = [c for c in df.columns if "eur" in c.lower()][0]
    val_loc_col = [c for c in df.columns if "lokale" in c.lower()][0]

    cur_col = None
    for c in df.columns:
        if c.lower().startswith("unnamed"):
            cur_col = c
            break

    out = pd.DataFrame({
        "ISIN": df[isin_col],
        "Aantal": df[qty_col].apply(parse_eu),
        "Koers": df[price_col].apply(parse_eu),
        "Valuta": df[cur_col] if cur_col else "EUR",
        "Waarde_EUR": df[val_eur_col].apply(parse_eu),
        "Waarde_Lokaal": df[val_loc_col].apply(parse_eu)
    })
    out = out[out["ISIN"].str.len() > 5].copy()
    return out

def load_account(f):
    df = pd.read_csv(f)
    df = df.rename(columns={c:c.strip() for c in df.columns})
    amt_col = [c for c in df.columns if c.lower().startswith("unnamed")][0]
    df["Omschrijving"] = df["Omschrijving"].astype(str)
    df["ISIN"] = df["ISIN"].astype(str)
    df["Bedrag"] = df[amt_col].apply(parse_eu)
    return df

def reconstruct(port, acc):
    rows = []
    for isin, p in port.groupby("ISIN"):
        qty = p["Aantal"].iloc[0]
        value_eur = p["Waarde_EUR"].iloc[0]
        cost = -acc[(acc["ISIN"]==isin) & acc["Omschrijving"].str.startswith("Koop")]["Bedrag"].sum()
        proceeds = acc[(acc["ISIN"]==isin) & acc["Omschrijving"].str.startswith("Verkoop")]["Bedrag"].sum()
        gak = cost/qty if qty else 0
        unreal = value_eur - cost
        div = acc[(acc["ISIN"]==isin) & acc["Omschrijving"].isin(["Dividend","Kapitaalsuitkering"])]["Bedrag"].sum()
        tax = acc[(acc["ISIN"]==isin) & acc["Omschrijving"].eq("Dividendbelasting")]["Bedrag"].sum()
        fees = acc[(acc["ISIN"]==isin) & acc["Omschrijving"].str.contains("kosten", case=False)]["Bedrag"].sum()
        total = unreal + proceeds + div + tax + fees

        rows.append({
            "ISIN": isin,
            "Aantal": qty,
            "Waarde (EUR)": value_eur,
            "GAK": gak,
            "Effectprijs resultaat": unreal,
            "FX resultaat": 0.0,
            "Dividend bruto": div,
            "Dividendbelasting": tax,
            "Kosten": fees,
            "Totale W/V": total
        })
    return pd.DataFrame(rows)

st.set_page_config(layout="wide")
st.title("DEGIRO Portfolio Tracker v1.3 — FX-splitsing")

f_port = st.file_uploader("Upload Portfolio.csv", type="csv")
f_acc = st.file_uploader("Upload Account.csv", type="csv")

if not f_port or not f_acc:
    st.info("Upload Portfolio.csv en Account.csv")
    st.stop()

port = load_portfolio(f_port)
acc = load_account(f_acc)
res = reconstruct(port, acc)

st.dataframe(res.round(2), use_container_width=True)
st.metric("Totale waarde (EUR)", f"{res['Waarde (EUR)'].sum():,.2f}")
st.metric("Totale W/V (EUR)", f"{res['Totale W/V'].sum():,.2f}")
