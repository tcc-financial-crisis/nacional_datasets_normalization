#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Normalizacao dos datasets de politica economica brasileira.

Estrutura de pastas:
  raw/         -> dados originais (fonte; nao sao alterados)
  normalized/  -> versoes normalizadas/limpas de cada serie
  ./           -> merges na raiz do projeto

Saidas:
  normalized/ipca.csv, selic.csv, producao_industrial.csv  (JSON -> CSV colunar: data, valor)
  normalized/spread_credito.csv                             (CSV BR -> CSV colunar padronizado)
  normalized/indice_volatilidade_limpo.csv                  (CSV diario limpo: data, valor)
  normalized/curva_juros_normalizado.csv                    (snapshot 1 dia -> formato tidy/longo)
  normalized/curva_juros_slope.csv                          (inclinacao diaria 10A-2A; Tesouro Direto)
  ./dataset_mensal_merged.csv                               (painel mensal final, colunar, por intersecao)

Entrada externa (baixada por baixar_tesouro_direto.py):
  raw/tesouro_direto_prefixado.csv  -> LTN + NTN-F (Tesouro Transparente)

Convencoes do painel final:
  - separador ','  | decimal '.'  | coluna 'data' no formato YYYY-MM
  - volatilidade diaria -> mensal usando o ULTIMO dia disponivel do mes
  - merge por INTERSECAO (so meses em que TODAS as series tem valor)
"""
import json
import csv
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
RAW = BASE / "raw"           # dados originais
NORM = BASE / "normalized"   # normalizacoes
NORM.mkdir(exist_ok=True)


def br_num(s):
    """Converte numero em formato BR ('1.008', '8,7774', '2,55E-03') para float."""
    if s is None:
        return None
    s = s.strip().strip('"')
    if s == "":
        return None
    # ponto = separador de milhar; virgula = decimal
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# 1) JSON (IPEADATA OData) -> CSV colunar
# ---------------------------------------------------------------------------
def json_para_csv(nome_json, nome_csv):
    with open(RAW / nome_json, encoding="utf-8") as f:
        dados = json.load(f)
    registros = []
    for item in dados["value"]:
        valdata = item["VALDATA"][:10]          # 'YYYY-MM-DD'
        valor = item["VALVALOR"]
        if valor is None:
            continue
        registros.append((valdata, float(valor)))
    registros.sort(key=lambda r: r[0])
    with open(NORM / nome_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["data", "valor"])
        w.writerows(registros)
    s = pd.Series({pd.Period(d[:7], freq="M"): v for d, v in registros})
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


print("== 1) Convertendo JSONs ==")
serie_ipca = json_para_csv("ipca.json", "ipca.csv")
serie_selic = json_para_csv("selic.json", "selic.csv")
serie_prod = json_para_csv("producao_industrial.json", "producao_industrial.csv")
for nome, s in [("ipca", serie_ipca), ("selic", serie_selic), ("producao_industrial", serie_prod)]:
    print(f"   {nome:22s}: {len(s):4d} meses  ({s.index.min()} -> {s.index.max()})")


# ---------------------------------------------------------------------------
# 2) indice_volatilidade.csv -> diario limpo (data, valor)
# ---------------------------------------------------------------------------
print("== 2) Limpando indice_volatilidade ==")
vol = pd.read_csv(RAW / "indice_volatilidade.csv", skiprows=6, usecols=[0, 1],
                  names=["data", "valor"], header=0, dtype=str)
vol = vol.dropna(subset=["data", "valor"])
vol = vol[vol["data"].str.strip() != ""]
vol["data"] = pd.to_datetime(vol["data"].str.strip(), format="%m/%d/%Y")
vol["valor"] = vol["valor"].astype(float)
vol = vol.sort_values("data").reset_index(drop=True)
vol_out = vol.copy()
vol_out["data"] = vol_out["data"].dt.strftime("%Y-%m-%d")
vol_out.to_csv(NORM / "indice_volatilidade_limpo.csv", index=False)
print(f"   {len(vol)} dias  ({vol['data'].min().date()} -> {vol['data'].max().date()})")


# ---------------------------------------------------------------------------
# 3) volatilidade diaria -> mensal (ULTIMO dia do mes)
# ---------------------------------------------------------------------------
vol["periodo"] = vol["data"].dt.to_period("M")
serie_vol = (vol.sort_values("data")
                .groupby("periodo")["valor"].last())
print(f"== 3) Volatilidade mensal (ultimo dia): {len(serie_vol)} meses "
      f"({serie_vol.index.min()} -> {serie_vol.index.max()}) ==")


# ---------------------------------------------------------------------------
# 4) curva_juros.csv (snapshot 1 dia) -> tidy/longo
# ---------------------------------------------------------------------------
print("== 4) Normalizando curva_juros (tidy) ==")
linhas = (RAW / "curva_juros.csv").read_text(encoding="latin-1").splitlines()
celulas = [ln.split(";") for ln in linhas]
data_ref = celulas[0][0].strip()  # '29/05/2026'
data_ref_iso = "-".join(reversed(data_ref.split("/")))  # '2026-05-29'
tidy = []  # data_referencia, secao, item, metrica, valor

# 4a) Parametros NSS (linha 1 cabecalho; linhas 2-3 valores)
nss_metricas = [c.strip() for c in celulas[0][1:]]
for row in celulas[1:3]:
    item = row[0].strip()
    if not item:
        continue
    for metrica, val in zip(nss_metricas, row[1:]):
        v = br_num(val)
        if v is not None:
            tidy.append((data_ref_iso, "parametros_nss", item, metrica, v))

# 4b) ETTJ Inflacao Implicita (cabecalho na linha indice 5)
ettj_hdr = ["ETTJ IPCA", "ETTJ PREF", "Inflacao Implicita"]
for row in celulas[6:73]:
    if not row or not row[0].strip():
        continue
    vertice = row[0].strip()
    for metrica, val in zip(ettj_hdr, row[1:]):
        v = br_num(val)
        if v is not None:
            tidy.append((data_ref_iso, "ettj_inflacao_implicita", vertice, metrica, v))

# 4c) PREFIXADOS (Circular 3.361): vertice;taxa  (linhas indice 76-85)
for row in celulas[76:86]:
    if not row or not row[0].strip():
        continue
    vertice = row[0].strip()
    v = br_num(row[1]) if len(row) > 1 else None
    if v is not None:
        tidy.append((data_ref_iso, "prefixados_circular_3361", vertice, "Taxa (%a.a.)", v))

# 4d) Erro Titulo a Titulo (linhas indice 89-121): Titulo;SELIC;Vencimento;Erro
for row in celulas[89:122]:
    if len(row) < 4 or not row[0].strip():
        continue
    titulo, selic_cod, venc, erro = row[0].strip(), row[1].strip(), row[2].strip(), row[3]
    v = br_num(erro)
    if v is not None:
        item = f"{titulo}|{selic_cod}|{venc}"
        tidy.append((data_ref_iso, "erro_titulo_a_titulo", item, "Erro (%a.a.)", v))

curva = pd.DataFrame(tidy, columns=["data_referencia", "secao", "item", "metrica", "valor"])
curva.to_csv(NORM / "curva_juros_normalizado.csv", index=False)
print(f"   {len(curva)} linhas tidy | secoes: "
      + ", ".join(f"{k}={v}" for k, v in curva['secao'].value_counts().sort_index().items()))


# ---------------------------------------------------------------------------
# 5) Curva de juros - inclinacao (slope) a partir do Tesouro Direto pre-fixado
#    slope = juro pre 10 anos - juro pre 2 anos (interpolacao linear no prazo;
#    np.interp faz clamp na ponta quando 10A excede o vertice mais longo do dia).
#    Equivalente BR do FRED T10Y2Y. Fonte: Tesouro Transparente (LTN + NTN-F).
# ---------------------------------------------------------------------------
print("== 5) Curva de juros: inclinacao 10A-2A (Tesouro Direto) ==")
td = pd.read_csv(RAW / "tesouro_direto_prefixado.csv", sep=";", dtype=str)
td.columns = [c.strip() for c in td.columns]
td["venc"] = pd.to_datetime(td["Data Vencimento"], format="%d/%m/%Y")
td["base"] = pd.to_datetime(td["Data Base"], format="%d/%m/%Y")
td["taxa"] = td["Taxa Compra Manha"].str.replace(",", ".").astype(float)
td["mat"] = (td["venc"] - td["base"]).dt.days / 365.25


def _slope(g):
    s = g.groupby("mat")["taxa"].mean()      # 2 titulos no mesmo prazo -> media
    if len(s) < 2:
        return np.nan
    y2, y10 = np.interp([2.0, 10.0], s.index.values, s.values)  # clamp fora do range
    return y10 - y2


slope_diario = td.groupby("base").apply(_slope, include_groups=False).dropna()

# normalizada (diaria): data, valor
(slope_diario.rename("valor").rename_axis("data").reset_index()
 .assign(data=lambda d: d["data"].dt.strftime("%Y-%m-%d"))
 .to_csv(NORM / "curva_juros_slope.csv", index=False))

# mensal: ultimo dia util do mes (consistente com a volatilidade)
serie_slope = slope_diario.groupby(slope_diario.index.to_period("M")).last()
print(f"   diario: {len(slope_diario)} dias | mensal: {len(serie_slope)} meses "
      f"({serie_slope.index.min()} -> {serie_slope.index.max()})")


# ---------------------------------------------------------------------------
# 6) Merge mensal por INTERSECAO das 6 series
# ---------------------------------------------------------------------------
print("== 6) Merge mensal (intersecao) ==")
# spread_credito.csv (BR: ';', decimal ',', datas DD/MM/YYYY)
spread = pd.read_csv(RAW / "spread_credito.csv", sep=";", dtype=str,
                     quotechar='"', encoding="utf-8")
spread.columns = [c.strip().strip('"') for c in spread.columns]
spread["periodo"] = pd.to_datetime(spread["data"], format="%d/%m/%Y").dt.to_period("M")
spread["valor"] = spread["valor"].apply(br_num)
serie_spread = spread.set_index("periodo")["valor"].sort_index()

# versao normalizada/padronizada do spread (data YYYY-MM-01, decimal '.')
(serie_spread.rename("valor")
 .rename_axis("periodo").reset_index()
 .assign(data=lambda d: d["periodo"].dt.to_timestamp().dt.strftime("%Y-%m-%d"))
 [["data", "valor"]]
 .to_csv(NORM / "spread_credito.csv", index=False))

painel = pd.concat(
    {
        "valor_ipca": serie_ipca,
        "valor_selic": serie_selic,
        "valor_producao_industrial": serie_prod,
        "valor_spread_credito": serie_spread,
        "valor_indice_volatilidade": serie_vol,
        "valor_curva_juros_slope": serie_slope,
    },
    axis=1, join="inner",
).sort_index()

painel.index.name = "data"
painel = painel.reset_index()
painel["data"] = painel["data"].astype(str)  # 'YYYY-MM'
painel.to_csv(BASE / "dataset_mensal_merged.csv", index=False)

print(f"   Painel final: {painel.shape[0]} meses x {painel.shape[1]} colunas")
print(f"   Periodo: {painel['data'].iloc[0]} -> {painel['data'].iloc[-1]}")
print(f"   Colunas: {list(painel.columns)}")
print(f"   Celulas vazias no painel: {painel.isna().sum().sum()}")
print("\nPrimeiras linhas:")
print(painel.head(3).to_string(index=False))
print("...")
print(painel.tail(3).to_string(index=False))
