# -*- coding: utf-8 -*-
"""Adaptador de arquivo padronizado para o NIRCLS (nirtooltik).

Nenhum dado bruto de pesquisa deve ser adicionado ao Git.
O alvo e um rótulo comercial PROVISÓRIO até confirmação anatômica.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from collections import defaultdict

def main():
    parser=argparse.ArgumentParser(description="Prepara entrada do NIRCLS para qualificação MicroNIR Viavi")
    parser.add_argument("--input",required=True,help="CSV com metadados padronizados e colunas nm_*")
    parser.add_argument("--output-dir",required=True)
    args=parser.parse_args()
    path=Path(args.input);out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    with path.open("r",newline="",encoding="utf-8-sig") as f:
        rd=csv.DictReader(f)
        spectral=[col for col in (rd.fieldnames or []) if col.startswith("nm_")]
        if not spectral: raise ValueError("Nenhuma coluna nm_* na base padronizada")
        nm=[col.removeprefix("nm_") for col in spectral]
        rows=[];bad=0;no_label=0
        for row in rd:
            if row.get("status_espectral")!="COMPLETO":
                bad+=1;continue
            popular=(row.get("nome_popular_padronizado") or "").strip()
            if not popular or popular=="Não informado":
                no_label+=1;continue
            code=(row.get("grupo_provisorio_A") or "").strip()
            if not code: raise ValueError("Código A ausente em registro válido")
            try: signals=[float(row[c]) for c in spectral]
            except (ValueError,TypeError) as e:
                bad+=1;continue
            rows.append((code,popular,signals))
    def dump(file, entries):
        with (out/file).open("w",newline="",encoding="utf-8-sig") as f:
            wr=csv.writer(f);wr.writerow(["grupo_codigo","nome_popular"]+nm)
            wr.writerows(entries)
    dump("NIRCLS_VIAVI_leituras.csv",rows)
    by_code=defaultdict(list); by_labels=defaultdict(set)
    for code,pop,signal in rows:
        by_code[code].append(signal);by_labels[code].add(pop)
    averaged=[];conflicted=[]
    for code in sorted(by_code):
        if len(by_labels[code])!=1:
            conflicted.append(code);continue
        signal=[sum(vals)/len(vals) for vals in zip(*by_code[code])]
        averaged.append((code,next(iter(by_labels[code])),signal))
    dump("NIRCLS_VIAVI_media_por_codigo.csv",averaged)
    from collections import Counter
    n_por_classe=Counter(l for _,l,_ in averaged)
    classes_para_modelo={k for k,n in n_por_classe.items() if n>=8}
    readings_model=[r for r in rows if r[1] in classes_para_modelo]
    averages_model=[r for r in averaged if r[1] in classes_para_modelo]
    dump("NIRCLS_VIAVI_modelo_min8grupos.csv",readings_model)
    dump("NIRCLS_VIAVI_media_modelo_min8grupos.csv",averages_model)
    info={"classes_elegiveis_min8grupos":len(classes_para_modelo),
          "leituras_para_modelo":len(readings_model),
          "codigos_para_modelo":len(averages_model),
          "fonte":str(path),"leitura_valida_com_rotulo":len(rows),
          "linhas_sem_espectro":bad,"linhas_sem_nome_popular":no_label,
          "n_codigos_distintos_na_saida":len(by_code),
          "n_codigos_medios":len(averaged),"n_comprimentos_onda":len(nm),
          "codigos_com_rotulo_inconsistente":conflicted,
          "nota":"Não confundir rótulos populares com referência anatômica autenticada; usar group_col em avaliação por leituras."}
    (out/"proveniencia_nircls.json").write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding="utf8")
    print(json.dumps(info,ensure_ascii=False))
if __name__=="__main__":main()
