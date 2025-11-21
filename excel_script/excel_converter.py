import os
import csv
import pandas as pd

CANDIDATE_ENCODINGS = [
    "utf-8", "utf-8-sig", "cp1252", "latin1", "iso-8859-1",
    "utf-16", "utf-16-le", "utf-16-be"
]
CANDIDATE_SEPARATORS = [",", ";", "\t", "|"]


def sniff_delimiter(sample_text, default=";"):
    """Prova a indovinare il delimitatore."""
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=",".join(CANDIDATE_SEPARATORS))
        return dialect.delimiter
    except Exception:
        first = sample_text.splitlines()[0] if sample_text else ""
        counts = {sep: first.count(sep) for sep in CANDIDATE_SEPARATORS}
        best = max(counts, key=counts.get) if counts else default
        return best if counts.get(best, 0) > 0 else default


def robust_read_csv(csv_path):
    """Legge il CSV provando vari encoding e separatori."""
    with open(csv_path, "rb") as f:
        raw = f.read(32768)

    sample_decoded = None
    enc_for_sample = None
    for enc in CANDIDATE_ENCODINGS:
        try:
            sample_decoded = raw.decode(enc)
            enc_for_sample = enc
            break
        except Exception:
            continue
    if sample_decoded is None:
        sample_decoded = raw.decode("latin1", errors="replace")
        enc_for_sample = "latin1(replace)"

    for enc in [enc_for_sample] + [e for e in CANDIDATE_ENCODINGS if e != enc_for_sample]:
        try:
            df = pd.read_csv(csv_path, encoding=enc, sep=None, engine="python")
            if df.shape[1] > 1:
                return df, enc, "inferred"
        except Exception:
            pass

        try:
            sep_guess = sniff_delimiter(sample_decoded, default=";")
            df = pd.read_csv(csv_path, encoding=enc, sep=sep_guess, engine="python")
            if df.shape[1] > 1:
                return df, enc, sep_guess
        except Exception:
            pass

        for sep in CANDIDATE_SEPARATORS:
            try:
                df = pd.read_csv(csv_path, encoding=enc, sep=sep, engine="python")
                if df.shape[1] > 1:
                    return df, enc, sep
            except Exception:
                continue

    df = pd.read_csv(csv_path, encoding="latin1", sep=None, engine="python", on_bad_lines="skip")
    return df, "latin1(on_bad_lines=skip)", "inferred/1col"


def convert_single_csv_to_excel(csv_path):
    """
    Converte un singolo file CSV in Excel.
    Crea una sottocartella 'excel_output' nella stessa directory del CSV.
    
    Args:
        csv_path (str): Percorso completo del file CSV
        
    Returns:
        str: Percorso del file Excel generato, o None se errore
    """
    if not os.path.isfile(csv_path) or not csv_path.lower().endswith(".csv"):
        print(f"❌ Non è un file CSV valido: {csv_path}")
        return None
    
    dirpath = os.path.dirname(csv_path)
    filename = os.path.basename(csv_path)
    excel_folder = os.path.join(dirpath, "excel_output")
    os.makedirs(excel_folder, exist_ok=True)
    
    excel_file = os.path.join(excel_folder, os.path.splitext(filename)[0] + ".xlsx")
    
    try:
        df, enc, sep = robust_read_csv(csv_path)
        df.to_excel(excel_file, index=False)
        print(f"✅ Convertito: {csv_path} → {excel_file}  (encoding: {enc} | sep: {sep} | cols: {df.shape[1]})")
        return excel_file
    except Exception as e:
        print(f"❌ ERRORE su {csv_path}: {e}")
        return None


# FUNZIONE ORIGINALE modificata per usare convert_single_csv_to_excel
def convert_csv_to_excel(path):
    """
    Converte CSV in Excel. Supporta sia file singolo che directory.
    
    Args:
        path (str): Percorso del file CSV o directory contenente CSV
    """
    if os.path.isfile(path) and path.lower().endswith(".csv"):
        # Caso singolo file CSV
        convert_single_csv_to_excel(path)
        
    elif os.path.isdir(path):
        # Caso directory
        for dirpath, _, files in os.walk(path):
            excel_folder = os.path.join(dirpath, "excel_output")
            csv_files = [f for f in files if f.lower().endswith(".csv")]
            if not csv_files:
                continue

            os.makedirs(excel_folder, exist_ok=True)

            for file in csv_files:
                csv_file = os.path.join(dirpath, file)
                excel_file = os.path.join(excel_folder, os.path.splitext(file)[0] + ".xlsx")

                if os.path.exists(excel_file):
                    print(f"⚠️  Excel già esiste, salto: {excel_file}")
                    continue

                convert_single_csv_to_excel(csv_file)
    else:
        print("❌ Il percorso specificato non è valido (né file .csv né directory).")


# Per uso standalone
if __name__ == "__main__":
    csv_root = r"C:\Users"  # Modifica questo path
    convert_csv_to_excel(csv_root)
