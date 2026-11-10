"""Evalúa en paralelo los datasets listados en un CSV y guarda TW, CT y CE90.

Orquestador: lanza un subproceso por fila (con un log por fila) y escribe el CSV de salida
con las columnas de entrada más TW, CT, CE90, status y error.
Worker (uso interno): --worker, evalúa un único pickle y vuelca el resultado en un JSON.
"""
import argparse
import csv
import json
import os
import pickle
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
METRIC_COLUMNS = ["TW", "CT", "CE90"]
EXTRA_COLUMNS = ["status", "error"]


def least_squares_transform(cc_data, pos_data):
    import numpy as np

    # Pad the data with ones, so that the transformation can do translations too
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
    unpad = lambda x: x[:, :-1]

    A, _, _, _ = np.linalg.lstsq(pad(cc_data), pad(pos_data), rcond=None)
    return unpad(np.dot(pad(cc_data), A))


def evaluate(data_path, model_name, model_path):
    import numpy as np
    from sklearn.manifold import trustworthiness

    import cc_tmcn.model.model_factory as model_factory
    from fsspec_sftp import fs

    with fs.fs.open(data_path, "rb") as f:
        data = list(pickle.load(f))

    input_data, pos = data[0], data[1]
    model = model_factory.load_model(model_name, model_path)
    X_embedded = model.predict(input_data)

    # 5% of the nearest neighbors are used for the KPIs
    n_neighbors = int(len(X_embedded) * 0.05)
    tw = trustworthiness(pos, X_embedded, n_neighbors=n_neighbors)
    ct = trustworthiness(X_embedded, pos, n_neighbors=n_neighbors)

    X_embedded = least_squares_transform(X_embedded, pos)
    ce90 = np.percentile(np.linalg.norm(X_embedded - pos, axis=1), 90)
    return {"TW": float(tw), "CT": float(ct), "CE90": float(ce90)}


def run_worker(args):
    print(f"Evaluando {args.data_path}", flush=True)
    metrics = evaluate(args.data_path, args.model_name, args.model_path)
    print(f"TW: {metrics['TW']:.6f}, CT: {metrics['CT']:.6f}, CE90: {metrics['CE90']:.4f}", flush=True)
    Path(args.worker_output).write_text(json.dumps(metrics))


def read_rows(input_csv):
    with open(input_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if "destination_file" not in fieldnames:
        raise SystemExit(f"{input_csv} debe tener la columna 'destination_file'")
    return fieldnames, rows


def write_results(output_csv, fieldnames, rows):
    columns = fieldnames + METRIC_COLUMNS + EXTRA_COLUMNS
    tmp = output_csv.with_suffix(output_csv.suffix + ".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, output_csv)


def launch(index, row, args, log_dir):
    name = Path(row["destination_file"]).name
    log_path = log_dir / f"evaluate_{index:03d}_{name}.log"
    result_path = log_dir / f"evaluate_{index:03d}_{name}.json"
    result_path.unlink(missing_ok=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [
        sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
        "--data-path", args.base_path + row["destination_file"],
        "--model-name", args.model_name,
        "--model-path", str(args.model_path),
        "--worker-output", str(result_path),
    ]
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            cmd, cwd=REPO_ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT,
        )
    return {"index": index, "proc": proc, "log": log_path, "result": result_path}


def run_orchestrator(args):
    output_csv = Path(args.output_csv)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames, rows = read_rows(args.input_csv)
    results = [dict(row) for row in rows]
    pending = list(enumerate(rows, start=1))
    running = []
    failed = 0
    done = 0

    def stop_children(*_):
        for job in running:
            job["proc"].terminate()
        sys.exit(130)

    signal.signal(signal.SIGTERM, stop_children)

    print(f"{len(rows)} evaluaciones, {args.jobs} en paralelo. Logs en {log_dir}", flush=True)
    try:
        while pending or running:
            while pending and len(running) < args.jobs:
                index, row = pending.pop(0)
                job = launch(index, row, args, log_dir)
                running.append(job)
                print(f"[{index}/{len(rows)}] Lanzada {row['destination_file']} (log: {job['log']})", flush=True)

            time.sleep(1)
            for job in [j for j in running if j["proc"].poll() is not None]:
                running.remove(job)
                result = results[job["index"] - 1]
                code = job["proc"].returncode
                done += 1
                if code == 0 and job["result"].exists():
                    result.update(json.loads(job["result"].read_text()))
                    result["status"] = "ok"
                    print(f"[{job['index']}/{len(rows)}] OK", flush=True)
                else:
                    failed += 1
                    result["status"] = "error"
                    result["error"] = f"código de salida {code}; ver {job['log']}"
                    print(f"[{job['index']}/{len(rows)}] ERROR (código {code}). Log: {job['log']}", file=sys.stderr, flush=True)
                job["result"].unlink(missing_ok=True)
                write_results(output_csv, fieldnames, results)
    except KeyboardInterrupt:
        stop_children()

    write_results(output_csv, fieldnames, results)
    print(f"Terminado: {done - failed} correctas, {failed} con error. Resultados en {output_csv}", flush=True)
    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser(description="Evalúa modelos desde un CSV y guarda TW, CT y CE90 en un CSV de salida.")
    ap.add_argument("--input-csv", help="CSV de entrada (columna destination_file obligatoria)")
    ap.add_argument("--output-csv", help="CSV de salida: columnas de entrada + TW, CT, CE90, status, error")
    ap.add_argument("--base-path", default="/data/datasets/fraunhofer5G/", help="Prefijo remoto de destination_file")
    ap.add_argument("--model-name", default="Model_5G", help="Name of the model")
    ap.add_argument("--model-path", default=str(REPO_ROOT / "output"), help="Path to the model")
    ap.add_argument("--jobs", type=int, default=5, help="Número de evaluaciones en paralelo")
    ap.add_argument("--log-dir", default=str(REPO_ROOT / "output" / "evaluate"), help="Directorio de logs por fila")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--data-path", help=argparse.SUPPRESS)
    ap.add_argument("--worker-output", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        run_worker(args)
        return 0
    if not args.input_csv or not args.output_csv:
        ap.error("--input-csv y --output-csv son obligatorios")
    if args.jobs < 1:
        ap.error("--jobs debe ser >= 1")
    return run_orchestrator(args)


if __name__ == "__main__":
    sys.exit(main())
