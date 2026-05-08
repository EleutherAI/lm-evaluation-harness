#!/usr/bin/env python3
import argparse
import json
import sys

from datasets import get_dataset_config_names, load_dataset


def features_to_simple_dict(features):
    # Convierte Features de HF a dict simple nombre -> tipo
    return {k: str(v) for k, v in features.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Valida consistencia de esquema entre splits de un dataset HF."
    )
    parser.add_argument("--repo", default="gplsi/cieaCOVA", help="Repo HF del dataset")
    parser.add_argument("--config", default="text_generation", help="Config del dataset")
    parser.add_argument("--splits", nargs="+", default=["train", "test"], help="Splits a validar")
    parser.add_argument("--show_sample", action="store_true", help="Muestra una fila de ejemplo por split")
    args = parser.parse_args()

    print(f"Dataset repo: {args.repo}")
    configs = get_dataset_config_names(args.repo)
    print(f"Configs disponibles: {configs}")

    if args.config not in configs:
        print(f"\nERROR: La config '{args.config}' no existe en {args.repo}")
        sys.exit(1)

    split_features = {}
    split_sizes = {}

    for split in args.splits:
        try:
            ds = load_dataset(args.repo, args.config, split=split)
        except Exception as e:
            print(f"\nERROR cargando split '{split}': {e}")
            sys.exit(1)

        split_sizes[split] = len(ds)
        split_features[split] = features_to_simple_dict(ds.features)

        print(f"\nSplit: {split}")
        print(f"Tamaño: {split_sizes[split]}")
        print("Features:")
        print(json.dumps(split_features[split], indent=2, ensure_ascii=False))

        if args.show_sample and len(ds) > 0:
            print("Ejemplo[0] keys:")
            print(list(ds[0].keys()))

    # Compara esquema del primer split contra el resto
    base_split = args.splits[0]
    base = split_features[base_split]
    ok = True

    for split in args.splits[1:]:
        current = split_features[split]

        base_keys = set(base.keys())
        cur_keys = set(current.keys())

        missing_in_current = sorted(base_keys - cur_keys)
        extra_in_current = sorted(cur_keys - base_keys)

        type_mismatch = []
        for k in sorted(base_keys & cur_keys):
            if base[k] != current[k]:
                type_mismatch.append((k, base[k], current[k]))

        if missing_in_current or extra_in_current or type_mismatch:
            ok = False
            print(f"\nINCONSISTENCIA detectada entre '{base_split}' y '{split}':")
            if missing_in_current:
                print(f"- Faltan en {split}: {missing_in_current}")
            if extra_in_current:
                print(f"- Sobran en {split}: {extra_in_current}")
            if type_mismatch:
                print("- Tipos distintos:")
                for k, t_base, t_cur in type_mismatch:
                    print(f"  - {k}: {base_split}={t_base} | {split}={t_cur}")

    if not ok:
        print("\nResultado: FAIL. Debes unificar columnas y tipos entre splits.")
        sys.exit(1)

    print("\nResultado: OK. Esquema consistente entre splits.")
    sys.exit(0)


if __name__ == "__main__":
    main()