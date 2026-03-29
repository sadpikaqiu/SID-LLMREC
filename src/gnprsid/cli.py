from __future__ import annotations

import argparse
import json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Linux-first GNPR-SID CLI")
    subparsers = parser.add_subparsers(dest="command_group", required=True)

    data_parser = subparsers.add_parser("data")
    data_sub = data_parser.add_subparsers(dest="data_command", required=True)
    import_parser = data_sub.add_parser("import-legacy")
    import_parser.add_argument("--dataset", default="NYC")
    import_parser.add_argument("--legacy-root", required=True)

    prepare_parser = data_sub.add_parser("prepare-nyc")
    prepare_parser.add_argument("--dataset", default="NYC")
    prepare_parser.add_argument("--current-k", type=int, default=49)
    prepare_parser.add_argument("--sid-map-path", default=None)

    sid_parser = subparsers.add_parser("sid")
    sid_sub = sid_parser.add_subparsers(dest="sid_command", required=True)
    sid_train_parser = sid_sub.add_parser("train")
    sid_train_parser.add_argument("--config", required=True)
    sid_export_parser = sid_sub.add_parser("export")
    sid_export_parser.add_argument("--config", required=True)
    sid_export_parser.add_argument("--checkpoint-path", default=None)

    align_parser = subparsers.add_parser("alignment")
    align_sub = align_parser.add_subparsers(dest="alignment_command", required=True)
    build_align = align_sub.add_parser("build-data")
    build_align.add_argument("--dataset", default="NYC")
    build_align.add_argument("--sid-map-path", default=None)
    build_align.add_argument("--valid-ratio", type=float, default=0.1)
    build_align.add_argument("--seed", type=int, default=42)
    build_align.add_argument("--semantic-schema", default="semantic_spatial_v2")
    build_align.add_argument("--grid-size", type=int, default=8)
    build_align.add_argument("--split-by", default="abc")
    eval_align = align_sub.add_parser("evaluate")
    eval_align.add_argument("--dataset", default="NYC")
    eval_align.add_argument("--model-config", required=True)
    eval_align.add_argument("--checkpoint-path", default=None)
    eval_align.add_argument("--split", default="valid", choices=["train", "valid"])
    eval_align.add_argument(
        "--task",
        default="sid_to_abc_profile",
        choices=["sid_to_abc_profile", "abc_profile_to_a", "abc_profile_to_ab", "abc_profile_to_abc"],
    )
    eval_align.add_argument("--data-path", default=None)
    eval_align.add_argument("--batch-size", type=int, default=1)
    eval_align.add_argument("--limit", type=int, default=None)
    eval_align.add_argument("--output-path", default=None)

    train_parser = subparsers.add_parser("train")
    train_sub = train_parser.add_subparsers(dest="train_command", required=True)
    run_parser = train_sub.add_parser("run")
    run_parser.add_argument("--stage", choices=["alignment", "sft", "grpo"], required=True)
    run_parser.add_argument("--config", required=True)
    merge_parser = train_sub.add_parser("merge-peft")
    merge_parser.add_argument("--model-config", required=True)
    merge_parser.add_argument("--adapter-path", required=True)
    merge_parser.add_argument("--output-path", default=None)

    retrieval_parser = subparsers.add_parser("retrieval")
    retrieval_sub = retrieval_parser.add_subparsers(dest="retrieval_command", required=True)
    bank_parser = retrieval_sub.add_parser("build-bank")
    bank_parser.add_argument("--dataset", default="NYC")
    bank_parser.add_argument("--repr", choices=["id", "sid"], required=True)
    bank_parser.add_argument("--output-path", default=None)

    similar_parser = retrieval_sub.add_parser("build-similar")
    similar_parser.add_argument("--dataset", default="NYC")
    similar_parser.add_argument("--repr", choices=["id", "sid"], required=True)
    similar_parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    similar_parser.add_argument("--config", default="configs/retrieval/default.yaml")
    similar_parser.add_argument("--model-config", default=None)
    similar_parser.add_argument("--bank-path", default=None)
    similar_parser.add_argument("--output-path", default=None)
    similar_parser.add_argument("--model-name-or-path", default=None)

    inspect_parser = retrieval_sub.add_parser("inspect-encoder")
    inspect_parser.add_argument("--bank-path", required=True)
    inspect_parser.add_argument("--repr", choices=["id", "sid"], required=True)
    inspect_parser.add_argument("--config", default="configs/retrieval/default.yaml")
    inspect_parser.add_argument("--model-config", default=None)
    inspect_parser.add_argument("--sample-count", type=int, default=2)
    inspect_parser.add_argument("--model-name-or-path", default=None)

    infer_parser = subparsers.add_parser("infer")
    infer_sub = infer_parser.add_subparsers(dest="infer_command", required=True)
    batch_parser = infer_sub.add_parser("batch")
    batch_parser.add_argument("--dataset", default="NYC")
    batch_parser.add_argument("--repr", choices=["id", "sid"], required=True)
    batch_parser.add_argument("--history-source", choices=["current", "original", "retrieval", "hybrid"], required=True)
    batch_parser.add_argument("--model-config", required=True)
    batch_parser.add_argument("--checkpoint-path", default=None)
    batch_parser.add_argument("--split", default="test")
    batch_parser.add_argument("--retrieval-bank-path", default=None)
    batch_parser.add_argument("--similar-map-path", default=None)
    batch_parser.add_argument("--history-path", default=None)
    batch_parser.add_argument("--top-k-retrieval", type=int, default=None)
    batch_parser.add_argument("--batch-size", type=int, default=1)
    batch_parser.add_argument("--limit", type=int, default=None)
    batch_parser.add_argument("--output-path", default=None)

    eval_parser = subparsers.add_parser("eval")
    eval_sub = eval_parser.add_subparsers(dest="eval_command", required=True)
    eval_run = eval_sub.add_parser("run")
    eval_run.add_argument("--predictions", required=True)
    eval_run.add_argument("--output-path", default=None)
    eval_summary = eval_sub.add_parser("summarize")
    eval_summary.add_argument("--dataset", default="NYC")
    eval_summary.add_argument("--eval-dir", default=None)
    eval_summary.add_argument("--output-path", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    display_result = None

    if args.command_group == "data":
        from gnprsid.data.legacy import import_legacy_dataset
        from gnprsid.data.prepare import prepare_nyc

        if args.data_command == "import-legacy":
            result = import_legacy_dataset(args.dataset, args.legacy_root)
        else:
            result = prepare_nyc(args.dataset, args.current_k, sid_map_path=args.sid_map_path)
    elif args.command_group == "sid":
        from gnprsid.sid.export import export_sid_from_config
        from gnprsid.sid.train import train_sid_from_config

        if args.sid_command == "train":
            result = train_sid_from_config(args.config)
        else:
            result = export_sid_from_config(args.config, checkpoint_path=args.checkpoint_path)
    elif args.command_group == "alignment":
        if args.alignment_command == "build-data":
            from gnprsid.alignment.build_data import build_alignment_data

            result = build_alignment_data(
                args.dataset,
                sid_map_path=args.sid_map_path,
                valid_ratio=args.valid_ratio,
                seed=args.seed,
                semantic_schema=args.semantic_schema,
                grid_size=args.grid_size,
                split_by=args.split_by,
            )
        else:
            from gnprsid.alignment.evaluate import evaluate_alignment

            result = evaluate_alignment(
                dataset=args.dataset,
                model_config_path=args.model_config,
                checkpoint_path=args.checkpoint_path,
                split=args.split,
                task=args.task,
                data_path=args.data_path,
                batch_size=args.batch_size,
                limit=args.limit,
                output_path=args.output_path,
            )
            display_result = result["metrics"]
    elif args.command_group == "train":
        if args.train_command == "run":
            from gnprsid.train.base import run_training_stage

            result = run_training_stage(args.config, stage_override=args.stage)
        else:
            from gnprsid.train.merge import merge_peft_adapter

            result = merge_peft_adapter(
                model_config_path=args.model_config,
                adapter_path=args.adapter_path,
                output_path=args.output_path,
            )
    elif args.command_group == "retrieval":
        if args.retrieval_command == "build-bank":
            from gnprsid.retrieval.bank import build_retrieval_bank

            result = build_retrieval_bank(args.dataset, args.repr, output_path=args.output_path)
        elif args.retrieval_command == "build-similar":
            from gnprsid.retrieval.similarity import build_similarity_map

            result = build_similarity_map(
                args.dataset,
                args.repr,
                split=args.split,
                retrieval_config_path=args.config,
                model_config_path=args.model_config,
                bank_path=args.bank_path,
                output_path=args.output_path,
                model_name_or_path=args.model_name_or_path,
            )
        else:
            from gnprsid.retrieval.inspect import inspect_encoder

            result = inspect_encoder(
                bank_path=args.bank_path,
                repr_name=args.repr,
                model_config_path=args.model_config,
                retrieval_config_path=args.config,
                sample_count=args.sample_count,
                model_name_or_path=args.model_name_or_path,
            )
    elif args.command_group == "infer":
        from gnprsid.inference.batch import run_batch_inference

        result = run_batch_inference(
            dataset=args.dataset,
            repr_name=args.repr,
            history_source=args.history_source,
            model_config_path=args.model_config,
            checkpoint_path=args.checkpoint_path,
            split=args.split,
            retrieval_bank_path=args.retrieval_bank_path,
            similar_map_path=args.similar_map_path,
            history_path=args.history_path,
            top_k_retrieval=args.top_k_retrieval,
            batch_size=args.batch_size,
            limit=args.limit,
            output_path=args.output_path,
        )
    else:
        from gnprsid.eval.run import run_evaluation
        from gnprsid.eval.summarize import summarize_evaluations

        if args.eval_command == "run":
            result = run_evaluation(args.predictions, output_path=args.output_path)
            display_result = result["metrics"]
        else:
            result = summarize_evaluations(args.dataset, eval_dir=args.eval_dir, output_path=args.output_path)

    if display_result is None:
        display_result = result
    print(json.dumps(display_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
