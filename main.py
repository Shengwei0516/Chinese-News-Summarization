import os
import json
import math
import torch
import argparse
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--train_file", 
        type=str, 
        default=None, 
        help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file",
        type=str, default=None,
        help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file",
        type=str, default=None,
        help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for the dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=0, 
        help="Total number of training epochs to perform."
        )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--stratgy",
        type=str,
        default="defaults",
        help="Generation Strategies (greedy, beam_search, top_k_sampling, top_p_sampling, temperature).",
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help="Where to store the final model."
        )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=11151033, 
        help="A seed for reproducible training."
        )
    parser.add_argument(
        "--prediction_path",
        type=str,
        default="prediction.jsonl",
        help="Path to the output prediction file."
    )
    parser.add_argument(
        "--gen_max_length",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--gen_num_beams",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--gen_top_k",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--gen_top_p",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--gen_temperature",
        type=float,
        default=1.0,
    )
    args = parser.parse_args()

    return args


def pad_to_title(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for i in range(len(lines)):
        json_obj = json.loads(lines[i])
        json_obj["title"] = ""
        lines[i] = json.dumps(json_obj)
    with open(file_path, 'w') as file:
        file.writelines('\n'.join(lines))


def learning_curves(result_history, output_dir):

    import matplotlib.pyplot as plt
    x = range(1, len(result_history) + 1)
    rouge1 = [result["rouge-1"] for result in result_history]
    rouge2 = [result["rouge-2"] for result in result_history]
    rougel = [result["rouge-l"] for result in result_history]
    
    for values, title in zip([rouge1, rouge2, rougel], ["Rouge-1", "Rouge-2", "Rouge-L"]):
        values_r = [value["r"] for value in values]
        values_p = [value["p"] for value in values]
        values_f = [value["f"] for value in values]
        plt.figure()
        plt.plot(x, values_r, label="Recall")
        plt.plot(x, values_p, label="Precision")
        plt.plot(x, values_f, label="F-score")
        plt.title(f"{title} Metrics History")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/learning_curves({title})")

    with open(os.path.join(output_dir, "result_history.json"), "w") as f:
        json.dump(result_history, f, indent=4)


def main():
    args = parse_args()
    set_seed(args.seed)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, trust_remote_code=False, config=config)

    data_files = {}
    if args.train_file is not None:
        data_key = "train"
        data_files[data_key] = args.train_file
    if args.validation_file is not None:
        from tw_rouge import get_rouge
        data_key = "validation"
        data_files[data_key] = args.validation_file
    if args.test_file is not None:
        data_key = "test"
        pad_to_title(args.test_file)
        data_files[data_key] = args.test_file
    raw_datasets = load_dataset("json", data_files=data_files)
    column_names = raw_datasets[data_key].column_names
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100
    )

    def preprocess_function(examples):
        inputs = tokenizer(
            ["summarize: " + inp for inp in examples["maintext"]], 
            max_length=args.max_length, 
            padding="max_length", 
            truncation=True,
            )
        labels = tokenizer(
            text_target=examples["title"],
            max_length=128,
            padding="max_length",
            truncation=True,
            )
        inputs["labels"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        return inputs

    def generation_strategies(stratgy):
        if stratgy == "greedy":
            gen_kwargs = {"max_length": args.gen_max_length}
        elif stratgy == "beam_search":
            gen_kwargs = {"max_length": args.gen_max_length, "num_beams": args.gen_num_beams}
        elif stratgy == "top_k_sampling":
            gen_kwargs = {"do_sample": True, "max_length": args.gen_max_length, "top_k": args.gen_top_k}
        elif stratgy == "top_p_sampling":
            gen_kwargs = {"do_sample": True, "max_length": args.gen_max_length, "top_p": args.gen_top_p}
        elif stratgy == "temperature":
            gen_kwargs = {"do_sample": True, "max_length": args.gen_max_length, "temperature": args.gen_temperature}
        else:
            gen_kwargs = {}
        print(gen_kwargs)
        return gen_kwargs

    def validation(dataloader):
        model.eval()
        all_preds = []
        all_labels = []
        gen_kwargs = generation_strategies(args.stratgy)
        for batch in tqdm(dataloader, desc=f"{args.stratgy}", ncols=100, unit_scale=True, colour="green"):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )
                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens,
                    dim=1, 
                    pad_index=tokenizer.pad_token_id,
                )
                labels = batch["labels"]
                generated_tokens, labels = accelerator.gather_for_metrics((generated_tokens, labels))
                generated_tokens = generated_tokens.cpu().numpy()
                labels = labels.cpu().numpy()
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                decoded_preds = [pred.strip() for pred in decoded_preds]
                decoded_labels = [label.strip() for label in decoded_labels]
                all_preds += decoded_preds
                all_labels += decoded_labels
        return all_preds, all_labels

    if args.epochs > 0 and args.train_file is not None:
        train_dataset = raw_datasets["train"].map(
            preprocess_function,
            batched=True,
            remove_columns=column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on Train dataset",
        )
        train_dataloader = DataLoader(
            train_dataset,
            collate_fn=data_collator,
            batch_size=args.batch_size
            )
        if args.validation_file is not None:
            valid_dataset = raw_datasets["validation"].map(
                preprocess_function,
                batched=True,
                remove_columns=column_names,
                load_from_cache_file=True,
                desc="Running tokenizer on validation dataset",
            )
            valid_dataloader = DataLoader(
                valid_dataset,
                collate_fn=data_collator,
                batch_size=args.batch_size
            )
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)

        if args.validation_file is not None:
            model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader, valid_dataloader
            )
        else:
            model, optimizer, train_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader
            )
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        max_train_steps = args.epochs * num_update_steps_per_epoch
        args.epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
        valid_history = []
        for epoch in range(args.epochs):
            model.train()
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}", ncols=100, unit_scale=True, colour="red"):
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()

            if args.validation_file is not None:
                all_preds, all_labels = validation(valid_dataloader)
                result = get_rouge(all_preds, all_labels, avg=True, ignore_empty=True)
                result = {i: {j: result[i][j] * 100 for j in result[i]} for i in result}
                print(f"rouge-1: {result['rouge-1']['f']:.1f}, rouge-2: {result['rouge-2']['f']:.1f} rouge-L: {result['rouge-l']['f']:.1f}")
                valid_history.append(result)

        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            tokenizer.save_pretrained(args.output_dir)
            with open(os.path.join(args.output_dir, "params.json"), 'w') as f:
                json.dump(vars(args), f, indent=4)
            if args.validation_file is not None:
                all_results = {f"eval_{k}": v for k, v in result.items()}
                learning_curves(valid_history, args.output_dir)
                with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                    json.dump(all_results, f, indent=4)
            print(f"\nThe final model has been saved to {args.output_dir}")

    if args.test_file is not None:
        test_dataset = raw_datasets["test"].map(
            preprocess_function,
            batched=True,
            remove_columns=column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on Test dataset",
        )
        test_dataloader = DataLoader(
            test_dataset,
            collate_fn=data_collator,
            batch_size=args.batch_size
        )
        model, test_dataloader = accelerator.prepare(
                model, test_dataloader
        )
        predictions, _ = validation(test_dataloader)
        predictions = [{"title": p, "id": i} for i, p in zip(raw_datasets["test"]["id"], predictions)]
        with open(args.prediction_path, "w") as json_file:
            for line in predictions:
                json_line = json.dumps(line)
                json_file.write(json_line + "\n")
        print(f"\nThe prediction results have been saved to {args.prediction_path}")

if __name__ == "__main__":
    main()
