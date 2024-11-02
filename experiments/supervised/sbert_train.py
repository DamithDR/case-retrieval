import argparse

from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sentence_transformers.losses import OnlineContrastiveLoss


def run(model_name):
    # 1. Load a model to finetune with 2. (Optional) model card data
    model = SentenceTransformer(
        model_name,
    )

    # 3. Load a dataset to finetune on
    train_candidates = load_dataset('Exploration-Lab/IL-TUR', "pcr", split='train_candidates')
    train_dataset = load_dataset("csv", data_files="data/prepare/ilpcr.csv")
    # train_dataset = dataset["train"].select(range(100_000))
    eval_dataset = load_dataset("csv", data_files="data/prepare/ilpcr_dev.csv")
    # test_dataset = dataset["test"]

    # 4. Define a loss function
    loss = OnlineContrastiveLoss(model)

    # 5. (Optional) Specify training arguments
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir="outputs/",
        # Optional training parameters:
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=100,
        run_name=f"legal-bert-sbert-ilpcr",  # Will be used in W&B if `wandb` is installed
    )

    dev_evaluator = BinaryClassificationEvaluator(
        sentences1=eval_dataset['train']["reference"],
        sentences2=eval_dataset['train']["candidate"],
        labels=eval_dataset['train']["label"],
        name="legal-bert-sbert-ilpcr_dev",
        batch_size=16,
        show_progress_bar=True,
    )
    dev_evaluator(model)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=dev_evaluator,
    )
    trainer.train()

    model.save_pretrained("outputs/final")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''case vectoriser arguments''')
    parser.add_argument('--model_name', type=str, default="nlpaueb/legal-bert-base-uncased", required=False,
                        help='model_name')
    args = parser.parse_args()
    run(args.model_name)
