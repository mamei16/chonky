from typing import Optional, Union
from dataclasses import field, dataclass

import numpy as np
import torch
import evaluate
from datasets import load_from_disk, load_dataset, Dataset
import datasets
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification, is_torch_xla_available,
)
from transformers.trainer_utils import SaveStrategy

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm


torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True


DATASET_NAMES = ["bn", "bpy", "zh", "zh-classical", "ht", "ky", "new", "nn", "no", "fa", "fy", "af", "sq", "ar", "an",
                 "hy", "ast", "az", "ba", "eu", "bar", "be", "bs", "br", "bg", "my", "ca", "ce", "cv", "hr", "cs", "da",
                 "nl", "en", "et", "fi", "fr", "gl", "ka", "de", "el", "gu", "he", "hi", "hu", "is", "io", "id", "ga",
                 "it", "ja", "jv", "kn", "kk", "ko", "la", "lv", "lt", "lmo", "nds-nl", "lb", "mk", "mg", "ms", "ml",
                 "mr", "min", "ne", "oc", "pms", "pl", "pt", "pa", "ro", "ru", "sco", "sr", "sh", "scn", "sk", "sl",
                 "azb", "es", "su", "sw", "tl", "tg", "ta", "tt", "te", "tr", "uk", "ur", "uz", "vi", "vo", "cy", "pnb",
                 "yo", "th", "mn", "ceb", "war", "sv"]

FULL_DATASET_NAMES = ["Bangla", "Bishnupriya", "Chinese", "Classical Chinese", "Haitian Creole", "Kyrgyz", "Newari",
                      "Norwegian Nynorsk", "Norwegian", "Persian", "Western Frisian", "Afrikaans", "Albanian", "Arabic",
                      "Aragonese", "Armenian", "Asturian", "Azerbaijani", "Bashkir", "Basque", "Bavarian", "Belarusian",
                      "Bosnian", "Breton", "Bulgarian", "Burmese", "Catalan", "Chechen", "Chuvash", "Croatian", "Czech",
                      "Danish", "Dutch", "English", "Estonian", "Finnish", "French", "Galician", "Georgian", "German",
                      "Greek", "Gujarati", "Hebrew", "Hindi", "Hungarian", "Icelandic", "Ido", "Indonesian", "Irish",
                      "Italian", "Japanese", "Javanese", "Kannada", "Kazakh", "Korean", "Latin", "Latvian", "Lithuanian",
                      "Lombard", "Low Saxon", "Luxembourgish", "Macedonian", "Malagasy", "Malay", "Malayalam", "Marathi",
                      "Minangkabau", "Nepali", "Occitan", "Piedmontese", "Polish", "Portuguese", "Punjabi", "Romanian",
                      "Russian", "Scots", "Serbian", "Serbo-Croatian", "Sicilian", "Slovak", "Slovenian",
                      "South Azerbaijani", "Spanish", "Sundanese", "Swahili", "Tagalog", "Tajik", "Tamil", "Tatar",
                      "Telugu", "Turkish", "Ukrainian", "Urdu", "Uzbek", "Vietnamese", "Volapük", "Welsh",
                      "Western Punjabi", "Yoruba", "Thai", "Mongolian", "Cebuano", "Waray", "Swedish"]


def distilbert(model_name):
    id2label = {
        0: "O",
        1: "separator",
    }
    label2id = {
        "O": 0,
        "separator": 1,
    }

    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=2, id2label=id2label, label2id=label2id
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


def modernbert(model_id):
    id2label = {
        0: "O",
        1: "separator",
    }
    label2id = {
        "O": 0,
        "separator": 1,
    }

    model = AutoModelForTokenClassification.from_pretrained(
        model_id,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        _attn_implementation="flash_attention_2",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    return model, tokenizer

@dataclass
class CustomEvalTrainingArguments(TrainingArguments):
    full_eval_steps: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Run an evaluation on a extra validation set every X steps. "
                "Should be an integer."
            )
        },
    )

class CustomEvalTrainer(Trainer):
    def __init__(self, full_eval_dataset: Optional[Union[Dataset, dict[str, Dataset], "datasets.Dataset"]] = None,
                 **kwargs):
        self.full_eval_dataset = full_eval_dataset
        super().__init__(**kwargs)

    def _maybe_log_save_evaluate(
            self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None
    ):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)

        if (self.full_eval_dataset and self.args.full_eval_steps and
            self.state.global_step % self.args.full_eval_steps == 0):
            metrics = self.evaluate(eval_dataset=self.full_eval_dataset)

            f1_scores = np.array([metrics[metric] for metric in metrics.keys() if metric.endswith("f1")])
            mean_f1 = np.mean(f1_scores)
            std_f1 = np.std(f1_scores)
            median_f1 = np.median(f1_scores)

            accumulated_metrics = {"eval_acc_mean_f1": mean_f1.item(),
                                   "eval_acc_std_f1": std_f1.item(),
                                   "eval_acc_median_f1": median_f1.item()}
            self.log(accumulated_metrics)

            is_new_best_metric = self._determine_best_metric(metrics=accumulated_metrics, trial=trial)

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric


        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


def main(dataset_id, model_id, output_dir, batch_size):
    #model, tokenizer = modernbert(model_id=model_id)
    model, tokenizer = distilbert(model_name=model_id)

    dataset_train = load_dataset(dataset_id, "all_combined_train", num_proc=16)["train"]
    print(dataset_train)

    fast_eval_dataset_dict = {}
    full_eval_dataset_dict = {}
    for language_code, full_name in zip(DATASET_NAMES, FULL_DATASET_NAMES):
        fast_eval_dataset_dict[f"fast_{full_name}"] = load_dataset(dataset_id, language_code, split="fast_val")
        full_eval_dataset_dict[f"full_{full_name}"] = load_dataset(dataset_id, language_code, split="full_val")

    label_list = ["O", "separator"]
    seqeval = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }


    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    training_args = CustomEvalTrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=1,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=1000,
        full_eval_steps=10000,
        save_strategy="steps",
        save_steps=5000,
        load_best_model_at_end=True,
        push_to_hub=True,
        hub_strategy="every_save",
        hub_token="",
        fp16=True,
        eval_on_start=True,
        save_total_limit=5,
        save_only_model=True,
        metric_for_best_model="eval_acc_mean_f1",
        label_smoothing_factor=0.1,
        report_to=["tensorboard"],
        torch_compile=True,
        dataloader_num_workers=2
    )

    trainer = CustomEvalTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=fast_eval_dataset_dict,
        full_eval_dataset=full_eval_dataset_dict,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    main(
        dataset_id="mamei16/multilingual-wikipedia-paragraphs",
        model_id="distilbert/distilbert-base-multilingual-cased",
        output_dir="chonky_distilbert-base-multilingual-cased",
        batch_size=64
    )
