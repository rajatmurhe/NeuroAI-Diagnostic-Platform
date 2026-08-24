# src/train_bert.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, classification_report
from preprocess import get_train_test_splits, serialize_rows

MODEL_CHECKPOINT = "bert-base-uncased"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../models/bert_alzheimer_model")

class AlzheimerDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item

def train_bert_classifier():
    X_train, X_test, _, _, y_train, y_test, y_train_enc, y_test_enc, le = get_train_test_splits()
    
    # 1. Serialize features into sentences
    train_texts = serialize_rows(X_train)
    test_texts = serialize_rows(X_test)
    
    # 2. Tokenize inputs
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)
    
    train_dataset = AlzheimerDataset(train_encodings, y_train_enc.tolist())
    test_dataset = AlzheimerDataset(test_encodings, y_test_enc.tolist())
    
    # 3. Initialize BERT for sequence classification
    num_labels = len(le.classes_)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=num_labels)
    
    training_args = TrainingArguments(
        output_dir="./results_bert",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=15,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
    )
    
    print("\n--- Training Fine-Tuned BERT ---")
    trainer.train()
    
    # 4. Evaluate on Test Set
    preds = trainer.predict(test_dataset)
    y_pred = preds.predictions.argmax(axis=1)
    
    print("\nBERT Test Set Performance:")
    print(f"Accuracy: {accuracy_score(y_test_enc, y_pred):.4f}")
    print(f"Weighted F1: {f1_score(y_test_enc, y_pred, average='weighted'):.4f}")
    print(classification_report(y_test_enc, y_pred, target_names=le.classes_))
    
    # 5. Save model and tokenizer
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Fine-tuned BERT saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train_bert_classifier()