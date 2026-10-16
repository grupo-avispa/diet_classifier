import argparse
import torch
import os
import time
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from diet_classifier.training import DataLoader
from diet_classifier.model import DIETModel, SparseFeatureExtractor
import langs
from langs import load_json

def main():
    parser = argparse.ArgumentParser(description="Validate the DIET model of a language")
    parser.add_argument("--lang", required=True, help="Language code (en es fr de it ...)")
    parser.add_argument("--no-plot", action="store_true", help="Do not show the confusion matrix")
    parser.add_argument("--device", default="cpu", choices=["cpu", "auto", "cuda"],
                        help="Device to use (default: cpu; auto: CUDA only if it works, else CPU)")
    args = parser.parse_args()
    lang = args.lang
    files = langs.model_files(lang)
    # Parameters
    batch_size = 1
    device = langs.pick_device(args.device)
    print(f"Using device: {device}")
    # Load data
    data_path = os.path.join(langs.data_dir(lang), "validation.yml")
    entity_labels = load_json(files["entity_labels"])
    intent_labels = load_json(files["intent_labels"])

    # Initialize DataLoader
    data_loader = DataLoader(data_path=data_path, 
                             batch_size=batch_size,
                             intent_labels=intent_labels,
                             entity_labels=entity_labels,
                             cls_token="[CLS]",
                             pad_token="[PAD]",
                             pad_entity_tag="PAD")
    
    # Create SparseFeatureExtractor instance
    cfg = langs.load_model_config(lang)
    sparse_extractor = SparseFeatureExtractor(
        word_dict_size=cfg["word_dict_size"],
        ngram_dict_size=cfg["ngram_dict_size"],
        ngram_overflow_size=cfg["ngram_overflow_size"],
        ngram_min=cfg["ngram_min"],
        ngram_max=cfg["ngram_max"],
        pad_token="[PAD]",
        cls_token="[CLS]",
        unk_token="[UNK]"
    )

    # Load dictionaries
    sparse_extractor.load_dicts(files["word_dict"], files["ngram_dict"])

    # Initialize DIET model
    model = DIETModel(
        device=device,
        sparse_extractor=sparse_extractor,
        num_entity_tags=len(entity_labels),
        num_intent_tags=len(intent_labels),
        pad_entity_tag_idx=entity_labels.index("PAD"),
        eos_entity_tag_idx=entity_labels.index("EOS"),
        bos_entity_tag_idx=entity_labels.index("BOS")
    )

    # Test saved model
    model.load_state_dict(torch.load(files["model"], map_location=device))
    model.to(device)  # Move model to the correct device
    model.eval()

    # Print Classification Info
    print("--"*30)
    print("Available intents:", intent_labels)
    print("Available entity tags:", entity_labels)
    print("--"*30)
    
    # For entity F1 score (excluding PAD, BOS, EOS, O)
    entity_true_positives = {label: 0 for label in entity_labels if label not in ["PAD", "BOS", "EOS", "O"]}
    entity_false_positives = {label: 0 for label in entity_labels if label not in ["PAD", "BOS", "EOS", "O"]}
    entity_false_negatives = {label: 0 for label in entity_labels if label not in ["PAD", "BOS", "EOS", "O"]}
    
    total_samples = 0
    correct_intents = 0

    # Do inference on validation data
    # Compute F1 score for entities and accuracy for intents
    # and confusion matrix for intents
    entity_preds = []
    intent_preds = []
    entity_true = []
    intent_true = []
    for batch in data_loader.data:
        # 1. Fromat the batch
        (text_inputs, entity_tag_indices, 
         one_hot_intent_labels) = data_loader.format_batch(batch)
        # Do inference
        init_time = time.perf_counter()
        with torch.no_grad():
            tensor_entities, tensor_intent = model(text_inputs)
        end_time = time.perf_counter()
        print(f"Inference time for batch: {(end_time - init_time)*1000:.2f} ms")
        
        # Process each sample in batch
        for b, sentence in enumerate(text_inputs):
            # Get predicted and true intents
            predicted_intent_idx = torch.argmax(tensor_intent[b]).item()
            true_intent_idx = torch.argmax(one_hot_intent_labels[b]).item()
            intent_preds.append(predicted_intent_idx)
            intent_true.append(true_intent_idx)
            
            total_samples += 1
            
            if predicted_intent_idx == true_intent_idx:
                correct_intents += 1
            
            # Get predicted and true entity tags
            predicted_entities = tensor_entities[b].tolist()
            true_entities = entity_tag_indices[b]
            entity_preds.append(predicted_entities)
            entity_true.append(true_entities.tolist())
            
            # Calculate entity metrics (excluding special tokens)
            for i, (pred_idx, true_idx) in enumerate(zip(predicted_entities, true_entities)):
                pred_label = entity_labels[pred_idx]
                true_label = entity_labels[true_idx]
                
                # Skip special tokens
                if true_label in ["PAD", "BOS", "EOS", "O"] and pred_label in ["PAD", "BOS", "EOS", "O"]:
                    continue
                
                # Count entity matches
                if true_label not in ["PAD", "BOS", "EOS", "O"]:
                    if pred_label == true_label:
                        entity_true_positives[true_label] += 1
                    else:
                        entity_false_negatives[true_label] += 1
                        if pred_label not in ["PAD", "BOS", "EOS", "O"]:
                            entity_false_positives[pred_label] += 1
                elif pred_label not in ["PAD", "BOS", "EOS", "O"]:
                    entity_false_positives[pred_label] += 1
            
            # Print results
            print(f"Sentence: '{sentence}'")
            print("Predicted entity tags:", [entity_labels[idx] for idx in predicted_entities])
            print("True entity tags:", [entity_labels[idx] for idx in true_entities])
            print("Predicted intent:", intent_labels[predicted_intent_idx])
            print("True intent:", intent_labels[true_intent_idx])
            print("--"*30)

    # Calculate and print metrics
    print("\n" + "=="*30)
    print("VALIDATION METRICS")
    print("=="*30)
    
    # -----------------------------------------------------------------------------------

    # Intent accuracy
    intent_accuracy = (correct_intents / total_samples) * 100 if total_samples > 0 else 0
    print(f"\nIntent Accuracy: {intent_accuracy:.2f}% ({correct_intents}/{total_samples})")

    # -----------------------------------------------------------------------------------
    
    # Entity F1 Score
    print("\n" + "--"*30)
    print("Entity Recognition Metrics (F1 Score):")
    print("--"*30)
    
    overall_tp = 0
    overall_fp = 0
    overall_fn = 0
    
    for label in entity_true_positives.keys():
        tp = entity_true_positives[label]
        fp = entity_false_positives[label]
        fn = entity_false_negatives[label]
        
        overall_tp += tp
        overall_fp += fp
        overall_fn += fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{label:20s} - Precision: {precision:.3f}, Recall: {recall:.3f}," + 
              f"F1: {f1:.3f} (TP:{tp}, FP:{fp}, FN:{fn})")
    
    # -----------------------------------------------------------------------------------
    
    # Confusion Matrix for Intents
    print("\nConfusion Matrix for Intents:")
    confusion_matrix_intent = confusion_matrix(intent_true, 
                                               intent_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_intent, 
                                  display_labels=intent_labels)
    disp.plot(cmap='Blues')
    if not args.no_plot:
        plt.show()

if __name__ == "__main__":
    main()