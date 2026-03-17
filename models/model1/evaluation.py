import numpy as np
import evaluate

# Load the standard BLEU metric
metric = evaluate.load("bleu")

def get_compute_metrics_fn(tokenizer):
    """
    Returns a function that can be passed to Seq2SeqTrainer.
    The tokenizer is needed to decode the predictions back to text.
    """
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Decode predictions and labels into text
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some post-processing: sacrebleu expects a list of lists for references
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        # Compute BLEU-4 (standard)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        
        # Extract individual n-gram precisions
        # 'precisions' is a list: [p1, p2, p3, p4]
        precisions = result.get("precisions", [0, 0, 0, 0])
        
        return {
            "bleu": result["bleu"] * 100, # Overall BLEU-4
            "bleu-1": precisions[0] * 100,
            "bleu-2": precisions[1] * 100,
            "bleu-3": precisions[2] * 100,
            "bleu-4": precisions[3] * 100,
        }
    
    return compute_metrics
