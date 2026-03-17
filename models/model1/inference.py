import torch
from transformers import EncoderDecoderModel, BertTokenizer

class Translator:
    def __init__(self, model_folder):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading model from {model_folder}...")
        # This loads both the encoder and decoder configuration automatically
        self.tokenizer = BertTokenizer.from_pretrained(model_folder)
        self.model = EncoderDecoderModel.from_pretrained(model_folder).to(self.device)
        self.model.eval()

    def translate(self, text, max_length=100):
        # 1. Tokenize the input text
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=max_length
        ).to(self.device)

        # 2. Generate output using Beam Search (matching your training validation)
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                num_beams=4,
                length_penalty=0.6,
                no_repeat_ngram_size=3,
                early_stopping=True,
                decoder_start_token_id=self.model.config.decoder_start_token_id,
                pad_token_id=self.model.config.pad_token_id,
                eos_token_id=self.model.config.eos_token_id
            )

        # 3. Decode back to strings
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

# ==========================================
# INTERACTIVE INTERFACE
# ==========================================
if __name__ == "__main__":
    # Point this to the folder where model.save_pretrained() was called
    MODEL_PATH = "final_model" 
    
    try:
        translator = Translator(MODEL_PATH)
        print("\n" + "="*40)
        print("🌍 BERT-to-BERT TRANSLATOR READY")
        print("Type 'quit' or 'q' to exit.")
        print("="*40 + "\n")
        
        while True:
            text = input("Source Text: ").strip()
            if not text or text.lower() in ['q', 'quit']:
                break
                
            prediction = translator.translate(text)
            print(f"Prediction: {prediction}\n")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"Check if {MODEL_PATH} contains config.json and pytorch_model.bin")
