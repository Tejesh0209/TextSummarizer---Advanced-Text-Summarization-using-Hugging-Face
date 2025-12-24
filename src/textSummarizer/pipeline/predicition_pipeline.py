from src.textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer
from transformers import pipeline
import os


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()


    
    def predict(self, text):
        # Check if trained model and tokenizer exist locally
        model_exists = os.path.exists(self.config.model_path)
        tokenizer_exists = os.path.exists(self.config.tokenizer_path)
        
        if model_exists and tokenizer_exists:
            # Load from local trained model
            tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path, local_files_only=True)
            model_path = self.config.model_path
        else:
            # Fallback to base pretrained model from HuggingFace
            tokenizer = AutoTokenizer.from_pretrained("google/pegasus-cnn_dailymail")
            model_path = "google/pegasus-cnn_dailymail"
            print(f"Warning: Trained model not found. Using base model 'google/pegasus-cnn_dailymail'")
        
        gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}

        pipe = pipeline("summarization", model=model_path, tokenizer=tokenizer)

        print("Dialogue:")
        print(text)

        output = pipe(text, **gen_kwargs)[0]["summary_text"]
        print("\nModel Summary:")
        print(output)

        return output