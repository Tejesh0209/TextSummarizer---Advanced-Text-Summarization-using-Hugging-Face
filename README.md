# TextSummarizer - Advanced Text Summarization using Hugging Face

A production-ready, modular, and scalable text summarization application built with **Hugging Face Transformers**, **PyTorch**, and **FastAPI**. This project implements the **PEGASUS model** fine-tuned on the SAMSum dataset for abstractive text summarization.

---

##  Project Overview

This project demonstrates a complete end-to-end machine learning pipeline with:

- **Data Ingestion**: Automated dataset downloading and extraction
- **Data Transformation**: Tokenization and preprocessing using Hugging Face tokenizers
- **Model Training**: Fine-tuning PEGASUS on abstractive summarization tasks
- **Model Evaluation**: ROUGE metric computation for quality assessment
- **REST API**: FastAPI endpoints for training and prediction
- **Modular Architecture**: Clean separation of concerns following best practices

### Model Architecture

- **Base Model**: `google/pegasus-cnn_dailymail` - Pre-trained on CNN/DailyMail dataset
- **Task**: Abstractive Text Summarization (Seq2Seq)
- **Dataset**: SAMSum - 16k multi-turn messenger-like conversations with human-annotated summaries
- **Evaluation Metrics**: ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum

---

##  Project Structure

```
textsummarizer/
├── app.py                          # FastAPI application entry point
├── main.py                         # Training pipeline orchestrator
├── Dockerfile                      # Docker containerization
├── requirements.txt                # Python dependencies
├── params.yaml                     # Training hyperparameters
├── config/
│   └── config.yaml                # Configuration for all stages
├── src/textSummarizer/
│   ├── __init__.py
│   ├── logging/                   # Logging configuration
│   │   └── __init__.py
│   ├── entity/                    # Data classes for config objects
│   │   └── __init__.py
│   ├── constants/                 # Application constants
│   │   └── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── configuration.py       # Configuration Manager (DI pattern)
│   ├── components/                # Core ML components
│   │   ├── __init__.py
│   │   ├── data_ingestion.py      # Stage 1: Data download & extraction
│   │   ├── data_transformation.py # Stage 2: Tokenization & preprocessing
│   │   ├── model_trainer.py       # Stage 3: Model fine-tuning
│   │   └── model_evaluation.py    # Stage 4: Model evaluation
│   ├── pipeline/                  # Pipeline orchestrators
│   │   ├── __init__.py
│   │   ├── stage_1_data_ingestion_pipeline.py
│   │   ├── stage_2_data_transformation_pipeline.py
│   │   ├── stage_3_model_trainer_pipeline.py
│   │   ├── stage_4_model_evaluation.py
│   │   └── predicition_pipeline.py # Inference pipeline
│   └── utils/
│       ├── __init__.py
│       └── common.py              # Utility functions
├── research/                      # Jupyter notebooks for experimentation
├── artifacts/                     # Generated outputs
│   ├── data_ingestion/
│   ├── data_transformation/
│   ├── model_trainer/            # Fine-tuned model & tokenizer
│   └── model_evaluation/         # Evaluation metrics
└── logs/                          # Application logs
```

---

## Modular Architecture & Design Patterns

### 1. **Dependency Injection Pattern**
The project uses the **Configuration Manager** pattern to inject dependencies:

```python
class ConfigurationManager:
    def __init__(self, config_path=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_filepath)
```

### 2. **Dataclass-based Configuration Objects**
All configuration objects are defined as Python dataclasses:

```python
@dataclass
class DataIngestionConfig:
    root_dir: Path
    source_URL: Path
    local_data_file: Path
    unzip_dir: Path

@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    tokenizer_name: Path

@dataclass
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    model_ckpt: Path
    num_train_epochs: int
    # ... other hyperparameters

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    data_path: Path
    model_path: Path
    tokenizer_path: Path
    metric_file_name: Path
```

### 3. **Pipeline Pattern**
Each stage is organized as a separate pipeline:

```python
class DataIngestionTrainingPipeline:
    def initiate_data_ingestion(self):
        config = ConfigurationManager()
        config_obj = config.get_data_ingestion_config()
        component = DataIngestion(config=config_obj)
        component.downlaod_file()
        component.extract_zip_file()
```

### 4. **Component-based Architecture**
Each ML stage is a separate component with single responsibility:

- `DataIngestion`: Handle data download and extraction
- `DataTransformation`: Tokenization and preprocessing
- `ModelTrainer`: Fine-tuning and model training
- `ModelEvaluation`: Evaluation and metrics computation

---

## Configuration Files

### config/config.yaml
```yaml
artifacts_root: artifacts

data_ingestion:
  root_dir: artifacts/data_ingestion
  source_URL: https://github.com/krishnaik06/datasets/raw/refs/heads/main/summarizer-data.zip
  local_data_file: artifacts/data_ingestion/data.zip
  unzip_dir: artifacts/data_ingestion

data_transformation:
  root_dir: artifacts/data_transformation
  data_path: artifacts/data_ingestion/samsum_dataset
  tokenizer_name: google/pegasus-cnn_dailymail

model_trainer:
  root_dir: artifacts/model_trainer
  data_path: artifacts/data_transformation/samsum_dataset
  model_ckpt: google/pegasus-cnn_dailymail

model_evaluation:
  root_dir: artifacts/model_evaluation
  data_path: artifacts/data_transformation/samsum_dataset
  model_path: artifacts/model_trainer/pegasus-samsum-model
  tokenizer_path: artifacts/model_trainer/tokenizer
  metric_file_name: artifacts/model_evaluation/metrics.csv
```

### params.yaml
```yaml
TrainingArguments:
  num_train_epochs: 1
  warmup_steps: 500
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  weight_decay: 0.01
  logging_steps: 10
  evaluation_strategy: steps
  eval_steps: 500
  save_steps: 1e6
  gradient_accumulation_steps: 16
```

---

## Installation & Setup

### Prerequisites
- Python 3.9+
- pip or conda
- Virtual environment (recommended)

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/textsummarizer.git
cd textsummarizer
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

---

## Core Components Explained

### Stage 1: Data Ingestion
**File**: `src/textSummarizer/components/data_ingestion.py`

Responsible for:
- Downloading datasets from remote URLs
- Extracting zip files
- Organizing data directories

```python
class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def downlaod_file(self):
        """Download dataset if not already present"""
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_file
            )

    def extract_zip_file(self):
        """Extract downloaded zip file"""
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(self.config.unzip_dir)
```

**Output**: `artifacts/data_ingestion/samsum_dataset/`

---

### Stage 2: Data Transformation
**File**: `src/textSummarizer/components/data_transformation.py`

Responsible for:
- Loading tokenizer from Hugging Face
- Converting text examples to token sequences
- Creating training-ready datasets

```python
class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def convert_examples_to_features(self, example_batch):
        """Convert dialogue and summary to token IDs"""
        input_encodings = self.tokenizer(
            example_batch['dialogue'],
            max_length=1024,
            truncation=True
        )
        
        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(
                example_batch['summary'],
                max_length=128,
                truncation=True
            )
        
        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }

    def convert(self):
        """Process entire dataset using map transformation"""
        dataset_samsum = load_from_disk(self.config.data_path)
        dataset_samsum_pt = dataset_samsum.map(
            self.convert_examples_to_features,
            batched=True
        )
        dataset_samsum_pt.save_to_disk(
            os.path.join(self.config.root_dir, "samsum_dataset")
        )
```

**Key Features**:
- **Tokenizer**: `google/pegasus-cnn_dailymail` (SentencePiece-based)
- **Max Input Length**: 1024 tokens
- **Max Target Length**: 128 tokens
- **Batch Processing**: Efficient dataset transformation

**Output**: `artifacts/data_transformation/samsum_dataset/`

---

### Stage 3: Model Training
**File**: `src/textSummarizer/components/model_trainer.py`

Responsible for:
- Loading pre-trained PEGASUS model
- Fine-tuning on SAMSum dataset
- Saving model and tokenizer

```python
class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_ckpt
        ).to(device)
        
        # Data collator for seq2seq tasks
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        
        # Load preprocessed dataset
        dataset = load_from_disk(self.config.data_path)
        
        # Training arguments
        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=self.config.num_train_epochs,
            warmup_steps=self.config.warmup_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            evaluation_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=trainer_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            train_dataset=dataset["test"],
            eval_dataset=dataset["validation"]
        )
        
        # Train model
        trainer.train()
        
        # Save fine-tuned model and tokenizer
        model.save_pretrained(
            os.path.join(self.config.root_dir, "pegasus-samsum-model")
        )
        tokenizer.save_pretrained(
            os.path.join(self.config.root_dir, "tokenizer")
        )
```

**Training Configuration**:
- **Optimizer**: AdamW (default in Transformers)
- **Learning Rate**: 5e-5 (default)
- **Batch Size**: 1 (per device)
- **Gradient Accumulation**: 16 steps (effective batch size: 16)
- **Training Epochs**: 1
- **Warmup Steps**: 500

**Output**: 
- `artifacts/model_trainer/pegasus-samsum-model/`
- `artifacts/model_trainer/tokenizer/`

---

### Stage 4: Model Evaluation
**File**: `src/textSummarizer/components/model_evaluation.py`

Responsible for:
- Computing ROUGE metrics on test set
- Generating model summaries
- Saving evaluation results

```python
class ModelEvaluation:
    def calculate_metric_on_test_ds(self, dataset, metric, model, tokenizer,
                                    batch_size=16, device="cuda",
                                    column_text="article",
                                    column_summary="highlights"):
        """Calculate metrics on test dataset"""
        article_batches = list(
            self.generate_batch_sized_chunks(dataset[column_text], batch_size)
        )
        target_batches = list(
            self.generate_batch_sized_chunks(dataset[column_summary], batch_size)
        )
        
        for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches),
            total=len(article_batches)
        ):
            # Tokenize inputs
            inputs = tokenizer(
                article_batch,
                max_length=1024,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Generate summaries
            summaries = model.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                length_penalty=0.8,
                num_beams=8,
                max_length=128
            )
            
            # Decode summaries
            decoded_summaries = [
                tokenizer.decode(s, skip_special_tokens=True,
                                clean_up_tokenization_spaces=True)
                for s in summaries
            ]
            
            # Add to metric
            metric.add_batch(
                predictions=decoded_summaries,
                references=target_batch
            )
        
        return metric.compute()

    def evaluate(self):
        """Run full evaluation pipeline"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_path
        ).to(device)
        
        # Load dataset
        dataset = load_from_disk(self.config.data_path)
        
        # Compute ROUGE metrics
        rouge_metric = evaluate.load('rouge')
        score = self.calculate_metric_on_test_ds(
            dataset['test'][0:10],
            rouge_metric,
            model,
            tokenizer,
            batch_size=2,
            column_text='dialogue',
            column_summary='summary'
        )
        
        # Save results
        rouge_dict = {
            rn: score[rn] for rn in ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        }
        df = pd.DataFrame(rouge_dict, index=['pegasus'])
        df.to_csv(self.config.metric_file_name, index=False)
```

**Evaluation Metrics**:
- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap
- **ROUGE-L**: Longest common subsequence
- **ROUGE-Lsum**: Summary-level LCS

**Generation Parameters**:
- **Beam Search**: 8 beams (more accurate but slower)
- **Length Penalty**: 0.8 (encourages shorter summaries)
- **Max Length**: 128 tokens

**Output**: `artifacts/model_evaluation/metrics.csv`

---

## 🔧 Inference Pipeline

**File**: `src/textSummarizer/pipeline/predicition_pipeline.py`

```python
class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

    def predict(self, text):
        """Generate summary for input text"""
        # Check if trained model exists
        model_exists = os.path.exists(self.config.model_path)
        tokenizer_exists = os.path.exists(self.config.tokenizer_path)
        
        if model_exists and tokenizer_exists:
            # Use fine-tuned model
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.tokenizer_path,
                local_files_only=True
            )
            model_path = self.config.model_path
        else:
            # Fallback to pre-trained model
            tokenizer = AutoTokenizer.from_pretrained(
                "google/pegasus-cnn_dailymail"
            )
            model_path = "google/pegasus-cnn_dailymail"
        
        # Generation parameters
        gen_kwargs = {
            "length_penalty": 0.8,
            "num_beams": 8,
            "max_length": 128
        }
        
        # Create pipeline and generate summary
        pipe = pipeline("summarization", model=model_path, tokenizer=tokenizer)
        output = pipe(text, **gen_kwargs)[0]["summary_text"]
        
        return output
```

---

## 🌐 REST API

**File**: `app.py`

### Endpoints

#### 1. Root Endpoint
```
GET /
Redirects to: /docs (Swagger UI)
```

#### 2. Training Endpoint
```
GET /train
Description: Trigger the complete training pipeline
Response: "Training successful !!"
Note: Runs main.py which executes all 4 stages
```

#### 3. Prediction Endpoint
```
POST /predict?text=<your_text>
Description: Generate summary for input text
Parameters:
  - text (str): Input text to summarize
Response: Summarized text
```

### Example Usage

```bash
# Start server
python app.py
# Server runs on http://0.0.0.0:8080

# Interactive API docs available at:
# http://localhost:8080/docs

# Trigger training
curl -X GET "http://localhost:8080/train"

# Generate summary
curl -X POST "http://localhost:8080/predict?text=Your%20long%20text%20here"
```

---

## 📚 Key Technologies & Libraries

### Deep Learning & NLP
- **transformers** (4.57.3): State-of-the-art transformer models
- **torch** (2.9.1): PyTorch - Deep learning framework
- **datasets** (4.4.2): Hugging Face datasets library
- **tokenizers** (0.22.1): Fast tokenization

### Training & Evaluation
- **evaluate** (0.4.6): Evaluation metrics (ROUGE, BLEU, etc.)
- **sacrebleu** (2.5.1): Machine translation metrics
- **rouge_score** (0.1.2): ROUGE metric implementation

### Web Framework
- **fastapi** (0.78.0): Modern async web framework
- **uvicorn** (0.18.3): ASGI server
- **starlette** (0.19.1): Lightweight ASGI toolkit

### Data & Configuration
- **datasets** (4.4.2): Dataset manipulation
- **pandas** (2.3.3): Data analysis
- **PyYAML** (6.0.3): YAML parsing
- **python-box** (6.0.2): Dot notation for dicts

### Utilities
- **tqdm** (4.67.1): Progress bars
- **nltk** (3.9.2): Natural language toolkit
- **matplotlib** (3.10.8): Visualization

---

## Utility Functions

**File**: `src/textSummarizer/utils/common.py`

```python
@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Read YAML file and return as ConfigBox (dot notation access)
    
    Args:
        path_to_yaml: Path to YAML file
        
    Raises:
        ValueError: If YAML file is empty
        
    Returns:
        ConfigBox: Configuration object with dot notation access
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """
    Create list of directories with logging
    
    Args:
        path_to_directories: List of directory paths
        verbose: Whether to log directory creation
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")
```

---

## 📝 Logging System

**File**: `src/textSummarizer/logging/__init__.py`

```python
import logging
import os
import sys

log_dir = "logs"
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"
log_filepath = os.path.join(log_dir, "continuos_logs.log")

os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("summarizerlogger")
```

**Features**:
- Console output (stdout)
- File output (`logs/continuos_logs.log`)
- Timestamps and module names
- Consistent formatting

---

## 🔍 Configuration Manager

**File**: `src/textSummarizer/config/configuration.py`

Implements **Dependency Injection** pattern:

```python
class ConfigurationManager:
    def __init__(self,
                 config_path=CONFIG_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        return DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([config.root_dir])
        return DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            tokenizer_name=config.tokenizer_name
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.TrainingArguments
        create_directories([config.root_dir])
        return ModelTrainerConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_ckpt=config.model_ckpt,
            num_train_epochs=params.num_train_epochs,
            warmup_steps=params.warmup_steps,
            per_device_train_batch_size=params.per_device_train_batch_size,
            weight_decay=params.weight_decay,
            logging_steps=params.logging_steps,
            evaluation_strategy=params.evaluation_strategy,
            eval_steps=params.eval_steps,
            save_steps=params.save_steps,
            gradient_accumulation_steps=params.gradient_accumulation_steps
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        create_directories([config.root_dir])
        return ModelEvaluationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_path=config.model_path,
            tokenizer_path=config.tokenizer_path,
            metric_file_name=config.metric_file_name
        )
```

---

## 🚀 Execution Workflow

### 1. Complete Training Pipeline (main.py)
```
Stage 1: Data Ingestion
├── Download dataset from URL
└── Extract zip file

Stage 2: Data Transformation
├── Load tokenizer (google/pegasus-cnn_dailymail)
├── Tokenize dialogue and summaries
└── Save transformed dataset

Stage 3: Model Training
├── Load pre-trained PEGASUS model
├── Configure training arguments
├── Fine-tune on SAMSum dataset
├── Save model and tokenizer

Stage 4: Model Evaluation
├── Load fine-tuned model
├── Generate summaries on test set
├── Compute ROUGE metrics
└── Save metrics to CSV
```

### 2. Inference (app.py)
```
API Request → PredictionPipeline
├── Load model & tokenizer
├── Tokenize input text
├── Generate summary (beam search)
└── Return summary
```

---

## 🎯 Training Strategy & Fine-tuning Details

### Pre-trained Model: PEGASUS
- **Architecture**: Transformer-based Seq2Seq model
- **Pre-training Objective**: Gap-sentence generation on large corpora
- **Base Checkpoint**: google/pegasus-cnn_dailymail (already trained on CNN/DailyMail)

### Fine-tuning on SAMSum
- **Dataset**: 16,000 multi-turn messenger conversations with summaries
- **Train/Val/Test Split**: ~14,732 / 819 / 818
- **Strategy**: Transfer learning from CNN/DailyMail domain

### Hyperparameter Tuning
```yaml
num_train_epochs: 1           # Single epoch for quick convergence
warmup_steps: 500             # Linear warmup schedule
learning_rate: 5e-5           # Default for fine-tuning
batch_size: 16                # 1 per device × 16 gradient accumulation
weight_decay: 0.01            # L2 regularization
evaluation_strategy: steps    # Eval every 500 steps
gradient_accumulation_steps: 16  # Simulate larger batch size
```

---

## 📊 Model Generation Parameters

```python
gen_kwargs = {
    "length_penalty": 0.8,    # Penalty for longer sequences
    "num_beams": 8,           # Beam search width
    "max_length": 128         # Maximum output tokens
}
```

### Why These Parameters?
- **Beam Search (8)**: Balances quality and speed
- **Length Penalty (0.8)**: Encourages concise summaries
- **Max Length (128)**: Typical summary length

---

## 🐳 Docker Support

**File**: `Dockerfile`

Containerize the application:
```bash
docker build -t text-summarizer:latest .
docker run -p 8080:8080 text-summarizer:latest
```

---

## 📈 Performance Considerations

### Computational Requirements
- **Training**: GPU recommended (CUDA)
- **Inference**: Can run on CPU or GPU
- **Memory**: ~4-8GB GPU VRAM for training
- **Disk**: ~5GB for dataset and models

### Optimization Techniques
- **Gradient Accumulation**: Simulate larger batches
- **Beam Search**: Trade-off between quality and speed
- **Batch Processing**: Process multiple sequences simultaneously
- **Mixed Precision**: Optional FP16 training

---

## Best Practices Implemented

1. **Configuration Management**: Externalized in YAML files
2. **Separation of Concerns**: Each component has single responsibility
3. **Dependency Injection**: ConfigurationManager pattern
4. **Type Hints**: Throughout the codebase (with @ensure_annotations)
5. **Logging**: Comprehensive logging to file and console
6. **Error Handling**: Try-except blocks with informative messages
7. **Modular Design**: Easy to extend and modify
8. **Dataclasses**: Type-safe configuration objects

---

## Learning Outcomes

This project demonstrates:
- End-to-end ML pipeline development
- Working with Hugging Face ecosystem
- Fine-tuning transformer models
- Production-ready code structure
- RESTful API design with FastAPI
- YAML-based configuration management
- ROUGE evaluation metrics
- atch processing and tokenization

---

## License

This project is open source and available under the MIT License.

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## Support

For issues, questions, or suggestions:
- Open an GitHub issue
- Check existing documentation
- Review example notebooks in `/research` directory

---

## Acknowledgments

- Hugging Face for Transformers library
- Google for PEGASUS model
- SAMSum dataset creators
- FastAPI community

---

**Happy Summarizing!**