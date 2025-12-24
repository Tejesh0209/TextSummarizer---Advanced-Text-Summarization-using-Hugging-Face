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

---

### Stage 2: Data Transformation
**File**: `src/textSummarizer/components/data_transformation.py`

Responsible for:
- Loading tokenizer from Hugging Face
- Converting text examples to token sequences
- Creating training-ready datasets



---

### Stage 3: Model Training
**File**: `src/textSummarizer/components/model_trainer.py`

Responsible for:
- Loading pre-trained PEGASUS model
- Fine-tuning on SAMSum dataset
- Saving model and tokenizer


---

##  REST API

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

## Key Technologies & Libraries

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

## Execution Workflow

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

## Training Strategy & Fine-tuning Details

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

## Model Generation Parameters

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

## Docker Support

**File**: `Dockerfile`

Containerize the application:
```bash
docker build -t text-summarizer:latest .
docker run -p 8080:8080 text-summarizer:latest
```

---

## CI/CD Pipeline with GitHub Actions

This project includes a comprehensive **Continuous Integration/Continuous Deployment (CI/CD)** pipeline using GitHub Actions.

### Workflow Overview

**File**: `.github/workflows/main.yaml`

The pipeline consists of three automated jobs:

#### 1. **Continuous Integration (Testing & Linting)**
-  Checks code quality with Black (Python formatter)
- Validates import sorting with isort
- Lints code with Flake8
- Runs unit tests
- Only proceeds if all checks pass

#### 2. **Continuous Delivery (Build & Push to ECR)**
- Builds Docker image from Dockerfile
- Tags image with commit SHA and `latest`
- Pushes to **Amazon ECR** (Elastic Container Registry)
- Creates versioned releases for rollback capability

#### 3. **Continuous Deployment (Deploy to EC2)**
- Pulls latest image from ECR
- Stops and removes old container
- Deploys new container to EC2 instance
- Verifies container is running
- Performs health check on API endpoints

### Pipeline Flow

```
GitHub Push (main branch)
        ↓
┌───────────────────────────────┐
│   Integration Tests           │ (Code quality & linting)
│   - Black formatting check    │
│   - isort import check        │
│   - Flake8 linting           │
│   - Unit tests               │
└─────────────┬─────────────────┘
              ↓ (if all pass)
┌───────────────────────────────┐
│   Build & Push to ECR         │ (Docker image)
│   - Docker build              │
│   - Tag with commit SHA       │
│   - Push to Amazon ECR        │
└─────────────┬─────────────────┘
              ↓
┌───────────────────────────────┐
│   Deploy to EC2               │ (Production)
│   - Pull from ECR             │
│   - Stop old container        │
│   - Start new container       │
│   - Health check             │
│   - API verification         │
└───────────────────────────────┘
```

### Deployment Architecture

```
┌─────────────────────┐
│  GitHub Repository  │
│   (main branch)     │
└──────────┬──────────┘
           │ (trigger on push)
           ↓
┌─────────────────────────────┐
│  GitHub Actions Workflow    │
│  - Integration Tests        │
│  - Build Docker Image       │
│  - Security Scanning        │
└──────────┬──────────────────┘
           │ (push image)
           ↓
┌─────────────────────────────┐
│  Amazon ECR Registry        │
│  (Container Image Storage)  │
└──────────┬──────────────────┘
           │ (pull & deploy)
           ↓
┌─────────────────────────────┐
│  AWS EC2 Instance           │
│  - Running Docker Container │
│  - Port 8080 exposed        │
│  - FastAPI Server           │
└─────────────────────────────┘
```

### AWS Configuration Required

Before the CI/CD pipeline works, set up the following AWS resources:

#### 1. **Create IAM Role for GitHub Actions**
### GitHub Repository Secrets

Add the following secrets to your GitHub repository (Settings → Secrets and variables):

| Secret Name | Value |
|------------|-------|
| `AWS_ACCESS_KEY_ID` | Your AWS access key |
| `AWS_SECRET_ACCESS_KEY` | Your AWS secret access key |
| `AWS_REGION` | `us-east-1` (or your region) |
| `ECR_REPOSITORY_NAME` | `text-summarizer` |

### Monitoring the Pipeline

1. **Push to main branch**:
   ```bash
   git add .
   git commit -m "Your changes"
   git push origin main
   ```

2. **Check GitHub Actions**:
   - Go to: https://github.com/Tejesh0209/TextSummarizer---Advanced-Text-Summarization-using-Hugging-Face/actions
   - Watch the workflow execute in real-time
   - View logs for each job

3. **Verify Deployment**:
   ```bash
   # SSH to EC2 instance
   ssh -i your-key.pem ec2-user@your-ec2-ip
   
   # Check running containers
   docker ps
   
   # View logs
   docker logs text-summarizer
   
   # Test API
   curl http://localhost:8080/docs
   ```

### Rollback Process

If deployment fails:

1. **Automatic rollback**: Old container is preserved during deployment
2. **Manual rollback**: SSH to EC2 and run previous image tag:
   ```bash
   docker stop text-summarizer
   docker run -d -p 8080:8080 --name text-summarizer \
     <ECR_URI>/text-summarizer:<previous-commit-sha>
   ```

### Cost Optimization

- **ECR**: Only charged for storage (~$0.10 per GB/month)
- **GitHub Actions**: Free tier includes 2000 minutes/month
- **EC2**: Charges per hour (t3.medium ~$0.04/hour)

---

## Performance Considerations

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