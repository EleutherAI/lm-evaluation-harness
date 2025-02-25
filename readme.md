# Legal AI Model Evaluation Framework


## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/legal-ai-evaluation.git
   cd legal-ai-evaluation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with the following environment variables:
   ```
   AZURE_OPENAI_API_KEY=your_azure_openai_key
   AZURE_OPENAI_ENDPOINT=your_azure_endpoint
   HUGGINGFACE_TOKEN=your_huggingface_token
   ARGILLA_API_KEY=your_argilla_key  # Optional
   ARGILLA_API_URL=your_argilla_url  # Optional
   ```

## Setting Up MLflow

MLflow is used for tracking experiments, storing metrics, and visualizing results.

1. Install MLflow if not included in your requirements:
   ```bash
   pip install mlflow
   ```

2. By default, MLflow will store data locally in an `mlruns` directory. For production use, you might want to configure a tracking server:

   ```bash
   # Start MLflow tracking server
   mlfow ui
   ```

3. Configure your environment to use this server by adding to your `.env` file:
   ```
   MLFLOW_TRACKING_URI=http://localhost:5000
   ```

4. Access the MLflow UI by navigating to `http://localhost:5000` in your browser.

## Setting Up Argilla

Argilla is used for qualitative evaluation and human review of model responses.

1. mkdir argilla 
2. cd argilla
3. Install Argilla if not included in your requirements:
   ```bash
   pip install argilla
   ```

4. For local development, you can run Argilla using Docker:
   ```bash
   # Pull and start Argilla
   docker run -d --name argilla -p 6900:6900 argilla/argilla-quickstart:latest
   ```

5. Update your `.env` file with the Argilla credentials:
   ```
   ARGILLA_API_KEY=your_api_key
   ARGILLA_API_URL=http://localhost:6900
   ```

6. Access the Argilla UI by navigating to `http://localhost:6900` in your browser

## Usage

Run the evaluation with:
Enter lm-evaluation-harness dir 
```bash
python main.py
```

## Results Analysis

### MLflow Integration

Results are automatically logged to MLflow for tracking experiments. To view the results:

1. Start the MLflow UI (if not using a persistent server):
   ```bash
   mlflow ui
   ```

2. Open your browser to http://localhost:5000

3. Navigate to the "AIBE_Model_Evaluation" experiment to view:
   - Accuracy metrics
   - Additional classification metrics (if implemented)
   - Run parameters
   - Artifact files with detailed results

### Argilla Integration

For qualitative analysis of individual question results:

1. After running the evaluation, log in to your Argilla instance (default: http://localhost:6900)
2. Navigate to the "aibe_evaluation" dataset
3. Review individual questions with:
   - Question text and options
   - Model's response
   - Correct answer
   - Validation status (correct/incorrect)
4. Add comments or annotations for further analysis



