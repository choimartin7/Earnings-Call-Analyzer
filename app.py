from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import logging
import re
import time


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests

# Global variables for model and tokenizer
model = None
tokenizer = None
MOCK_MODE = True  # Force mock mode to True since we don't have the model files

# Add parent directory to path to import from improved_finbert_finetuning
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Try to import model-related components
try:
    import torch
    from transformers import BertTokenizer
    from improved_finbert_finetuning import (
        BertForSequenceClassification, 
        MultiTaskModel,
        clean_transcript,
        extract_important_sections
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_LEN = 512
    logger.info(f"Successfully imported required modules")
except ImportError as e:
    logger.warning(f"Import error: {e}. Running in mock mode.")
    MOCK_MODE = True
    # Define minimal functions for mock mode
    def clean_transcript(text):
        return text

    device = "cpu"
    MAX_LEN = 512

def load_model():
    """Load the FinBERT model and tokenizer"""
    global model, tokenizer
    
    if MOCK_MODE:
        logger.info("Running in mock mode - no model will be loaded")
        return True
    
    try:
        logger.info(f"Using device: {device}")
        
        # Paths to model files
        model_path = './improved_finbert_finetuning.py'
        
        # Check if model directory exists
        if not os.path.exists(model_path):
            model_path = './improved_finbert_multitask_model'
            if not os.path.exists(model_path):
                logger.error(f"Model directory not found: {model_path}")
                return False
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = BertTokenizer.from_pretrained(model_path)
        
        # Load base model
        logger.info("Loading base model...")
        base_model = BertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=3,
            output_hidden_states=True
        )
        
        # Create multi-task model
        logger.info("Creating multi-task model...")
        model = MultiTaskModel(base_model)
        
        # Load model weights
        model_file = os.path.join(model_path, 'pytorch_model.bin')
        if not os.path.exists(model_file):
            model_file = '../best_improved_multitask_model.bin'
            if not os.path.exists(model_file):
                logger.error(f"Model weights file not found: {model_file}")
                return False
                
        logger.info(f"Loading model weights from {model_file}...")
        model.load_state_dict(torch.load(model_file, map_location=device))
        model.to(device)
        model.eval()
        
        logger.info("Model loaded successfully!")
        return True
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def process_transcript(text):
    """Process the transcript text for prediction"""
    if MOCK_MODE:
        return None, None
        
    try:
        # Clean and extract important sections
        cleaned_text = clean_transcript(text)
        
        # Tokenize
        encoding = tokenizer(
            cleaned_text,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        return input_ids, attention_mask
    
    except Exception as e:
        logger.error(f"Error processing transcript: {e}")
        return None, None

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if MOCK_MODE:
        return jsonify({'status': 'ok', 'message': 'Service is healthy (MOCK MODE)'}), 200
        
    if model is None or tokenizer is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 503
    
    return jsonify({
        'status': 'ok',
        'model_loaded': not MOCK_MODE,
        'mock_mode': MOCK_MODE
    }), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    if MOCK_MODE:
        # Return mock prediction data
        time.sleep(1)  # Simulate processing time
        
        # Use current time to vary the mock predictions
        time_seed = time.time()
        sentiment_selector = time_seed % 3  # 0, 1, or 2
        
        if sentiment_selector < 1:
            confidence = [0.75, 0.20, 0.05]  # Bearish
        elif sentiment_selector < 2:
            confidence = [0.15, 0.75, 0.10]  # Neutral
        else:
            confidence = [0.05, 0.20, 0.75]  # Bullish
            
        # Normalize confidence to sum to 1
        confidence_sum = sum(confidence)
        confidence = [c / confidence_sum for c in confidence]
        
        # Get sentiment based on highest confidence
        sentiment_idx = confidence.index(max(confidence))
        label_map = {0: 'Bearish', 1: 'Neutral', 2: 'Bullish'}
        sentiment_label = label_map[sentiment_idx]
        
        # Mock price change between -5% and 5%
        price_change = (time_seed % 10) - 5
        
        response = {
            'sentiment': sentiment_label,
            'sentiment_idx': sentiment_idx,
            'confidence': confidence,
            'price_change': round(price_change, 2),
            'price_change_raw': price_change,
            'mock': True
        }
        
        logger.info(f"Mock prediction: {response}")
        return jsonify(response), 200
    
    if model is None or tokenizer is None:
        return jsonify({'error': 'Model not loaded, please try again later'}), 503
    
    # Check if file or text was provided
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        try:
            # Read file content
            transcript_text = file.read().decode('utf-8')
        except UnicodeDecodeError:
            # Try different encoding if utf-8 fails
            try:
                transcript_text = file.read().decode('latin-1')
            except:
                return jsonify({'error': 'Failed to decode file. Please ensure it is a valid text file.'}), 400
    
    elif 'text' in request.form:
        transcript_text = request.form['text']
    else:
        return jsonify({'error': 'No file or text provided'}), 400
    
    if not transcript_text or len(transcript_text.strip()) < 50:
        return jsonify({'error': 'Transcript text too short or empty'}), 400
    
    # Process transcript
    input_ids, attention_mask = process_transcript(transcript_text)
    if input_ids is None:
        return jsonify({'error': 'Failed to process transcript'}), 500
    
    # Make prediction
    try:
        with torch.no_grad():
            _, logits, price_pred = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Get sentiment prediction
        probs = torch.nn.functional.softmax(logits, dim=1)
        confidence_scores = probs.cpu().numpy()[0].tolist()
        
        _, sentiment_pred = torch.max(logits, dim=1)
        sentiment_idx = sentiment_pred.item()
        
        # Label mapping
        label_map = {0: 'Bearish', 1: 'Neutral', 2: 'Bullish'}
        sentiment_label = label_map[sentiment_idx]
        
        # Get price change prediction
        price_change = float(price_pred.squeeze().cpu().numpy())
        
        # Prepare response
        response = {
            'sentiment': sentiment_label,
            'sentiment_idx': sentiment_idx,
            'confidence': confidence_scores,
            'price_change': round(price_change, 2),
            'price_change_raw': price_change
        }
        
        logger.info(f"Prediction: {response}")
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
if __name__ == '__main__':
    # Load model at startup
    if load_model() or MOCK_MODE:
        # Run the Flask app
        port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT is not set
        app.run(debug=True, host='0.0.0.0', port=port)
    else:
        logger.error("Failed to load model. Exiting.")
        sys.exit(1)

