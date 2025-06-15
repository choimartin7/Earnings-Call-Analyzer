import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import numpy as np
import json
import random
from transformers import BertTokenizer, BertForSequenceClassification, BertForMaskedLM, get_linear_schedule_with_warmup
from transformers import DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error
import logging
import re
from tqdm import tqdm
import csv
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Constants
TRANSCRIPTS_PATH = "./transcript_final"  # Updated to the correct directory
DATA_MAPPING_PATH = "./data_mapping (1-day).csv"  # Updated to use 1-day price change data
MAX_LEN = 512  # Maximum sequence length for tokens
BATCH_SIZE = 4  # Batch size for training
EPOCHS = 5  # Number of epochs
LEARNING_RATE = 2e-5
REGRESSION_WEIGHT = 0.2  # Further reduced to emphasize classification more
CHUNK_STRATEGY = "all"  # Process all chunks
FOCAL_LOSS_GAMMA = 3.0  # Increased gamma for focal loss to focus more on hard examples
MLM_EPOCHS = 1  # Number of epochs for continued pre-training with MLM (reduced from 2)
MLM_BATCH_SIZE = 8  # Batch size for MLM pre-training
RLHF_BATCH_SIZE = 2  # Smaller batch size for RLHF fine-tuning
RLHF_EPOCHS = 2  # Number of epochs for RLHF (reduced from 3)
RLHF_LEARNING_RATE = 1e-6  # Lower learning rate for RLHF fine-tuning
PPO_EPOCHS = 5  # Number of policy optimization epochs
CLIP_EPSILON = 0.2  # Clip parameter for PPO

# Financial jargon and earnings call terminology to focus on during MLM
FINANCIAL_JARGON = [
    "revenue", "earnings", "profit", "margin", "guidance", "eps", "ebitda", 
    "diluted", "gaap", "non-gaap", "forecast", "outlook", "quarter", "fiscal",
    "dividend", "growth", "decline", "increase", "decrease", "performance",
    "billion", "million", "sequentially", "year-over-year", "yoy", "q1", "q2", "q3", "q4",
    "operating", "expenses", "capex", "cash flow", "balance sheet", "debt", "equity",
    "adjusted", "restructuring", "impairment", "acquisition", "merger", "guidance",
    "segment", "product", "service", "customer", "market", "competition", "strategy",
    "risk", "opportunity", "pipeline", "backlog", "bookings", "visibility", "momentum",
    "headwind", "tailwind", "macroeconomic", "inflation", "recession", "recovery"
]

# Focal Loss implementation for handling class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            if isinstance(alpha, torch.Tensor):
                self.alpha = alpha
            else:
                self.alpha = torch.tensor(alpha)
        else:
            self.alpha = None
            
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

# Multi-Task Model class (classification + regression)
class MultiTaskModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        # Regressor for price prediction
        self.regressor = nn.Linear(base_model.config.hidden_size, 1)
        
        # Additional classifier layer to improve performance
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(base_model.config.hidden_size, base_model.config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(base_model.config.hidden_size // 2, 3)  # 3 sentiment classes
        )
        
        # Temperature parameter for softmax (learnable)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        
    def forward(self, input_ids, attention_mask, labels=None):
        # Get base model outputs
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        # Get the hidden states
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]
        else:
            # Access the last_hidden_state from the base model output
            hidden_states = self.base.get_input_embeddings()(input_ids)
            for layer in self.base.encoder.layer:
                hidden_states = layer(hidden_states, attention_mask)[0]
        
        # Average pooling for regression and classification
        pooled_output = hidden_states.mean(dim=1)
        
        # Regression prediction
        regression_pred = self.regressor(pooled_output)
        
        # Enhanced classification with our custom classifier
        logits = self.classifier(pooled_output)
        
        # Apply temperature scaling to logits
        logits = logits / self.temperature
        
        # Calculate classification loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return loss, logits, regression_pred

# Load data mapping
def load_data_mapping():
    try:
        # Read the CSV file and display column names for debugging
        df = pd.read_csv(DATA_MAPPING_PATH)
        logger.info(f"CSV columns: {df.columns.tolist()}")
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        logger.info(f"CSV columns after stripping: {df.columns.tolist()}")
        
        # Fill NaN values in the Sentiment Label column with an empty string
        if 'Sentiment Label' in df.columns:
            # Drop rows with empty sentiment labels
            df = df.dropna(subset=['Sentiment Label'])
            logger.info(f"After dropping NaN, shape: {df.shape}")
        else:
            logger.error(f"'Sentiment Label' column not found. Available columns: {df.columns.tolist()}")
            return None
        
        return df
    except Exception as e:
        logger.error(f"Error loading data mapping: {str(e)}")
        return None

def extract_important_sections(text):
    """
    Extract the most important parts of the earnings call transcript 
    focusing on CEO/CFO statements, guidance, and financial results.
    """
    # Split into sections
    sections = text.split('\n\n')
    
    important_sections = []
    financial_sections = []
    guidance_sections = []
    qa_sections = []
    
    # Keywords to identify important sections
    financial_keywords = [
        'revenue', 'earnings', 'profit', 'income', 'margin', 'growth', 
        'billion', 'million', 'quarter', 'fiscal', 'financial',
        'EBITDA', 'EPS', 'diluted', 'GAAP', 'non-GAAP'
    ]
    
    guidance_keywords = [
        'outlook', 'guidance', 'expect', 'forecast', 'future', 
        'next quarter', 'coming year', 'anticipate'
    ]
    
    executive_speakers = [
        'CEO', 'Chief Executive Officer', 'CFO', 'Chief Financial Officer',
        'Chairman', 'President'
    ]
    
    for section in sections:
        lower_section = section.lower()
        
        # Check if this is an executive speaking
        is_executive_speaking = any(title in section for title in executive_speakers)
        
        # Check if this contains financial information
        has_financial_info = any(keyword in lower_section for keyword in financial_keywords)
        
        # Check if this contains guidance
        has_guidance = any(keyword in lower_section for keyword in guidance_keywords)
        
        # Check if this is part of Q&A
        is_qa = 'question' in lower_section or 'analyst' in lower_section
        
        # Prioritize sections
        if has_guidance:
            guidance_sections.append(section)
        elif has_financial_info:
            financial_sections.append(section)
        elif is_executive_speaking:
            important_sections.append(section)
        elif is_qa:
            qa_sections.append(section)
    
    # Prioritize sections in this order: guidance, financial, executive statements, Q&A
    all_important_sections = guidance_sections + financial_sections + important_sections + qa_sections
    
    # If we found important sections, join them together
    if all_important_sections:
        return ' '.join(all_important_sections)
    
    # If we couldn't find any important sections, return the original text
    return text

# Clean transcript text
def clean_transcript(text):
    # Remove header information (look for "Prepared Remarks:" or similar markers)
    lines = text.split('\n')
    cleaned_lines = []
    content_started = False
    
    for line in lines:
        if not content_started and ("Prepared Remarks:" in line or "Operator" in line):
            content_started = True
        if content_started:
            cleaned_lines.append(line)
    
    # If no marker was found, use the entire transcript
    if not cleaned_lines:
        cleaned_lines = lines
    
    # Join the cleaned lines
    cleaned_text = '\n\n'.join(cleaned_lines)
    
    # Remove promotional content like "Where to invest $1,000 right now"
    cleaned_text = re.sub(r'Where to invest \$1,000 right now.*?\*Stock Advisor returns as of.*?$', '', cleaned_text, flags=re.DOTALL)
    
    # Extract the most important sections
    important_text = extract_important_sections(cleaned_text)
    
    # Remove extra whitespaces for the final text
    final_text = re.sub(r'\s+', ' ', important_text).strip()
    
    return final_text

def chunk_transcript(text, tokenizer, max_len):
    """
    Divide a long transcript into chunks, focusing on the most important parts
    and staying within token limits.
    """
    # First, clean and extract important sections
    cleaned_text = clean_transcript(text)
    
    # Tokenize the cleaned text
    tokens = tokenizer.encode(cleaned_text, add_special_tokens=False)
    
    # If it's already within limits, return the whole text
    if len(tokens) <= max_len - 2:  # -2 for [CLS] and [SEP] tokens
        return [cleaned_text]
    
    # Otherwise, we need to chunk the text
    chunks = []
    
    # Start by extracting the most important chunks
    sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        # Tokenize the sentence to get its length
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        sentence_length = len(sentence_tokens)
        
        # If adding this sentence would exceed max length, finalize the current chunk
        if current_length + sentence_length > max_len - 2:
            if current_chunk:  # Only append if we have content
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    # Apply chunk selection strategy
    if CHUNK_STRATEGY == "random" and len(chunks) > 1:
        # Return a random chunk for training diversity
        return [chunks[np.random.randint(0, len(chunks))]]
    elif CHUNK_STRATEGY == "important_first":
        # Return the first chunk as it contains the most important content
        return [chunks[0]]
    elif CHUNK_STRATEGY == "beginning":
        # Just return the first chunk
        return [chunks[0]]
    elif CHUNK_STRATEGY == "all":
        # Return all chunks, with no limit to ensure we process the full transcript
        return chunks  # Changed from chunks[:5] to chunks (no limit)
    else:
        # Default: return all chunks
        return chunks

# Load and process transcripts
def load_transcripts(filenames, tokenizer):
    texts = []
    expanded_filenames = []  # Create a new list instead of modifying the original
    encoding_issues = 0
    
    for filename in filenames:
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'ISO-8859-1']
            text = None
            
            for encoding in encodings:
                try:
                    with open(os.path.join(TRANSCRIPTS_PATH, filename), 'r', encoding=encoding) as f:
                        text = f.read()
                    break  # If successful, break the encoding loop
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logger.error(f"Error reading file with encoding {encoding}: {e}")
            
            if text is None:
                logger.error(f"Failed to read {filename} with any encoding")
                texts.append("")
                expanded_filenames.append(filename)
                encoding_issues += 1
                continue
                
            # Process the transcript into chunks
            chunks = chunk_transcript(text, tokenizer, MAX_LEN)
            
            if CHUNK_STRATEGY == "all" and chunks:
                # Add all chunks to the texts list
                texts.extend(chunks)
                # Add the filename multiple times to match the number of chunks
                expanded_filenames.append(filename)  # Add the first chunk
                # Add more filenames for additional chunks
                for _ in range(1, len(chunks)):
                    expanded_filenames.append(filename)
            else:
                # Add first chunk (or only chunk) to the texts list
                if chunks:
                    texts.append(chunks[0])
                    expanded_filenames.append(filename)
                else:
                    texts.append("")
                    expanded_filenames.append(filename)
                    logger.warning(f"No valid chunks extracted from {filename}")
            
        except Exception as e:
            logger.error(f"Error loading transcript {filename}: {e}")
            texts.append("")  # Add empty string as placeholder
            expanded_filenames.append(filename)
            encoding_issues += 1
    
    if encoding_issues > 0:
        logger.warning(f"Encountered encoding issues with {encoding_issues} files")
    
    logger.info(f"Created {len(texts)} text chunks from {len(set(expanded_filenames))} unique transcripts")
    return texts, expanded_filenames  # Return expanded filenames list to match with labels

# Dataset class for multi-task learning
class EarningsCallDataset(Dataset):
    def __init__(self, texts, labels, price_changes, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.price_changes = price_changes
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Convert string labels to numeric
        self.label_map = {'Bearish': 0, 'Neutral': 1, 'Bullish': 2}
        self.numeric_labels = [self.label_map[label] for label in self.labels]
        
        # Calculate class weights for balancing
        label_counts = {}
        for label in self.numeric_labels:
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
        
        total = len(self.numeric_labels)
        self.class_weights = {
            class_idx: total / (count * 3)  # Multiply by number of classes (3)
            for class_idx, count in label_counts.items()
        }
        
        # Add default weight for any missing classes
        for i in range(3):
            if i not in self.class_weights:
                self.class_weights[i] = 1.0
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.numeric_labels[idx]
        price_change = self.price_changes[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Add class weight
        class_weight = self.class_weights[label]
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'price_change': torch.tensor(price_change, dtype=torch.float),
            'class_weight': torch.tensor(class_weight, dtype=torch.float)
        }

# Fine-tuning function for multi-task learning
def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device):
    model.train()
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # MSE Loss for regression
    mse_loss = nn.MSELoss()
    
    # Focal loss for classification with alpha based on class distribution
    class_distribution = {}
    for batch in train_dataloader:
        labels = batch['labels'].cpu().numpy()
        for label in labels:
            if label not in class_distribution:
                class_distribution[label] = 0
            class_distribution[label] += 1
            
    total_samples = sum(class_distribution.values())
    alpha = {}
    for label, count in class_distribution.items():
        alpha[label] = 1.0 - (count / total_samples)
        
    # Create alpha tensor (default to 0.5 if a class is missing)
    alpha_tensor = torch.tensor([alpha.get(i, 0.5) for i in range(3)], device=device)
    
    # Initialize focal loss
    focal_loss = FocalLoss(alpha=alpha_tensor, gamma=FOCAL_LOSS_GAMMA)
    
    # Cross entropy loss with weights for comparing
    weighted_cross_entropy = nn.CrossEntropyLoss(reduction='none')
    
    for epoch in range(EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{EPOCHS}")
        total_train_loss = 0
        
        # Training
        model.train()
        progress_bar = tqdm(train_dataloader, desc="Training")
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            price_change = batch['price_change'].to(device)
            class_weights = batch['class_weight'].to(device)
            
            # Forward pass
            # The model returns classification_loss, but we'll recalculate it with focal loss
            _, logits, regression_pred = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )
            
            # Calculate focal loss for classification
            cls_loss = focal_loss(logits, labels)
            
            # Calculate regression loss
            reg_loss = mse_loss(regression_pred.squeeze(), price_change)
            
            # Combined loss
            loss = cls_loss + REGRESSION_WEIGHT * reg_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        logger.info(f"Average training loss: {avg_train_loss}")
        
        # Validation
        model.eval()
        total_val_loss = 0
        
        for batch in tqdm(val_dataloader, desc="Validation"):
            with torch.no_grad():
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                price_change = batch['price_change'].to(device)
                
                # Forward pass
                _, logits, regression_pred = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    labels=labels
                )
                
                # Calculate focal loss for classification
                cls_loss = focal_loss(logits, labels)
                
                # Calculate regression loss
                reg_loss = mse_loss(regression_pred.squeeze(), price_change)
                
                # Combined loss
                loss = cls_loss + REGRESSION_WEIGHT * reg_loss
                
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        logger.info(f"Average validation loss: {avg_val_loss}")
        
        # Save model if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_improved_multitask_model.bin')
            logger.info("Model checkpoint saved")
    
    return train_losses, val_losses

# Evaluation function for multi-task model
def evaluate_model(model, test_dataloader, device):
    model.eval()
    sentiment_predictions = []
    actual_sentiment_labels = []
    price_predictions = []
    actual_price_changes = []
    
    # Store predictions by transcript file for later aggregation
    transcript_preds = {}
    transcript_labels = {}
    transcript_price_preds = {}
    transcript_price_actuals = {}
    
    # Load test data information for grouping chunks
    test_data = [(item['input_ids'], item['attention_mask'], item['labels'], item['price_change'], 
                  test_dataloader.dataset.texts[i], test_dataloader.dataset.labels[i])
                 for i, item in enumerate([test_dataloader.dataset[i] for i in range(len(test_dataloader.dataset))])]
    
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            price_change = batch['price_change'].to(device)
            
            # Forward pass
            _, logits, regression_pred = model(
                input_ids=input_ids, 
                attention_mask=attention_mask
            )
            
            # Get sentiment predictions
            _, sentiment_preds = torch.max(logits, dim=1)
            
            # Collect predictions and actual values
            sentiment_predictions.extend(sentiment_preds.cpu().tolist())
            actual_sentiment_labels.extend(labels.cpu().tolist())
            
            # Handle regression predictions - fix for batches of size 1
            regression_values = regression_pred.squeeze().cpu()
            if regression_values.dim() == 0:  # It's a scalar tensor
                price_predictions.append(float(regression_values))
            else:  # It's a tensor with multiple values
                price_predictions.extend(regression_values.tolist())
                
            actual_price_changes.extend(price_change.cpu().tolist())
    
    # Map numeric sentiment labels back to strings for report
    label_map_reverse = {0: 'Bearish', 1: 'Neutral', 2: 'Bullish'}
    pred_labels = [label_map_reverse[p] for p in sentiment_predictions]
    true_labels = [label_map_reverse[a] for a in actual_sentiment_labels]
    
    # Print classification report
    logger.info("Classification Report:")
    logger.info(classification_report(true_labels, pred_labels))
    
    # Print confusion matrix
    logger.info("Confusion Matrix:")
    cm = confusion_matrix(true_labels, pred_labels, labels=['Bearish', 'Neutral', 'Bullish'])
    logger.info(cm)
    
    # Calculate regression metrics
    mse = mean_squared_error(actual_price_changes, price_predictions)
    mae = mean_absolute_error(actual_price_changes, price_predictions)
    logger.info(f"Regression MSE: {mse:.4f}")
    logger.info(f"Regression MAE: {mae:.4f}")
    
    return pred_labels, true_labels, price_predictions, actual_price_changes

# Class for Masked Language Modeling dataset
class EarningsCallMLMDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

# Custom data collator for MLM that emphasizes financial jargon
class FinancialJargonMLMDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=True, mlm_probability=0.15, jargon_terms=None, jargon_weight=3.0):
        super().__init__(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability)
        self.jargon_terms = jargon_terms or []
        self.jargon_token_ids = []
        self.jargon_weight = jargon_weight  # How much more likely jargon terms are to be masked
        
        # Get token IDs for jargon terms
        for term in self.jargon_terms:
            # Convert term to lowercase and tokenize
            tokens = self.tokenizer.tokenize(term.lower())
            if tokens:
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                self.jargon_token_ids.extend(token_ids)
    
    def torch_mask_tokens(self, inputs, special_tokens_mask=None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        With higher probability for financial jargon terms.
        """
        labels = inputs.clone()
        
        # Create probability mask with higher probability for jargon terms
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()
        
        # Increase masking probability for financial jargon terms
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if labels[i, j].item() in self.jargon_token_ids:
                    probability_matrix[i, j] = min(self.mlm_probability * self.jargon_weight, 0.9)
        
        # Set probability of masking special tokens to 0
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        # Create the mask
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens
        
        # 80% of the time, replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        
        # 10% of the time, replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        
        # The rest of the time (10%), keep the masked input tokens unchanged
        return inputs, labels

# Function for continued pre-training with MLM
def continued_pretraining_mlm(texts, tokenizer, device):
    logger.info("Starting continued pre-training with Masked Language Modeling...")
    
    # Create MLM dataset
    mlm_dataset = EarningsCallMLMDataset(texts, tokenizer, MAX_LEN)
    mlm_dataloader = DataLoader(mlm_dataset, batch_size=MLM_BATCH_SIZE, shuffle=True)
    
    # Load model for pre-training
    model = BertForMaskedLM.from_pretrained('yiyanghkust/finbert-tone')
    model = model.to(device)
    
    # Initialize data collator for MLM with emphasis on financial jargon
    data_collator = FinancialJargonMLMDataCollator(
        tokenizer=tokenizer, 
        mlm=True, 
        mlm_probability=0.15, 
        jargon_terms=FINANCIAL_JARGON, 
        jargon_weight=3.0
    )
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    total_steps = len(mlm_dataloader) * MLM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    model.train()
    for epoch in range(MLM_EPOCHS):
        logger.info(f"MLM Pre-training - Epoch {epoch + 1}/{MLM_EPOCHS}")
        total_loss = 0
        
        progress_bar = tqdm(mlm_dataloader, desc="MLM Pre-training")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Create masked inputs and labels
            masked_inputs, labels = data_collator.torch_mask_tokens(input_ids.clone())
            
            # Forward pass
            outputs = model(
                input_ids=masked_inputs,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(mlm_dataloader)
        logger.info(f"MLM Pre-training - Average loss: {avg_loss}")
    
    # Save pre-trained model
    mlm_output_dir = './finbert_mlm_pretrained'
    os.makedirs(mlm_output_dir, exist_ok=True)
    model.save_pretrained(mlm_output_dir)
    tokenizer.save_pretrained(mlm_output_dir)
    logger.info(f"Saved MLM pre-trained model to {mlm_output_dir}")
    
    return mlm_output_dir

# Reward Model for RLHF
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.reward_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(base_model.config.hidden_size, 1)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract the hidden representation
        # BertForSequenceClassification output format is different
        # It returns a tuple with logits as the first element
        if hasattr(outputs, 'last_hidden_state'):
            # For models that return last_hidden_state
            hidden_states = outputs.last_hidden_state[:, 0, :]  # CLS token
        elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            # For models with hidden_states attribute
            hidden_states = outputs.hidden_states[-1][:, 0, :]  # CLS token from last layer
        else:
            # Fallback: use pooler output or directly use base model output
            pooler_output = self.base.get_input_embeddings()(input_ids)
            # Mean pooling
            hidden_states = torch.mean(pooler_output * attention_mask.unsqueeze(-1), dim=1)
        
        reward = self.reward_head(hidden_states)
        return reward

# Human Feedback Dataset for RLHF
class HumanFeedbackDataset(Dataset):
    def __init__(self, chosen_texts, rejected_texts, tokenizer, max_len):
        self.chosen_texts = chosen_texts
        self.rejected_texts = rejected_texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.chosen_texts)
    
    def __getitem__(self, idx):
        chosen_text = self.chosen_texts[idx]
        rejected_text = self.rejected_texts[idx]
        
        chosen_encoding = self.tokenizer(
            chosen_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        rejected_encoding = self.tokenizer(
            rejected_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'chosen_input_ids': chosen_encoding['input_ids'].flatten(),
            'chosen_attention_mask': chosen_encoding['attention_mask'].flatten(),
            'rejected_input_ids': rejected_encoding['input_ids'].flatten(),
            'rejected_attention_mask': rejected_encoding['attention_mask'].flatten()
        }

# Function to train the reward model using human feedback
def train_reward_model(chosen_texts, rejected_texts, tokenizer, device):
    logger.info("Training reward model using human feedback...")
    
    # Create dataset and dataloader
    dataset = HumanFeedbackDataset(chosen_texts, rejected_texts, tokenizer, MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Load the base model
    base_model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', output_hidden_states=True)
    reward_model = RewardModel(base_model)
    reward_model = reward_model.to(device)
    
    # Optimizer
    optimizer = AdamW(reward_model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    reward_model.train()
    total_steps = len(dataloader) * EPOCHS
    
    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Reward Model Training Epoch {epoch+1}/{EPOCHS}")
        
        for batch in progress_bar:
            chosen_input_ids = batch['chosen_input_ids'].to(device)
            chosen_attention_mask = batch['chosen_attention_mask'].to(device)
            rejected_input_ids = batch['rejected_input_ids'].to(device)
            rejected_attention_mask = batch['rejected_attention_mask'].to(device)
            
            # Forward pass
            chosen_reward = reward_model(chosen_input_ids, chosen_attention_mask)
            rejected_reward = reward_model(rejected_input_ids, rejected_attention_mask)
            
            # Loss is the log of sigmoid of the difference
            # We want chosen to have higher reward than rejected
            loss = -torch.log(torch.sigmoid(chosen_reward - rejected_reward)).mean()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reward_model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Reward Model Training - Epoch {epoch+1} - Average Loss: {avg_loss}")
    
    # Save the reward model
    reward_model_dir = './finbert_reward_model'
    os.makedirs(reward_model_dir, exist_ok=True)
    torch.save(reward_model.state_dict(), os.path.join(reward_model_dir, 'reward_model.pt'))
    logger.info(f"Saved reward model to {reward_model_dir}")
    
    return reward_model

# PPO Implementation for RLHF
def rlhf_ppo_training(model, reward_model, texts, tokenizer, device):
    logger.info("Starting RLHF PPO training...")
    
    # Create dataset and dataloader
    class RLHFDataset(Dataset):
        def __init__(self, texts, tokenizer, max_len):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_len = max_len
            
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = self.texts[idx]
            
            encoding = self.tokenizer(
                text,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'text': text
            }
    
    dataset = RLHFDataset(texts, tokenizer, MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=RLHF_BATCH_SIZE, shuffle=True)
    
    # Create a reference model (frozen copy of the current model)
    base_model = BertForSequenceClassification.from_pretrained(
        'yiyanghkust/finbert-tone', 
        num_labels=3,
        output_hidden_states=True
    )
    ref_model = MultiTaskModel(base_model)
    ref_model.load_state_dict(model.state_dict())
    ref_model = ref_model.to(device)
    ref_model.eval()  # Set to evaluation mode, no gradient updates
    
    # Set reward model to evaluation mode
    reward_model.eval()
    
    # PPO optimizer (lower learning rate)
    optimizer = AdamW(model.parameters(), lr=RLHF_LEARNING_RATE)
    
    # Training loop
    model.train()
    total_steps = len(dataloader) * RLHF_EPOCHS
    
    for epoch in range(RLHF_EPOCHS):
        total_reward = 0
        total_kl_div = 0
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"RLHF Epoch {epoch+1}/{RLHF_EPOCHS}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Store batch texts for potential human feedback collection
            batch_texts = batch['text']
            
            # Get logits from reference model (no grad)
            with torch.no_grad():
                ref_loss, ref_logits, _ = ref_model(input_ids=input_ids, attention_mask=attention_mask)
                ref_probs = F.softmax(ref_logits, dim=-1)
            
            # PPO training loop
            for _ in range(PPO_EPOCHS):
                # Forward pass current model
                _, logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = F.softmax(logits, dim=-1)
                
                # Calculate rewards using reward model
                with torch.no_grad():
                    rewards = reward_model(input_ids, attention_mask).squeeze(-1)
                
                # Calculate KL divergence between current and reference policy
                kl_div = F.kl_div(probs.log(), ref_probs, reduction='batchmean')
                
                # PPO policy loss
                # Calculate advantage - normalize rewards
                advantages = rewards - rewards.mean()
                advantages = advantages / (rewards.std() + 1e-8)
                
                # Calculate ratio between current and reference policy
                log_ratio = (probs / (ref_probs + 1e-8)).log()
                
                # Reshape advantages to have the right dimensions for broadcasting
                # Unsqueeze to add a dimension: [batch_size] -> [batch_size, 1]
                advantages_expanded = advantages.unsqueeze(-1)
                
                # Clipped PPO loss
                ratio = log_ratio.exp()
                # Now both have compatible shapes for multiplication
                clip_adv = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * advantages_expanded
                # Take mean across batch and class dimensions
                policy_loss = -torch.min(ratio * advantages_expanded, clip_adv).mean()
                
                # Final loss with KL penalty
                kl_penalty = 0.1 * kl_div  # Beta parameter to control KL influence
                loss = policy_loss + kl_penalty
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            total_reward += rewards.mean().item()
            total_kl_div += kl_div.item()
            total_loss += loss.item()
            
            progress_bar.set_postfix({
                'reward': rewards.mean().item(), 
                'kl_div': kl_div.item(),
                'loss': loss.item()
            })
        
        # Log epoch metrics
        avg_reward = total_reward / len(dataloader)
        avg_kl_div = total_kl_div / len(dataloader)
        avg_loss = total_loss / len(dataloader)
        
        logger.info(f"RLHF - Epoch {epoch+1} - Avg Reward: {avg_reward} - Avg KL Div: {avg_kl_div} - Avg Loss: {avg_loss}")
    
    # Save RLHF-tuned model
    rlhf_model_dir = './finbert_rlhf_tuned'
    os.makedirs(rlhf_model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(rlhf_model_dir, 'rlhf_model.pt'))
    logger.info(f"Saved RLHF tuned model to {rlhf_model_dir}")
    
    return model

# Function to collect human feedback
def collect_human_feedback(model, tokenizer, texts, labels, device, output_path='human_feedback.csv', real_feedback_path='real_human_feedback.csv'):
    """
    Collect human feedback by comparing model predictions with human judgments.
    First tries to load real human feedback data if available, otherwise falls back to synthetic feedback.
    
    In a real-world application, this would involve showing predictions to humans
    and recording their preferences/feedback.
    """
    logger.info("Collecting human feedback...")
    
    # First try to load real human feedback if available
    if os.path.exists(real_feedback_path):
        logger.info(f"Using real human feedback from {real_feedback_path}")
        chosen_texts = []
        rejected_texts = []
        
        with open(real_feedback_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                chosen_texts.append(row.get('chosen_text', ''))
                rejected_texts.append(row.get('rejected_text', ''))
        
        if chosen_texts and rejected_texts:
            logger.info(f"Loaded {len(chosen_texts)} real human feedback pairs")
            return chosen_texts, rejected_texts
        else:
            logger.warning("Real feedback file exists but contains no valid data, falling back to synthetic feedback")
    
    # Check if the synthetic feedback file already exists
    if os.path.exists(output_path):
        logger.info(f"Using existing synthetic feedback from {output_path}")
        
        # Load existing feedback data
        chosen_texts = []
        rejected_texts = []
        
        with open(output_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                chosen_texts.append(row['chosen_text'])
                rejected_texts.append(row['rejected_text'])
        
        return chosen_texts, rejected_texts
    
    # If no existing feedback, simulate collection
    logger.info("No existing feedback file found. Creating simulated feedback...")
    
    # Prepare model for prediction
    model.eval()
    
    # Create synthetic feedback pairs based on model predictions
    chosen_texts = []
    rejected_texts = []
    
    # Sample subset of data for feedback 
    sample_indices = random.sample(range(len(texts)), min(100, len(texts)))
    
    # Get model predictions
    for idx in tqdm(sample_indices, desc="Generating synthetic feedback"):
        text = texts[idx]
        actual_label = labels[idx]
        
        # Create a modified version of the text with opposite sentiment words
        modified_text = text
        
        if actual_label == "Bullish":
            # Create a more bearish version
            for word in ["growth", "increase", "positive"]:
                modified_text = modified_text.replace(word, "decline")
            for word in ["strong", "excellent"]:
                modified_text = modified_text.replace(word, "weak")
        elif actual_label == "Bearish":
            # Create a more bullish version
            for word in ["decline", "decrease", "negative"]:
                modified_text = modified_text.replace(word, "growth")
            for word in ["weak", "poor"]:
                modified_text = modified_text.replace(word, "strong")
        
        # If no significant modification was made, continue
        if text == modified_text:
            continue
        
        # Always prefer original text for bullish and bearish labels (simulated preference)
        if actual_label in ["Bullish", "Bearish"]:
            chosen_texts.append(text)
            rejected_texts.append(modified_text)
    
    # Save the feedback data
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['chosen_text', 'rejected_text', 'timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for chosen, rejected in zip(chosen_texts, rejected_texts):
            writer.writerow({
                'chosen_text': chosen,
                'rejected_text': rejected,
                'timestamp': datetime.now().isoformat()
            })
    
    logger.info(f"Generated {len(chosen_texts)} synthetic feedback pairs and saved to {output_path}")
    return chosen_texts, rejected_texts

# Main function extension to incorporate RLHF
def main():
    # Load mapping data
    mapping_df = load_data_mapping()
    if mapping_df is None:
        return
    
    logger.info(f"Loaded {len(mapping_df)} entries from data mapping")
    
    # Load tokenizer first for transcript processing
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    
    # Load transcripts
    filenames = mapping_df['Earning Call Transcript'].tolist()
    original_labels = mapping_df['Sentiment Label'].tolist()
    
    # Get price changes - updated to use 1-day price change
    original_price_changes = mapping_df['1 Day Price Change (%)'].values.astype(float)
    
    # Create a mapping from filename to label and price change
    label_map = {fname: label for fname, label in zip(filenames, original_labels)}
    price_map = {fname: price for fname, price in zip(filenames, original_price_changes)}
    
    logger.info(f"Loading and processing {len(filenames)} transcripts...")
    texts, expanded_filenames = load_transcripts(filenames, tokenizer)
    logger.info(f"Loaded {len(texts)} transcripts")
    
    # Expand labels and price changes to match the chunked texts
    expanded_labels = [label_map[fname] for fname in expanded_filenames]
    expanded_price_changes = [price_map[fname] for fname in expanded_filenames]
    
    # Perform data augmentation for minority classes
    # Count occurrences of each class
    class_counts = {}
    for label in expanded_labels:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1
    
    # Find the majority class
    majority_class = max(class_counts, key=class_counts.get)
    majority_count = class_counts[majority_class]
    
    # Apply synthetic data augmentation for minority classes
    augmented_texts = texts.copy()
    augmented_labels = expanded_labels.copy()
    augmented_prices = expanded_price_changes.copy()
    
    for label, count in class_counts.items():
        if label != majority_class:
            # Get indices of samples with this label
            indices = [i for i, l in enumerate(expanded_labels) if l == label]
            
            # Determine how many samples to duplicate (to approach majority class size)
            duplicate_factor = min(3, majority_count // count)  # Cap at 3x duplication
            
            if duplicate_factor > 1:
                for _ in range(duplicate_factor - 1):  # -1 because we already have one copy
                    for idx in indices:
                        # Add small random noise to price changes for robustness
                        price = expanded_price_changes[idx] * (1 + np.random.normal(0, 0.05))
                        
                        # Text augmentation: add emphasis on sentiment-related keywords
                        text = texts[idx]
                        if label == "Bullish":
                            emphasis_tokens = ["growth", "increase", "success", "positive", "strong"]
                        elif label == "Bearish":
                            emphasis_tokens = ["decline", "decrease", "challenge", "negative", "weak"]
                        else:
                            emphasis_tokens = []
                            
                        # Simple augmentation by adding emphasis to sentiment words
                        for token in emphasis_tokens:
                            if token in text.lower():
                                text = text.replace(token, token + " " + token)
                        
                        augmented_texts.append(text)
                        augmented_labels.append(label)
                        augmented_prices.append(price)
    
    logger.info(f"After augmentation: {len(augmented_texts)} samples (was {len(texts)})")
    
    # Perform continued pre-training with MLM
    mlm_model_dir = continued_pretraining_mlm(texts, tokenizer, device)
    
    # Split data using augmented datasets
    train_texts, test_texts, train_labels, test_labels, train_prices, test_prices = train_test_split(
        augmented_texts, augmented_labels, augmented_prices, test_size=0.2, random_state=42, stratify=augmented_labels
    )
    
    train_texts, val_texts, train_labels, val_labels, train_prices, val_prices = train_test_split(
        train_texts, train_labels, train_prices, test_size=0.2, random_state=42, stratify=train_labels
    )
    
    logger.info(f"Train set: {len(train_texts)}, Validation set: {len(val_texts)}, Test set: {len(test_texts)}")
    
    # Load the MLM pre-trained model for fine-tuning
    base_model = BertForSequenceClassification.from_pretrained(
        mlm_model_dir, 
        num_labels=3,  # Three sentiment classes
        output_attentions=False,
        output_hidden_states=True,  # We need hidden states for regression
    )
    
    # Create multi-task model
    model = MultiTaskModel(base_model)
    model = model.to(device)
    
    # Create datasets and dataloaders
    train_dataset = EarningsCallDataset(train_texts, train_labels, train_prices, tokenizer, MAX_LEN)
    val_dataset = EarningsCallDataset(val_texts, val_labels, val_prices, tokenizer, MAX_LEN)
    test_dataset = EarningsCallDataset(test_texts, test_labels, test_prices, tokenizer, MAX_LEN)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Prepare optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Train the model
    logger.info("Starting multi-task fine-tuning...")
    train_losses, val_losses = train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device)
    
    # Load best model for evaluation
    model.load_state_dict(torch.load('best_improved_multitask_model.bin'))
    
    # Evaluate the model after initial training
    logger.info("Evaluating model after initial fine-tuning...")
    pred_labels, true_labels, price_predictions, actual_price_changes = evaluate_model(model, test_dataloader, device)
    
    # RLHF starts from here
    # 1. Collect human feedback
    logger.info("Starting RLHF process...")
    chosen_texts, rejected_texts = collect_human_feedback(model, tokenizer, augmented_texts, augmented_labels, device)
    
    if len(chosen_texts) > 10:  # Make sure we have enough feedback
        # 2. Train a reward model
        reward_model = train_reward_model(chosen_texts, rejected_texts, tokenizer, device)
        
        # 3. Fine-tune the model with PPO
        rlhf_model = rlhf_ppo_training(model, reward_model, train_texts, tokenizer, device)
        
        # 4. Evaluate the RLHF model
        logger.info("Evaluating RLHF-tuned model...")
        rlhf_pred_labels, rlhf_true_labels, rlhf_price_predictions, rlhf_actual_price_changes = evaluate_model(rlhf_model, test_dataloader, device)
        
        # Save the RLHF model
        output_dir = './improved_finbert_rlhf_model'
        os.makedirs(output_dir, exist_ok=True)
        tokenizer.save_pretrained(output_dir)
        torch.save(rlhf_model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
        logger.info(f"RLHF model saved to {output_dir}")
        
        # Use RLHF model for final results
        model = rlhf_model
        pred_labels = rlhf_pred_labels
        true_labels = rlhf_true_labels
        price_predictions = rlhf_price_predictions
        actual_price_changes = rlhf_actual_price_changes
    else:
        logger.info("Not enough human feedback for RLHF. Using the model from initial training.")
    
    # Save model components separately
    output_dir = './improved_finbert_multitask_model'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    # Save model configuration
    model_config = {
        'hidden_size': base_model.config.hidden_size,
        'num_labels': 3,
    }
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(model_config, f)
    
    # Save state dictionary
    torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
    
    logger.info(f"Model saved to {output_dir}")
    
    # Create results dataframe for inspection
    results_df = pd.DataFrame({
        'Transcript': test_texts,
        'True Label': true_labels,
        'Predicted Label': pred_labels,
        'Sentiment Correct': [t == p for t, p in zip(true_labels, pred_labels)],
        'Actual Price Change': actual_price_changes,
        'Predicted Price Change': price_predictions,
        'Price Error': [abs(a - p) for a, p in zip(actual_price_changes, price_predictions)]
    })
    
    results_df.to_csv('improved_multitask_prediction_results.csv', index=False)
    logger.info("Predictions saved to improved_multitask_prediction_results.csv")

if __name__ == "__main__":
    main() 