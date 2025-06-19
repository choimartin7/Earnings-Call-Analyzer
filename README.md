PS: Most the files are simply here for hosting the backend server using this git repository. If you want to run the whole yourself, just download the 3 folders (backend, frontend, final model) and follow the instructions below.
# FinBERT Earnings Call Analyzer UI

A web interface for analyzing earnings call transcripts using the fine-tuned FinBERT model. This application predicts sentiment and 1-day price changes from earnings call transcripts.

## Features

- Upload earnings call transcript files (TXT, PDF, DOC, DOCX)
- Paste transcript text directly
- Get sentiment predictions (Bullish, Neutral, Bearish) with confidence scores
- Get 1-day price change predictions
- Modern, responsive UI

## Project Structure

The project is divided into two main parts:

- `backend`: Flask API server that loads the FinBERT model and processes transcript data
- `frontend`: React application built with Vite that provides the user interface

## Setup and Installation

### Prerequisites

- Python 3.8+ with pip
- Node.js 14+ with npm
- The fine-tuned FinBERT model from the training process

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd UI-Interface/backend
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the Flask server:
   ```bash
   python app.py
   ```

   The server will run on http://localhost:5000

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd UI-Interface/frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

   The application will be available at http://localhost:3000

## Using the Application

1. Ensure both the backend and frontend servers are running
2. Open http://localhost:3000 in your web browser
3. Choose either "Upload File" or "Paste Text" to input an earnings call transcript
4. Click "Analyze Transcript" to get predictions
5. View the results showing sentiment classification and price change prediction

## API Endpoints

The backend provides the following API endpoints:

- `GET /api/health`: Check if the server and model are running
- `POST /api/predict`: Submit a transcript for analysis
  - Accepts form data with either a `file` field (file upload) or a `text` field (pasted text)
  - Returns JSON with sentiment prediction, confidence scores, and price change prediction

## Troubleshooting

- **Model Loading Issues**: Make sure the model files exist in the expected location and are accessible
- **CORS Errors**: The backend has CORS enabled, but if you encounter issues, check your browser's CORS settings
- **File Upload Problems**: Ensure files are in one of the supported formats (TXT, PDF, DOC, DOCX)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
