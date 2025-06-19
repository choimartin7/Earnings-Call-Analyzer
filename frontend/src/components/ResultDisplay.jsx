import React from 'react'

const ResultDisplay = ({ prediction }) => {
  const { sentiment, confidence, price_change } = prediction
  
  // Helper functions
  const getSentimentClass = (sentiment) => {
    return sentiment.toLowerCase()
  }
  
  const getPriceChangeClass = (value) => {
    if (value > 0.5) return 'positive'
    if (value < -0.5) return 'negative'
    return 'neutral'
  }
  
  const formatPriceChange = (value) => {
    const sign = value > 0 ? '+' : ''
    return `${sign}${value.toFixed(2)}%`
  }
  
  const formatConfidence = (value) => {
    return `${(value * 100).toFixed(1)}%`
  }
  
  return (
    <div className="result-container">
      <div className="result-header">
        <h2>Analysis Results</h2>
      </div>
      
      <div className="result-section">
        <h3>Predicted Sentiment</h3>
        <div 
          className={`sentiment-badge ${getSentimentClass(sentiment)}`}
        >
          {sentiment}
        </div>
      </div>
      
      <div className="result-section">
        <h3>1-Day Price Change Prediction</h3>
        <div className={`price-value ${getPriceChangeClass(price_change)}`}>
          {formatPriceChange(price_change)}
        </div>
      </div>
      
      <div className="result-section">
        <h3>Sentiment Confidence</h3>
        <div className="confidence-bars">
          <div className="confidence-bar-item">
            <span className="confidence-label">Bearish</span>
            <div className="confidence-bar-container">
              <div 
                className="confidence-bar bearish" 
                style={{ width: `${confidence[0] * 100}%` }}
              ></div>
            </div>
            <span className="confidence-value">{formatConfidence(confidence[0])}</span>
          </div>
          
          <div className="confidence-bar-item">
            <span className="confidence-label">Neutral</span>
            <div className="confidence-bar-container">
              <div 
                className="confidence-bar neutral" 
                style={{ width: `${confidence[1] * 100}%` }}
              ></div>
            </div>
            <span className="confidence-value">{formatConfidence(confidence[1])}</span>
          </div>
          
          <div className="confidence-bar-item">
            <span className="confidence-label">Bullish</span>
            <div className="confidence-bar-container">
              <div 
                className="confidence-bar bullish" 
                style={{ width: `${confidence[2] * 100}%` }}
              ></div>
            </div>
            <span className="confidence-value">{formatConfidence(confidence[2])}</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ResultDisplay 