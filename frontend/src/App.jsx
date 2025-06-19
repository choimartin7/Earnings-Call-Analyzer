import { useState, useEffect } from 'react'
import './App.css'
import FileUpload from './components/FileUpload'
import TextInput from './components/TextInput'
import ResultDisplay from './components/ResultDisplay'
import axios from 'axios'

function App() {
  const [inputType, setInputType] = useState('file') // 'file' or 'text'
  const [file, setFile] = useState(null)
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [prediction, setPrediction] = useState(null)
  const [error, setError] = useState(null)
  const [serverStatus, setServerStatus] = useState('checking') // 'checking', 'online', 'offline'

  // Check server status on component mount
  useEffect(() => {
    const checkServerStatus = async () => {
      try {
        const response = await axios.get('/api/health')
        if (response.status === 200) {
          setServerStatus('online')
        } else {
          setServerStatus('offline')
        }
      } catch (err) {
        console.error('Server health check failed:', err)
        setServerStatus('offline')
      }
    }

    checkServerStatus()
  }, [])

  const handleFileChange = (uploadedFile) => {
    setFile(uploadedFile)
    setPrediction(null)
    setError(null)
  }

  const handleTextChange = (inputText) => {
    setText(inputText)
    setPrediction(null)
    setError(null)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    if (inputType === 'file' && !file) {
      setError('Please select a file')
      return
    }
    
    if (inputType === 'text' && (!text || text.trim().length < 50)) {
      setError('Please enter transcript text (minimum 50 characters)')
      return
    }
    
    setLoading(true)
    setError(null)
    
    const formData = new FormData()
    
    if (inputType === 'file') {
      formData.append('file', file)
    } else {
      formData.append('text', text)
    }
    
    try {
      const response = await axios.post('/api/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })
      
      setPrediction(response.data)
    } catch (err) {
      console.error('Prediction error:', err)
      setError(err.response?.data?.error || 'An error occurred during analysis')
    } finally {
      setLoading(false)
    }
  }

  const handleInputTypeChange = (type) => {
    setInputType(type)
    setPrediction(null)
    setError(null)
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>FinBERT Earnings Call Analyzer</h1>
        <p className="app-description">
          Upload an earnings call transcript or paste text to analyze sentiment and predict price change
        </p>
        {serverStatus === 'checking' && (
          <div className="server-status checking">Checking server status...</div>
        )}
        {serverStatus === 'offline' && (
          <div className="server-status offline">
            Server is offline. Please ensure the backend server is running.
          </div>
        )}
      </header>
      
      <main className="app-main">
        <div className="input-container">
          <div className="input-type-selector">
            <button
              className={inputType === 'file' ? 'active' : ''}
              onClick={() => handleInputTypeChange('file')}
            >
              Upload File
            </button>
            <button
              className={inputType === 'text' ? 'active' : ''}
              onClick={() => handleInputTypeChange('text')}
            >
              Paste Text
            </button>
          </div>
          
          <form onSubmit={handleSubmit}>
            {inputType === 'file' ? (
              <FileUpload onFileChange={handleFileChange} file={file} />
            ) : (
              <TextInput onTextChange={handleTextChange} text={text} />
            )}
            
            {error && <div className="error-message">{error}</div>}
            
            <button 
              type="submit" 
              className="submit-button"
              disabled={loading || serverStatus !== 'online' || (inputType === 'file' && !file) || (inputType === 'text' && (!text || text.trim().length < 50))}
            >
              {loading ? 'Analyzing...' : 'Analyze Transcript'}
            </button>
          </form>
        </div>
        
        {prediction && (
          <ResultDisplay prediction={prediction} />
        )}
      </main>
      
      <footer className="app-footer">
        <p>Powered by FinBERT with RLHF fine-tuning</p>
      </footer>
    </div>
  )
}

export default App 