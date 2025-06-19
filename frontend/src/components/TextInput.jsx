import { useState, useEffect } from 'react'

const TextInput = ({ onTextChange, text }) => {
  const [charCount, setCharCount] = useState(0)
  
  useEffect(() => {
    setCharCount(text.length)
  }, [text])
  
  const handleChange = (e) => {
    const newText = e.target.value
    onTextChange(newText)
    setCharCount(newText.length)
  }
  
  return (
    <div className="text-input-container">
      <textarea
        placeholder="Paste earnings call transcript here..."
        value={text}
        onChange={handleChange}
      />
      <div className={`char-count ${charCount < 50 ? 'warning' : ''}`}>
        {charCount} characters {charCount < 50 ? '(minimum 50 required)' : ''}
      </div>
    </div>
  )
}

export default TextInput 