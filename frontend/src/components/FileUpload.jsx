import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'

const FileUpload = ({ onFileChange, file }) => {
  const [dragActive, setDragActive] = useState(false)

  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles && acceptedFiles[0]) {
      onFileChange(acceptedFiles[0])
    }
  }, [onFileChange])

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    accept: {
      'text/plain': ['.txt'],
      'application/pdf': ['.pdf'],
      'application/msword': ['.doc'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx']
    },
    multiple: false,
    onDragEnter: () => setDragActive(true),
    onDragLeave: () => setDragActive(false),
    onDropAccepted: () => setDragActive(false),
    onDropRejected: () => setDragActive(false)
  })

  return (
    <div>
      <div 
        {...getRootProps()} 
        className={`dropzone ${dragActive ? 'active' : ''}`}
      >
        <input {...getInputProps()} />
        <div className="dropzone-content">
          <div className="dropzone-icon">📄</div>
          <p>Drag & drop an earnings call transcript file, or click to select</p>
          <span className="dropzone-hint">Accepts .txt, .pdf, .doc, .docx</span>
        </div>
      </div>
      
      {file && (
        <div className="file-info">
          <span className="file-icon">📎</span>
          <span className="file-name">{file.name}</span>
          <span className="file-size">({(file.size / 1024).toFixed(1)} KB)</span>
        </div>
      )}
    </div>
  )
}

export default FileUpload 