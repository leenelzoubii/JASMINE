'use client';

import { useState, useRef, useEffect } from 'react';
import { Upload, FileText, Clock, CheckCircle, RefreshCw, Download, X, Folder } from 'lucide-react';
import { motion } from 'framer-motion';
import { getPatients } from '@/lib/patients';
import { getCurrentUser } from '@/lib/auth';
import { Patient } from '@/lib/patients';

const statusColors: Record<string, string> = {
  Completed: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
  Pending: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400',
};

const riskColors: Record<string, string> = {
  High: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400',
  Moderate: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400',
  Low: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
  Unknown: 'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400',
};

export default function ProfessionalAssessmentsPage() {
  const [selectedPatient, setSelectedPatient] = useState('');
  const [patients, setPatients] = useState<Patient[]>([]);
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [loadingPatients, setLoadingPatients] = useState(true);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const user = getCurrentUser();
    if (user) {
      getPatients(user.id)
        .then(setPatients)
        .catch(console.error)
        .finally(() => setLoadingPatients(false));
    } else {
      setLoadingPatients(false);
    }
  }, []);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const files = Array.from(e.dataTransfer.files);
    const validFiles = files.filter(file =>
      file.type === 'application/json' ||
      file.type === 'text/csv' ||
      file.type === 'video/mp4'
    );
    setUploadedFiles(prev => [...prev, ...validFiles]);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.currentTarget.files || []);
    setUploadedFiles(prev => [...prev, ...files]);
  };

  const removeFile = (index: number) => {
    setUploadedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleUpload = async () => {
    if (!selectedPatient || uploadedFiles.length === 0) {
      alert('Please select a patient and upload at least one file');
      return;
    }

    setUploading(true);

    setTimeout(() => {
      setUploading(false);
      setUploadedFiles([]);
      alert(`Successfully processed ${uploadedFiles.length} file(s) for patient ${selectedPatient}`);
    }, 2000);
  };

  const getFileIcon = (file: File) => {
    if (file.type === 'video/mp4') return '🎬';
    if (file.type === 'text/csv') return '📊';
    if (file.type === 'application/json') return '{ }';
    return '📄';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Assessments</h1>
        <p className="text-gray-500 dark:text-gray-400">Run and manage screenings</p>
      </div>

      {/* New Assessment Card */}
      <div className="p-6 bg-white dark:bg-dark-surface rounded-2xl border border-gray-200 dark:border-dark-deep">
        <div className="space-y-6">
          <div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Run New Assessment</h3>

            {/* Patient Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Select Patient
              </label>
              {loadingPatients ? (
                <div className="w-full px-4 py-3 bg-gray-50 dark:bg-dark-bg border border-gray-200 dark:border-dark-deep rounded-xl flex items-center gap-2">
                  <div className="w-4 h-4 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
                  <span className="text-gray-500">Loading patients...</span>
                </div>
              ) : patients.length === 0 ? (
                <div className="w-full px-4 py-3 bg-gray-50 dark:bg-dark-bg border border-gray-200 dark:border-dark-deep rounded-xl text-gray-500">
                  No patients found. Add a patient first.
                </div>
              ) : (
                <select
                  value={selectedPatient}
                  onChange={(e) => setSelectedPatient(e.target.value)}
                  className="w-full px-4 py-3 bg-gray-50 dark:bg-dark-bg border border-gray-200 dark:border-dark-deep rounded-xl focus:outline-none focus:ring-2 focus:ring-primary"
                >
                  <option value="">Choose a patient...</option>
                  {patients.map((patient) => (
                    <option key={patient.id} value={patient.name}>
                      {patient.name}
                    </option>
                  ))}
                </select>
              )}
            </div>

            {/* File Upload Area */}
            {selectedPatient && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-6"
              >
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Upload Assessment Files
                </label>
                <p className="text-xs text-gray-500 dark:text-gray-400 mb-3">
                  Supported formats: MP4 (video), CSV (pose data), JSON (OpenPose)
                </p>

                {/* Drag & Drop Zone */}
                <div
                  onDragEnter={handleDrag}
                  onDragLeave={handleDrag}
                  onDragOver={handleDrag}
                  onDrop={handleDrop}
                  className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-colors ${
                    dragActive
                      ? 'border-primary bg-primary/5'
                      : 'border-gray-200 dark:border-dark-deep bg-gray-50 dark:bg-dark-bg'
                  }`}
                >
                  <div className="flex flex-col items-center">
                    <Upload className={`w-12 h-12 mb-3 ${dragActive ? 'text-primary' : 'text-gray-400'}`} />
                    <p className="font-medium text-gray-900 dark:text-white">
                      {dragActive ? 'Drop files here' : 'Drag files here or click to select'}
                    </p>
                    <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                      You can upload multiple files or entire folders
                    </p>
                  </div>

                  {/* Hidden file inputs */}
                  <input
                    ref={fileInputRef}
                    type="file"
                    multiple
                    accept=".mp4,.csv,.json"
                    onChange={handleFileSelect}
                    className="hidden"
                  />
                  <input
                    ref={folderInputRef}
                    type="file"
                    multiple
                    onChange={handleFileSelect}
                    className="hidden"
                    {...{ webkitdirectory: 'true' as unknown as string, directory: 'true' as unknown as string }}
                  />

                  {/* Click area */}
                  <div
                    onClick={() => fileInputRef.current?.click()}
                    className="absolute inset-0 rounded-xl cursor-pointer"
                  />

                  {/* Button overlay */}
                  <div className="absolute bottom-4 right-4 flex gap-2">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        fileInputRef.current?.click();
                      }}
                      className="px-4 py-2 bg-primary hover:bg-primary-dark text-white text-sm font-medium rounded-lg transition-colors"
                    >
                      Select Files
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        folderInputRef.current?.click();
                      }}
                      className="px-4 py-2 bg-gray-200 hover:bg-gray-300 dark:bg-dark-deep dark:hover:bg-dark-deep/80 text-gray-900 dark:text-white text-sm font-medium rounded-lg transition-colors flex items-center gap-2"
                    >
                      <Folder className="w-4 h-4" />
                      Select Folder
                    </button>
                  </div>
                </div>

                {/* Uploaded Files List */}
                {uploadedFiles.length > 0 && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mt-4"
                  >
                    <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Uploaded Files ({uploadedFiles.length})
                    </p>
                    <div className="space-y-2 max-h-40 overflow-y-auto">
                      {uploadedFiles.map((file, index) => (
                        <div
                          key={index}
                          className="flex items-center justify-between p-3 bg-gray-50 dark:bg-dark-bg border border-gray-200 dark:border-dark-deep rounded-lg"
                        >
                          <div className="flex items-center gap-3 flex-1 min-w-0">
                            <span className="text-lg">{getFileIcon(file)}</span>
                            <div className="min-w-0">
                              <p className="text-sm font-medium text-gray-900 dark:text-white truncate">
                                {file.name}
                              </p>
                              <p className="text-xs text-gray-500 dark:text-gray-400">
                                {(file.size / 1024 / 1024).toFixed(2)} MB
                              </p>
                            </div>
                          </div>
                          <button
                            onClick={() => removeFile(index)}
                            className="p-1 text-gray-400 hover:text-red-500 transition-colors ml-2"
                          >
                            <X className="w-5 h-5" />
                          </button>
                        </div>
                      ))}
                    </div>
                  </motion.div>
                )}

                {/* Action Buttons */}
                <div className="flex gap-3 mt-6">
                  <button
                    onClick={handleUpload}
                    disabled={uploadedFiles.length === 0 || uploading}
                    className="flex-1 px-6 py-3 bg-primary hover:bg-primary-dark text-white font-medium rounded-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                  >
                    {uploading ? (
                      <>
                        <RefreshCw className="w-5 h-5 animate-spin" />
                        Processing...
                      </>
                    ) : (
                      <>
                        <Upload className="w-5 h-5" />
                        Run Assessment
                      </>
                    )}
                  </button>
                  {uploadedFiles.length > 0 && (
                    <button
                      onClick={() => setUploadedFiles([])}
                      className="px-6 py-3 border border-gray-200 dark:border-dark-deep text-gray-700 dark:text-gray-300 font-medium rounded-xl hover:bg-gray-50 dark:hover:bg-dark-deep transition-colors"
                    >
                      Clear
                    </button>
                  )}
                </div>
              </motion.div>
            )}
          </div>
        </div>
      </div>

      {/* Recent Assessments */}
      <div className="bg-white dark:bg-dark-surface rounded-2xl border border-gray-200 dark:border-dark-deep overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-100 dark:border-dark-deep">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Recent Assessments</h2>
        </div>
        <div className="px-6 py-12 text-center">
          <p className="text-gray-500 dark:text-gray-400">No recent assessments. Run an assessment to see results here.</p>
        </div>
      </div>
    </div>
  );
}