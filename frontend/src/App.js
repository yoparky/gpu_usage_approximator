import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { AlertCircle, HelpCircle, ChevronDown, ChevronUp, Server, Brain, MessageSquare, Zap, Settings, ThumbsUp, HardDrive, Bot } from 'lucide-react';        
import GpuAutosuggestSearch from './modules/GpuAutosuggestSearch';
import { API_URL } from './config';

const VLLMMemoryCalculator = () => {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showArchitecture, setShowArchitecture] = useState(true);
  
  const [formData, setFormData] = useState({
    model_name: "",
    tensor_parallel_size: 1,
    current_gpu_memory_size_gb: 24,
    max_seq_len: 4096,
    params_billions: null,
    token: "",
    max_num_seqs: 256,
    dtype: "",
    kv_cache_dtype: "",
    quantization: "",
    curr_gpu_use: 0,
    architecture: {
      num_layers: null,
      hidden_size: null,
      num_heads: null,
      head_dim: null,
      intermediate_size: null,
      num_key_value_heads: null
    }
  });

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    
    if (name.includes('.')) {
      const [parent, child] = name.split('.');
      setFormData({
        ...formData,
        [parent]: {
          ...formData[parent],
          [child]: value === '' ? null : (isNaN(value) ? value : Number(value))
        }
      });
    } else {
      setFormData({
        ...formData,
        [name]: value === '' ? '' : (isNaN(value) ? value : Number(value))
      });
    }
  };

  // GpuAutosuggestSearch 컴포넌트 용
  const handleGpuSelect = (selectedGpu) => {
    // memorySize 받거나 없으면 24 GB 디폴트
    const memorySize = selectedGpu && selectedGpu.memSize 
      ? parseFloat(selectedGpu.memSize) 
      : 24;
    
    // GPU memory size로 formData 업데이트
    setFormData({
      ...formData,
      current_gpu_memory_size_gb: memorySize
    });
  };

  const clearArchitectureValues = () => {
    setFormData({
      ...formData,
      architecture: {
        num_layers: null,
        hidden_size: null,
        num_heads: null,
        head_dim: null,
        intermediate_size: null,
        num_key_value_heads: null
      }
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!formData.model_name.trim()) {
      setError("모델 이름은 필수입니다.");
      return;
    }
    
    // 모델 이름 없으면, parameter count 필요
    if (!formData.model_name.trim() && !formData.params_billions) {
      setError("모델 이름이 없는 경우 파라미터 수를 입력해야 합니다.");
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      // non-null 만 포함
      const cleanArchitecture = {};
      Object.entries(formData.architecture).forEach(([key, value]) => {
        if (value !== null && value !== '') {
          cleanArchitecture[key] = value;
        }
      });
      
      const payload = {
        ...formData,
        architecture: Object.keys(cleanArchitecture).length > 0 ? cleanArchitecture : null,
        // 디폴트값 설정
        tensor_parallel_size: formData.tensor_parallel_size || 1,
        current_gpu_memory_size_gb: formData.current_gpu_memory_size_gb || 24,
        max_seq_len: formData.max_seq_len || 4096,
        max_num_seqs: formData.max_num_seqs || 256,
        curr_gpu_use: formData.curr_gpu_use || 0
      };
      
      // 없음 지움
      if (!formData.dtype) {
        delete payload.dtype;
      }
      
      if (!formData.kv_cache_dtype) {
        delete payload.kv_cache_dtype;
      }
      
      if (payload.quantization === '') {
        payload.quantization = null;
      }

      console.log("Request payload:", JSON.stringify(payload, null, 2));
      
      setTimeout(async () => {
        const response = await fetch(`${API_URL}/estimate-gpu-memory`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        
        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }
        
        const data = await response.json();
        setResult(data);
        setLoading(false);
      }, 1000);
      
    } catch (err) {
      setError(err.message || "오류가 발생했습니다. 다시 시도해주세요.");
      setLoading(false);
    }
  };

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8'];
  
  const formatMemoryData = (result) => {
    if (!result) return [];
    
    return [
      { name: '모델 파라미터', value: result.components_breakdown.model_params },
      { name: '활성화 메모리', value: result.components_breakdown.activation },
      { name: 'KV 캐시', value: result.components_breakdown.min_kv_cache },
      { name: 'CUDA 오버헤드', value: result.components_breakdown.cuda_overhead }
    ];
  };
  
  const getUtilizationColor = (utilization) => {
    if (utilization > 0 && utilization < 0.5) return 'text-green-500';
    if (utilization > 0 && utilization < 0.8) return 'text-yellow-500';
    return 'text-red-500';
  };

  // 모델 요구 최소 메모리가 현재 GPU 메모리보다 큰지 확인
  const isMemoryExceededModel = () => {
    if (!result) return false;
    return result.per_gpu_total_min_memory_gb > result.current_gpu_memory_size_gb;
  };
  // 전체 요구 최소 메모리가 현재 GPU 메모리보다 큰지 확인
  const isMemoryExceededTotal = () => {
    if (!result) return false;
    return result.model_memory_total_use_ratio > 1;
  };
  
  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-800 mb-2">vLLM 0.7.3 GPU 메모리 계산기</h1>
        <p className="text-gray-600">LLM 모델 서빙을 위한 GPU 메모리 요구 사항을 추정합니다</p>
        <div className="mt-2 text-sm text-gray-500">
          <code className="bg-gray-100 p-1 rounded">POST /estimate-gpu-memory</code>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        <div className="md:col-span-1 bg-gray-50 p-6 rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4 flex items-center">
            <Settings className="mr-2 h-5 w-5" />
            모델 설정
          </h2>
          
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                모델 이름 <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                name="model_name"
                value={formData.model_name}
                onChange={handleInputChange}
                className="w-full p-2 border border-gray-300 rounded-md"
                placeholder="예: meta-llama/Llama-3-8b"
                required
              />
              <p className="text-xs text-gray-500 mt-1">필수 입력 항목</p>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                파라미터 수 (십억)
              </label>
              <input
                type="number"
                name="params_billions"
                value={formData.params_billions || ''}
                onChange={handleInputChange}
                className="w-full p-2 border border-gray-300 rounded-md"
                placeholder="예: 7"
                min="0.01"
                step="0.01"
              />
              <p className="text-xs text-gray-500 mt-1">
                {!formData.model_name.trim() ? "모델 이름이 없는 경우 필수" : "선택 사항"}
              </p>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                최대 시퀀스 길이
              </label>
              <input
                type="number"
                name="max_seq_len"
                value={formData.max_seq_len}
                onChange={handleInputChange}
                className="w-full p-2 border border-gray-300 rounded-md"
                placeholder="기본값: 4096"
                min="1"
              />
              <p className="text-xs text-gray-500 mt-1">기본값: 4096</p>
            </div>   
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                텐서 병렬 크기
              </label>
              <input
                type="number"
                name="tensor_parallel_size"
                value={formData.tensor_parallel_size}
                onChange={handleInputChange}
                className="w-full p-2 border border-gray-300 rounded-md"
                placeholder="기본값: 1"
                min="1"
              />
              <p className="text-xs text-gray-500 mt-1">기본값: 1</p>
            </div>
            
            {/* Replace the GPU memory input with GpuAutosuggestSearch */}
            <div>
              <GpuAutosuggestSearch onGpuSelect={handleGpuSelect} />
            </div>
            
            {/* Display the current GPU memory from the selected GPU as read-only */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                현재 GPU 메모리 크기 (GB)
              </label>
              <input
                type="number"
                name="current_gpu_memory_size_gb"
                value={formData.current_gpu_memory_size_gb}
                readOnly
                className="w-full p-2 border border-gray-300 rounded-md bg-gray-100"
              />
              <p className="text-xs text-gray-500 mt-1">GPU 선택에 따라 자동 설정됨</p>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                현재 사용 중인 GPU 메모리 (GB)
              </label>
              <input
                type="number"
                name="curr_gpu_use"
                value={formData.curr_gpu_use}
                onChange={handleInputChange}
                className="w-full p-2 border border-gray-300 rounded-md"
                placeholder="기본값: 0"
                min="0"
                step="0.1"
              />
              <p className="text-xs text-gray-500 mt-1">기본값: 0 GB</p>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Hugging Face 토큰 (선택 사항)
              </label>
              <input
                type="password"
                name="token"
                value={formData.token}
                onChange={handleInputChange}
                className="w-full p-2 border border-gray-300 rounded-md"
                placeholder="게이티드 모델용 토큰"
              />
            </div>
            
            <div className="border-t border-gray-200 pt-4">
              <button 
                type="button" 
                className="flex items-center text-blue-600 hover:text-blue-800"
                onClick={() => setShowAdvanced(!showAdvanced)}
              >
                {showAdvanced ? <ChevronUp className="h-4 w-4 mr-1" /> : <ChevronDown className="h-4 w-4 mr-1" />}
                고급 옵션
              </button>
              
              {showAdvanced && (
                <div className="mt-3 space-y-3">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      최대 시퀀스 수 (max_num_seqs)
                    </label>
                    <input
                      type="number"
                      name="max_num_seqs"
                      value={formData.max_num_seqs}
                      onChange={handleInputChange}
                      className="w-full p-2 border border-gray-300 rounded-md"
                      placeholder="기본값: 256"
                      min="1"
                    />
                    <p className="text-xs text-gray-500 mt-1">기본값: 256</p>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      데이터 타입 (dtype)
                    </label>
                    <select 
                      name="dtype" 
                      value={formData.dtype}
                      onChange={handleInputChange}
                      className="w-full p-2 border border-gray-300 rounded-md"
                    >
                      <option value="">선택 안함 (기본값 사용)</option>
                      <option value="float16">float16</option>
                      <option value="float32">float32</option>
                      <option value="bfloat16">bfloat16</option>
                      <option value="int8">int8</option>
                    </select>
                    <p className="text-xs text-gray-500 mt-1">선택 사항입니다. 허깅페이스 Config 파일을 기반으로 설정됩니다.</p>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      KV 캐시 데이터 타입
                    </label>
                    <select 
                      name="kv_cache_dtype" 
                      value={formData.kv_cache_dtype}
                      onChange={handleInputChange}
                      className="w-full p-2 border border-gray-300 rounded-md"
                    >
                      <option value="">선택 안함 (기본값 사용)</option>
                      <option value="float16">float16</option>
                      <option value="float32">float32</option>
                      <option value="bfloat16">bfloat16</option>
                      <option value="int8">int8</option>
                    </select>
                    <p className="text-xs text-gray-500 mt-1">선택 사항입니다. 허깅페이스 Config 파일을 기반으로 설정됩니다.</p>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      양자화
                    </label>
                    <select 
                      name="quantization" 
                      value={formData.quantization}
                      onChange={handleInputChange}
                      className="w-full p-2 border border-gray-300 rounded-md"
                    >
                      <option value="">없음</option>
                      <option value="int8">int8</option>
                      <option value="int4">int4</option>
                      <option value="int1">int1</option>
                    </select>
                  </div>
                  
                  <div className="pt-2">
                    <div className="flex items-center justify-between">
                      <button 
                        type="button" 
                        className="flex items-center text-blue-600 hover:text-blue-800 font-medium"
                        onClick={() => setShowArchitecture(!showArchitecture)}
                      >
                        {showArchitecture ? <ChevronUp className="h-4 w-4 mr-1" /> : <ChevronDown className="h-4 w-4 mr-1" />}
                        모델 아키텍처 세부 정보
                      </button>
                      
                      {showArchitecture && (
                        <button
                          type="button"
                          onClick={clearArchitectureValues}
                          className="text-xs text-gray-500 hover:text-gray-700"
                        >
                          초기화
                        </button>
                      )}
                    </div>
                    
                    {showArchitecture && (
                      <div className="mt-3 space-y-3 bg-gray-50 p-4 rounded-md border border-gray-200">
                        <p className="text-sm text-gray-600 mb-2">
                          모델 아키텍처 매개변수를 직접 설정할 수 있습니다.
                        </p>
                        
                        <div className="grid grid-cols-1 gap-3">
                          <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1 break-words">
                              레이어 수 (num_layers)
                            </label>
                            <input
                              type="number"
                              name="architecture.num_layers"
                              value={formData.architecture.num_layers || ''}
                              onChange={handleInputChange}
                              className="w-full p-2 border border-gray-300 rounded-md"
                              placeholder="예: 24"
                              min="1"
                            />
                          </div>
                          
                          <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1 break-words">
                              히든 차원 (hidden_size)
                            </label>
                            <input
                              type="number"
                              name="architecture.hidden_size"
                              value={formData.architecture.hidden_size || ''}
                              onChange={handleInputChange}
                              className="w-full p-2 border border-gray-300 rounded-md"
                              placeholder="예: 2048"
                              min="1"
                            />
                          </div>
                          
                          <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1 break-words">
                              어텐션 헤드 수 (num_heads)
                            </label>
                            <input
                              type="number"
                              name="architecture.num_heads"
                              value={formData.architecture.num_heads || ''}
                              onChange={handleInputChange}
                              className="w-full p-2 border border-gray-300 rounded-md"
                              placeholder="예: 16"
                              min="1"
                            />
                          </div>
                          
                          <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1 break-words">
                              헤드 차원 (head_dim)
                            </label>
                            <input
                              type="number"
                              name="architecture.head_dim"
                              value={formData.architecture.head_dim || ''}
                              onChange={handleInputChange}
                              className="w-full p-2 border border-gray-300 rounded-md"
                              placeholder="예: 128"
                              min="1"
                            />
                          </div>
                          
                          <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1 break-words">
                              중간 크기 (intermediate_size)
                            </label>
                            <input
                              type="number"
                              name="architecture.intermediate_size"
                              value={formData.architecture.intermediate_size || ''}
                              onChange={handleInputChange}
                              className="w-full p-2 border border-gray-300 rounded-md"
                              placeholder="예: 5632"
                              min="1"
                            />
                          </div>
                          
                          <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1 break-words">
                              KV 헤드 수 (num_key_value_heads)
                            </label>
                            <input
                              type="number"
                              name="architecture.num_key_value_heads"
                              value={formData.architecture.num_key_value_heads || ''}
                              onChange={handleInputChange}
                              className="w-full p-2 border border-gray-300 rounded-md"
                              placeholder="예: 16"
                              min="1"
                            />
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
            
            <button
              type="submit"
              className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded-md font-medium flex items-center justify-center"
              disabled={loading}
            >
              {loading ? '계산 중...' : '메모리 계산하기'}
            </button>
            
            <p className="mt-2 text-xs text-center text-gray-500">
              <span className="text-red-500">*</span> 표시는 필수 입력 항목입니다
            </p>
          </form>
        </div>
        
        <div className="md:col-span-2">
          {error && (
            <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-6">
              <div className="flex items-center">
                <AlertCircle className="h-5 w-5 text-red-500 mr-2" />
                <p className="text-red-700">{error}</p>
              </div>
            </div>
          )}
          
          {result ? (
            <div className="bg-white rounded-lg shadow">
              <div className={`bg-gray-50 p-6 rounded-t-lg border-b ${(result.estimated_values && result.estimated_values.length > 0) || isMemoryExceededTotal() ? 'border-red-500 bg-red-50' : 'border-gray-200'}`}>
                <h2 className="text-2xl font-bold flex items-center">
                  <Bot className="mr-2 h-6 w-6 text-blue-600" />
                  {result.model_name}
                </h2>
                <div className="mt-2 text-gray-600 flex flex-wrap items-center gap-2">
                  <span className="flex items-center">
                    <Brain className="h-4 w-4 mr-1" />
                    {result.model_params_count ? result.model_params_count.toFixed(1) : '?'}B 파라미터
                  </span>
                  <span className="mx-2">•</span>
                  <span className="flex items-center">
                    <MessageSquare className="h-4 w-4 mr-1" />
                    컨텍스트 길이: {result.context_length.toLocaleString()}
                  </span>
                  <span className="mx-2">•</span>
                  <span className="flex items-center">
                    <Zap className="h-4 w-4 mr-1" />
                    {result.dtype} / {result.max_num_seqs} 시퀀스 동시 처리
                  </span>
                  <span className="flex items-center">
                    <ThumbsUp className="h-4 w-4 mr-1" />
                    최소 gpu_memory_utilization 값: {result.recommended_gpu_memory_utilization}
                  </span>
                  <span className="mx-2">•</span>
                  {result.tensor_parallel_size > 1 ?
                    (<span className="flex items-center">
                      <Server className="h-4 w-4 mr-1" />
                      GPU 병렬화: {result.tensor_parallel_size}개
                    </span>) : 
                    (<span className="flex items-center">
                      <HardDrive className="h-4 w-4 mr-1" />
                      GPU 병렬화: {result.tensor_parallel_size}개
                    </span>)
                  }
                </div>
                
                {/* GPU Memory Utilization Notification */}
                <div className="mt-3 flex items-center">
                  <div className={`px-3 py-1.5 rounded-lg flex items-center ${
                    (isMemoryExceededModel() ? 1 : result.recommended_gpu_memory_utilization) < 0.5 
                      ? 'bg-green-100 text-green-800' 
                      : (isMemoryExceededModel() ? 1 : result.recommended_gpu_memory_utilization) < 0.85
                        ? 'bg-yellow-100 text-yellow-800' 
                        : 'bg-red-100 text-red-800'
                  }`}>
                    <Server className="h-4 w-4 mr-1.5" />
                    <span className="font-medium">모델의 최소 GPU 메모리 요구량:</span>
                    <span className="ml-1.5 font-bold">
                      {isMemoryExceededModel() ? 100 : (result.recommended_gpu_memory_utilization * 100).toFixed(0)}%
                    </span>
                    <div className="ml-2 w-16 bg-white bg-opacity-50 rounded-full h-1.5">
                      <div 
                        className={`h-1.5 rounded-full ${
                          (isMemoryExceededModel() ? 1 : result.recommended_gpu_memory_utilization) < 0.5 
                            ? 'bg-green-500' 
                            : (isMemoryExceededModel() ? 1 : result.recommended_gpu_memory_utilization) < 0.85 
                              ? 'bg-yellow-500' 
                              : 'bg-red-500'
                        }`}
                        style={{ width: `${isMemoryExceededModel() ? 100 : (result.recommended_gpu_memory_utilization * 100).toFixed(0)}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
                {/* GPU Total Memory Utilization Notification */}
                <div className="mt-3 flex items-center">
                  <div className={`px-3 py-1.5 rounded-lg flex items-center ${
                    (isMemoryExceededTotal() ? 1 : result.model_memory_total_use_ratio) < 0.5 
                      ? 'bg-green-100 text-green-800' 
                      : (isMemoryExceededTotal() ? 1 : result.model_memory_total_use_ratio) < 0.85
                        ? 'bg-yellow-100 text-yellow-800'
                        : 'bg-red-100 text-red-800'
                  }`}>
                    <Server className="h-4 w-4 mr-1.5" />
                    <span className="font-medium">모델 로드 후 GPU 메모리 최소 사용량:</span>
                    <span className="ml-1.5 font-bold">
                      {(result.model_memory_total_use_ratio * 100).toFixed(0)}%
                    </span>
                    <div className="ml-2 w-16 bg-white bg-opacity-50 rounded-full h-1.5">
                      <div 
                        className={`h-1.5 rounded-full ${
                          result.model_memory_total_use_ratio < 0.5 
                            ? 'bg-green-500' 
                            : result.model_memory_total_use_ratio < 0.85
                              ? 'bg-yellow-500' 
                              : 'bg-red-500'
                        }`}
                        style={{ width: `${isMemoryExceededTotal() ? 100 : (result.model_memory_total_use_ratio * 100).toFixed(0)}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
                
                {result.estimated_values && result.estimated_values.length > 0 && (
                  <div className="mt-3 text-sm text-red-600 flex items-start">
                    <AlertCircle className="h-4 w-4 mr-1 flex-shrink-0 mt-0.5" />
                    <div>
                      <span className="font-medium">주의:</span> 일부 값이 추정되었습니다. 정확한 모델 정보가 부족하여 결과가 실제와 다를 수 있습니다.
                    </div>
                  </div>
                )}
              </div>
              
              <div className="p-6">
                {/* Memory Exceeded Alert */}
                {isMemoryExceededTotal() && (
                  <div className="bg-red-50 border-l-4 border-red-600 p-4 mb-6">
                    <div className="flex items-start">
                      <AlertCircle className="h-6 w-6 text-red-600 mr-2 flex-shrink-0" />
                      <div>
                        <h3 className="text-lg font-bold text-red-700 mb-1">메모리 부족 경고</h3>
                        <p className="text-red-700">
                          현재 GPU 메모리 사용량({result.current_gpu_memory_use_gb.toFixed(1)} GB)이 사용 가능한 GPU 메모리({result.current_gpu_memory_size_gb.toFixed(1)} GB)를 초과합니다. 이 모델은 현재 GPU 구성으로 서빙할 수 없습니다.
                        </p>
                        <p className="text-red-700 mt-1 font-semibold">
                          GPU 당 메모리 초과량: {(result.current_gpu_memory_use_gb - result.current_gpu_memory_size_gb).toFixed(1)} GB
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {result.warning && (
                  <div className="bg-yellow-50 border-l-4 border-yellow-500 p-4 mb-6">
                    <div className="flex">
                      <AlertCircle className="h-5 w-5 text-yellow-500 mr-2 flex-shrink-0" />
                      <p className="text-yellow-700">{result.warning}</p>
                    </div>
                  </div>
                )}
                
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div>
                    <h3 className="text-lg font-semibold mb-4">메모리 요구 사항</h3>
                    {result.tensor_parallel_size === 1 && ( // 일반
                      <div className="bg-blue-50 rounded-lg p-5 mb-5">
                        <div className="text-center">
                          <div className="text-4xl font-bold text-blue-600">{result.total_min_memory_gb.toFixed(1)} GB</div>
                          <div className="text-sm text-gray-600 mt-1">총 필요한 최소 GPU 메모리</div>
                          
                          {result.current_gpu_memory_size_gb && ( // GPU Usage card at the top
                            <div className="mt-2">
                              <div className={`text-sm font-semibold ${getUtilizationColor(result.recommended_gpu_memory_utilization)}`}>
                                GPU 메모리의 {isMemoryExceededModel() ? 100 : (result.recommended_gpu_memory_utilization * 100).toFixed(0)}% 사용
                              </div>
                              <div className="w-full bg-gray-200 rounded-full h-2.5 mt-1">
                                <div
                                  className={`h-2.5 rounded-full ${
                                    (isMemoryExceededModel() ? 1 : result.recommended_gpu_memory_utilization) < 0.5 
                                      ? 'bg-green-500' 
                                      : (isMemoryExceededModel() ? 1 : result.recommended_gpu_memory_utilization) < 0.8 
                                        ? 'bg-yellow-500' 
                                        : 'bg-red-500'
                                  }`}
                                  style={{ width: `${isMemoryExceededModel() ? 100 : (result.recommended_gpu_memory_utilization * 100).toFixed(0)}%` }}
                                ></div>
                              </div>
                            </div>
                          )}
                        </div>
                        
                        <div className="mt-4 grid grid-cols-2 gap-3">
                          <div className="bg-white p-3 rounded-lg">
                            <div className="text-sm text-gray-600">모델 파라미터</div>
                            <div className="text-lg font-semibold">{result.components_breakdown.model_params.toFixed(1)} GB</div>
                          </div>
                          <div className="bg-white p-3 rounded-lg">
                            <div className="text-sm text-gray-600">활성화 메모리</div>
                            <div className="text-lg font-semibold">{result.components_breakdown.activation.toFixed(1)} GB</div>
                          </div>
                          <div className="bg-white p-3 rounded-lg">
                            <div className="text-sm text-gray-600">{result.context_length} 토큰 시퀸스 당 필요한 KV 캐시</div>
                            <div className="text-lg font-semibold">{result.components_breakdown.min_kv_cache.toFixed(1)} GB</div>
                          </div>
                          <div className="bg-white p-3 rounded-lg">
                            <div className="text-sm text-gray-600">CUDA 오버헤드</div>
                            <div className="text-lg font-semibold">{result.components_breakdown.cuda_overhead.toFixed(1)} GB</div>
                          </div>
                        </div>
                      </div>
                    )}
                    {result.tensor_parallel_size > 1 && ( // 병렬화
                      <div className="bg-indigo-50 rounded-lg p-5">
                        <h4 className="font-medium text-indigo-800 mb-2">GPU당 메모리 (텐서 병렬화)</h4>
                        <div className="text-center mb-3">
                          <div className="text-2xl font-bold text-indigo-600">{result.per_gpu_total_min_memory_gb.toFixed(1)} GB</div>
                          <div className="text-sm text-gray-600">GPU당 필요 메모리</div>
                        </div>
                        {result.current_gpu_memory_size_gb && ( // GPU Usage card at the top
                            <div className="mt-2">
                              <div className={`text-sm font-semibold ${getUtilizationColor(result.recommended_gpu_memory_utilization)}`}>
                                각 GPU 메모리의 {isMemoryExceededModel() ? 100 : (result.recommended_gpu_memory_utilization * 100).toFixed(0)}% 사용
                              </div>
                              <div className="w-full bg-gray-200 rounded-full h-2.5 mt-1">
                                <div 
                                  className={`h-2.5 rounded-full ${
                                    (isMemoryExceededModel() ? 1 : result.recommended_gpu_memory_utilization) < 0.5 
                                      ? 'bg-green-500' 
                                      : (isMemoryExceededModel() ? 1 : result.recommended_gpu_memory_utilization) < 0.8 
                                        ? 'bg-yellow-500' 
                                        : 'bg-red-500'
                                  }`}
                                  style={{ width: `${isMemoryExceededModel() ? 100 : (result.recommended_gpu_memory_utilization * 100).toFixed(0)}%` }}
                                ></div>
                              </div>
                            </div>
                          )}
                        <div className="mt-4 grid grid-cols-2 gap-3">
                          <div className="bg-white p-3 rounded-lg">
                            <div className="text-sm text-gray-600">모델 파라미터</div>
                            <div className="text-lg font-semibold">{result.per_gpu_memory_breakdown.per_gpu_model_params.toFixed(1)} GB</div>
                          </div>
                          <div className="bg-white p-3 rounded-lg">
                            <div className="text-sm text-gray-600">활성화 메모리</div>
                            <div className="text-lg font-semibold">{result.per_gpu_memory_breakdown.per_gpu_activation.toFixed(1)} GB</div>
                          </div>
                          <div className="bg-white p-3 rounded-lg">
                            <div className="text-sm text-gray-600">{result.context_length} 토큰 시퀸스 당 필요한 KV 캐시</div>
                            <div className="text-lg font-semibold">{result.per_gpu_memory_breakdown.per_gpu_min_kv_cache.toFixed(1)} GB</div>
                          </div>
                          <div className="bg-white p-3 rounded-lg">
                            <div className="text-sm text-gray-600">CUDA 오버헤드</div>
                            <div className="text-lg font-semibold">{result.per_gpu_memory_breakdown.per_gpu_cuda_overhead.toFixed(1)} GB</div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                  
                  <div>
                    <h3 className="text-lg font-semibold mb-4">메모리 분석</h3>
                    
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={formatMemoryData(result)}
                            cx="50%"
                            cy="50%"
                            innerRadius={70}
                            outerRadius={90}
                            paddingAngle={2}
                            dataKey="value"
                            nameKey="name"
                            label={({name, percent}) => `${name} ${(percent * 100).toFixed(0)}%`}
                            labelLine={false}
                          >
                            {formatMemoryData(result).map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                            ))}
                          </Pie>
                          <Tooltip formatter={(value) => `${value.toFixed(1)} GB`} />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                    
                    <div className="mt-6">
                      <h3 className="text-lg font-semibold mb-4">메모리 내역</h3>
                      <div className="h-64">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            data={formatMemoryData(result)}
                            layout="vertical"
                            margin={{ top: 5, right: 30, left: 120, bottom: 5 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis type="number" unit=" GB" />
                            <YAxis dataKey="name" type="category" width={100} />
                            <Tooltip formatter={(value) => `${value.toFixed(1)} GB`} />
                            <Legend />
                            <Bar dataKey="value" name="메모리 (GB)" fill="#8884d8" />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="mt-8">
                  <h3 className="text-lg font-semibold mb-4">계산에 사용된 모델 아키텍처</h3>
                  <div className={`bg-gray-50 p-4 rounded-lg ${result.estimated_values && result.estimated_values.length > 0 ? 'border-2 border-red-500' : 'border border-gray-200'}`}>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                      <div>
                        <div className="text-sm text-gray-600">레이어 수</div>
                        <div className="font-medium">{result.architecture_used.num_layers}</div>
                      </div>
                      <div>
                        <div className="text-sm text-gray-600">히든 차원</div>
                        <div className="font-medium">{result.architecture_used.hidden_size}</div>
                      </div>
                      <div>
                        <div className="text-sm text-gray-600">어텐션 헤드 수</div>
                        <div className="font-medium">{result.architecture_used.num_heads}</div>
                      </div>
                      <div>
                        <div className="text-sm text-gray-600">헤드 차원</div>
                        <div className="font-medium">{result.architecture_used.head_dim}</div>
                      </div>
                      <div>
                        <div className="text-sm text-gray-600">중간 크기</div>
                        <div className="font-medium">{result.architecture_used.intermediate_size}</div>
                      </div>
                      <div>
                        <div className="text-sm text-gray-600">KV 헤드 수</div>
                        <div className="font-medium">{result.architecture_used.num_key_value_heads}</div>
                      </div>
                    </div>
                    
                    {result.estimated_values && result.estimated_values.length > 0 && (
                      <div className="mt-4 border-t border-gray-200 pt-3">
                        <div className="flex items-start text-sm text-red-700">
                          <HelpCircle className="h-4 w-4 mr-1 flex-shrink-0 mt-0.5" />
                          <div>
                            <span className="font-medium">추정된 값: </span>
                            {result.estimated_values.includes("model_configuration") && "모델 구성, "}
                            {result.estimated_values.includes("parameter_count") && "파라미터 수, "}
                            {result.estimated_values.includes("context_length_from_config") && "컨텍스트 길이"}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ) : !loading && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-8 flex flex-col items-center justify-center h-full text-center">
              <Server className="h-16 w-16 text-blue-500 mb-4" />
              <h3 className="text-xl font-semibold text-gray-800 mb-2">GPU 메모리 계산하기</h3>
              <p className="text-gray-600 max-w-md">
                왼쪽 양식을 작성하고 '메모리 계산하기' 버튼을 클릭하여 vLLM에서 모델 서빙에 필요한 메모리 요구 사항을 확인하세요.
              </p>
            </div>
          )}
          
          {loading && (
            <div className="bg-white rounded-lg shadow p-8 flex flex-col items-center justify-center h-full">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mb-4"></div>
              <h3 className="text-xl font-semibold text-gray-800">계산 중...</h3>
              <p className="text-gray-600 mt-2">
                모델 구성을 분석하고 메모리 요구 사항을 계산하고 있습니다.
              </p>
            </div>
          )}
        </div>
      </div>
      
      <div className="mt-8 text-center text-sm text-gray-500">
        <p>주의! 해당 API에서 제공하는 값들은 예측값이라 정확하지 않을 수 있습니다!</p>
        <p>해당 API는 vLLM 0.7.3 기준입니다. 다른 버전의 vLLM 에서는 오차가 커질 수 있습니다!</p>
        <p className="mt-4"></p>
        <p>vLLM GPU 메모리 계산기 API v1.0</p>
        <p className="mt-1">
          <button 
            type="button"
            onClick={() => {
              const cleanArchitecture = {};
              Object.entries(formData.architecture).forEach(([key, value]) => {
                if (value !== null && value !== '') {
                  cleanArchitecture[key] = value;
                }
              });
              // 테스트용
              const sampleRequest = {
                ...formData,
                architecture: Object.keys(cleanArchitecture).length > 0 ? cleanArchitecture : null,
                tensor_parallel_size: formData.tensor_parallel_size || 1,
                current_gpu_memory_size_gb: formData.current_gpu_memory_size_gb || 24,
                max_seq_len: formData.max_seq_len || 4096,
                max_num_seqs: formData.max_num_seqs || 256
              };
              
              if (!sampleRequest.dtype) {
                delete sampleRequest.dtype;
              }
              
              if (!sampleRequest.kv_cache_dtype) {
                delete sampleRequest.kv_cache_dtype;
              }
              
              if (sampleRequest.quantization === '') {
                sampleRequest.quantization = null;
              }
              
              const jsonStr = JSON.stringify(sampleRequest, null, 2);
              
              // 테스트용 클립보드 요청 복사
              navigator.clipboard.writeText(jsonStr).then(() => {
                alert("API 요청 형식이 클립보드에 복사되었습니다.");
              });
            }}
            className="text-blue-500 hover:text-blue-700 underline"
          >
            API 요청 형식 보기
          </button>
        </p>
      </div>
    </div>
  );
};

export default VLLMMemoryCalculator;