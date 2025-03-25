import React, { useState, useEffect } from 'react';
import { AutoComplete } from 'primereact/autocomplete';

const GpuAutosuggestSearch = ({ onGpuSelect }) => {
    const [selectedGPU, setSelectedGPU] = useState(null);
    const [filteredGPUs, setFilteredGPUs] = useState([]);
    const [gpus, setGpus] = useState([]);

    useEffect(() => {
        const fetchGpus = async () => {
            try {
                const response = await fetch('http://localhost:8000/get_gpu_catalog');
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const data = await response.json();
                const gpuArray = Object.values(data);
                setGpus(gpuArray);
            } catch (error) {
                console.error('Error fetching GPU catalog:', error);
                setGpus([]); // 카탈로그 비었을 때 디폴트, 필요에 따라 수정 
            }
        };

        fetchGpus();
    }, []);

    const search = (event) => {
        const query = event.query.toLowerCase();
        const filtered = gpus.filter(gpu =>
            gpu.productName.toLowerCase().includes(query)
        );
        setFilteredGPUs(filtered);
    };

    const handleGpuSelect = (e) => {
        setSelectedGPU(e.value);
        
        // 선택 gpu로 콜백
        if (e.value && onGpuSelect) {
            onGpuSelect(e.value);
        }
    };

    const itemTemplate = (item) => {
        return (
            <div className="flex items-center py-2 px-3 hover:bg-blue-50 rounded-md transition-colors duration-150">
                <div className="flex-1">
                    <div className="text-gray-800 font-medium">{item.productName}</div>
                    <div className="text-xs text-gray-500 mt-1 flex space-x-2">
                        <span>{item.manufacturer}</span>
                        <span>•</span>
                        <span>{item.memSize} GB {item.memType}</span>
                    </div>
                </div>
            </div>
        );
    };

    return (
        <div className="w-full p-4 border border-gray-300 rounded-lg shadow-sm bg-white hover:shadow-md transition-shadow duration-300">
            <h2 className="text-lg font-semibold text-gray-800 mb-3">GPU 모델 선택</h2>
            <div className="p-fluid mb-4">
                <AutoComplete
                    field="productName"
                    value={selectedGPU}
                    suggestions={filteredGPUs}
                    completeMethod={search}
                    onChange={handleGpuSelect}
                    placeholder="GPU 이름을 검색하세요"
                    dropdown
                    className="w-full"
                    inputClassName="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
                    panelClassName="border border-gray-200 rounded-md shadow-lg bg-white mt-1 py-1 max-h-60"
                    itemClassName="border-b border-gray-100 last:border-b-0"
                    scrollHeight="250px"
                    itemTemplate={itemTemplate}
                    dropdownIcon="pi pi-chevron-down text-gray-400"
                    forceSelection={true}
                />
            </div>
            {selectedGPU && (
                <div className="mt-4 p-3 bg-gray-50 rounded-md border border-gray-200">
                    <h3 className="text-sm font-medium text-gray-500 mb-1">선택된 GPU:</h3>
                    <div className="flex flex-col space-y-2">
                        <p className="text-md font-semibold text-gray-800">{selectedGPU.productName}</p>
                        <div className="grid grid-cols-2 gap-2 text-sm">
                            <div className="flex">
                                <span className="text-gray-500">메모리:</span>
                                <span className="ml-2 font-medium">{selectedGPU.memSize} GB</span>
                            </div>
                            <div className="flex">
                                <span className="text-gray-500">제조사:</span>
                                <span className="ml-2 font-medium">{selectedGPU.manufacturer}</span>
                            </div>
                            <div className="flex">
                                <span className="text-gray-500">출시년도:</span>
                                <span className="ml-2 font-medium">{selectedGPU.releaseYear}</span>
                            </div>
                            <div className="flex">
                                <span className="text-gray-500">메모리 타입:</span>
                                <span className="ml-2 font-medium">{selectedGPU.memType}</span>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default GpuAutosuggestSearch;