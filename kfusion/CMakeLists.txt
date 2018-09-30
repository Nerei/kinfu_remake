list(APPEND CUDA_NVCC_FLAGS "--ftz=true;--prec-div=false;--prec-sqrt=false")
add_module_library(kfusion)
target_link_libraries(kfusion ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${OpenCV_LIBS} ${OPENNI_LIBRARY})
