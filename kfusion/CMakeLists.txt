set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true;--prec-div=false;--prec-sqrt=false")

#add_module_library(kfusion)
set(module_name kfusion)
set(Includes_srcs
    include/io/bin_grabber.hpp
    include/${module_name}/exports.hpp
    include/${module_name}/kinfu.hpp
    include/${module_name}/types.hpp)

set(Includes_cuda_srcs
    include/${module_name}/cuda/color_volume.hpp
    include/${module_name}/cuda/device_array.hpp
    include/${module_name}/cuda/device_memory.hpp
    include/${module_name}/cuda/imgproc.hpp
    include/${module_name}/cuda/kernel_containers.hpp
    include/${module_name}/cuda/projective_icp.hpp
    include/${module_name}/cuda/tsdf_volume.hpp)

set(Source_srcs
    src/bin_grabber.cpp
    src/color_volume.cpp
    src/core.cpp
    src/device_memory.cpp
    src/imgproc.cpp
    src/internal.hpp
    src/kinfu.cpp
    src/precomp.cpp
    src/precomp.hpp
    src/projective_icp.cpp
    src/safe_call.hpp
    src/tsdf_volume.cpp)

set(Source_cuda_srcs
    src/cuda/color_volume.cu
    src/cuda/device.hpp
    src/cuda/imgproc.cu
    src/cuda/proj_icp.cu
    src/cuda/temp_utils.hpp
    src/cuda/texture_binder.hpp
    src/cuda/tsdf_volume.cu)

if(OPENNI_FOUND)
    set(Source_srcs ${Source_srcs}
        src/capture.cpp)
    set(Include_srcs ${Include_srcs}
        include/io/capture.hpp)
endif(OPENNI_FOUND)

include_directories(include include/kfusion/cuda/ src src/cuda)

set(__has_cuda OFF)
check_cuda(__has_cuda)

set(__lib_type STATIC)
if (${ARGV1} MATCHES "SHARED|STATIC")
    set(__lib_type ${ARGV1})
endif()

if (__has_cuda)
    cuda_add_library(${module_name} ${__lib_type} ${Source_srcs} ${Source_cuda_srcs} )
else()
    add_library(${module_name} ${__lib_type} ${Source_srcs})
endif()

if(MSVC)
    set_target_properties(${module_name} PROPERTIES DEFINE_SYMBOL KFUSION_API_EXPORTS)
else()
    add_definitions(-DKFUSION_API_EXPORTS)
endif()

default_properties(${module_name})

if(USE_PROJECT_FOLDERS)
    set_target_properties(${module_name} PROPERTIES FOLDER "Libraries")
endif()

set_target_properties(${module_name} PROPERTIES INSTALL_NAME_DIR lib)

install(TARGETS ${module_name}
        RUNTIME DESTINATION bin COMPONENT main
        LIBRARY DESTINATION lib COMPONENT main
        ARCHIVE DESTINATION lib COMPONENT main)

install(DIRECTORY include/ DESTINATION include/ FILES_MATCHING PATTERN "*.h*")

target_link_libraries(kfusion ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${OpenCV_LIBS})

MESSAGE("OpenNI FOUND = ${OPENNI_FOUND}")

if (OPENNI_FOUND)
    target_link_libraries(kfusion ${OPENNI_LIBRARY})
endif(OPENNI_FOUND)