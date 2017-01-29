include_directories(${CMAKE_SOURCE_DIR}/kfusion/include)

# Not recommended by CMake to use GLOB
# file(GLOB srcs *.cpp *.hpp)

# APP 01: OpenNI capture
if (OPENNI_FOUND)
    add_executable(demo demo.cpp)
    target_link_libraries(demo ${OpenCV_LIBS} kfusion)

    set_target_properties(demo PROPERTIES
        DEBUG_POSTFIX "d"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

    install(TARGETS demo RUNTIME DESTINATION bin COMPONENT main)
    install(FILES demo.cpp DESTINATION app COMPONENT main)
endif(OPENNI_FOUND)

# APP 02: BinGrabber capture
add_executable(demo_bin demo_bin.cpp)
target_link_libraries(demo_bin ${OpenCV_LIBS} kfusion)

set_target_properties(demo_bin PROPERTIES
        DEBUG_POSTFIX "d"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

install(TARGETS demo_bin RUNTIME DESTINATION bin COMPONENT main)
install(FILES demo_bin.cpp DESTINATION app COMPONENT main)
