
include(ExternalProject)

message("${CMAKE_BUILD_TYPE}")

ExternalProject_Add(
        ext-yaml-cpp
        URL ${CMAKE_CURRENT_SOURCE_DIR}/deps/yaml-cpp.zip
        CMAKE_ARGS
        -DYAML_CPP_BUILD_TESTS=OFF
        -DYAML_CPP_BUILD_TOOLS=OFF
        -DYAML_CPP_INSTALL=OFF
        -DYAML_CPP_BUILD_CONTRIB=OFF
        -DMSVC_SHARED_RT=OFF
        -DBUILD_SHARED_LIBS=OFF
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
        -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
        -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
        -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=${CMAKE_BINARY_DIR}/ext/yaml-cpp/lib
        -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=${CMAKE_BINARY_DIR}/ext/yaml-cpp/lib
        PREFIX "${CMAKE_BINARY_DIR}/ext/yaml-cpp"
        # Disable install step
        INSTALL_COMMAND ""
        LOG_DOWNLOAD ON
        LOG_BUILD 1
)

