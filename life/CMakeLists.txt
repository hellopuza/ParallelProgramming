cmake_minimum_required(VERSION 3.9)

set(EXEC_NAME life)
project(${EXEC_NAME})

file(GLOB_RECURSE SOURCES *.c*)
file(GLOB_RECURSE HEADERS *.h*)

add_executable(${EXEC_NAME} ${SOURCES})
target_include_directories(${EXEC_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/)

file(COPY ${CMAKE_CURRENT_BINARY_DIR}/${EXEC_NAME} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(${EXEC_NAME} PROPERTIES
    CXX_STANDARD          20
    CXX_STANDARD_REQUIRED ON
)

find_package(MPI REQUIRED)
target_include_directories(${EXEC_NAME} PRIVATE ${MPI_C_INCLUDE_PATH}) 
target_link_libraries(${EXEC_NAME} PRIVATE -lmpi_cxx -lmpi)

find_package(SFML 2.5.1 REQUIRED COMPONENTS graphics window system audio)
target_include_directories(${EXEC_NAME} PRIVATE ${SFML_INCLUDE_DIR})
target_link_libraries(${EXEC_NAME} PRIVATE sfml-graphics sfml-window sfml-system sfml-audio)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3"
)