# Cmake needs this line.
cmake_minimum_required(VERSION 3.0)

# Define project name.
PROJECT (Peptide__Network_CPP)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "    OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Set the Source file that meant to be executed.
set( NAME_SRC
    src/main.cpp
)

# Set Headers.
set( NAME_HEADERS
     include/Net.h
     include/Neuron.h
     include/Net.cpp
     include/Neuron.cpp
     include/Connection.h
)

# Build the executable file.
add_executable( Peptide__Network_CPP ${NAME_SRC} ${NAME_HEADERS} )

# Show Warnings.
add_compile_options(-Wall)


# Link your application with OpenCV libraries
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )
