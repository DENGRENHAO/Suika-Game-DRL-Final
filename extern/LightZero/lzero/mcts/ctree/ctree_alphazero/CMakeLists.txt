# Declare the minimum version of CMake that can be used
# To understand and build the project
cmake_minimum_required(VERSION 3.4...3.18)

# Set the project name to mcts_alphazero and set the version to 1.0
project(mcts_alphazero VERSION 1.0)

# Find and get the details of Python package
# This is required for embedding Python in the project
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# --- Integrate pybind11 using FetchContent ---
# This will download pybind11 if it's not already available in the build tree.
include(FetchContent)

# Define the content to fetch: pybind11 from its GitHub repository
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.12.0 # Use a specific, stable version tag (e.g., v2.12.0 or latest)
  GIT_SHALLOW    TRUE
)

# Populate the content. This will download pybind11 into the build directory
# and make its CMake targets available.
FetchContent_MakeAvailable(pybind11)
# --- End pybind11 integration ---

# Add pybind11 as a subdirectory,
# so that its build files are generated alongside the current project.
# This is necessary because the current project depends on pybind11
add_subdirectory(pybind11)

# Add two .cpp files to the mcts_alphazero module
# These files are compiled and linked into the module
pybind11_add_module(mcts_alphazero mcts_alphazero.cpp node_alphazero.cpp)

# Add the Python header file paths to the include paths
# of the mcts_alphazero library. This is necessary for the
# project to find the Python header files it needs to include
target_include_directories(mcts_alphazero PRIVATE ${Python3_INCLUDE_DIRS})

# Link the mcts_alphazero library with the pybind11::module target.
# This is necessary for the mcts_alphazero library to use the functions and classes defined by pybind11
target_link_libraries(mcts_alphazero PRIVATE pybind11::module)

# Set the Python standard to the version of Python found by find_package(Python3)
# This ensures that the code will be compiled against the correct version of Python
set_target_properties(mcts_alphazero PROPERTIES PYTHON_STANDARD ${Python3_VERSION})