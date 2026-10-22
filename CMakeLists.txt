# Napa. Copyright (c) 2026 Zachary Wilson. PolyForm Noncommercial License 1.0.0: see LICENSE.
# Napa's detector core as a plain C library, for C and C++ programs. Python users: pip install napa-vision.
#
#   cmake -S . -B build && cmake --build build && cmake --install build --prefix /usr/local
#
# Then, in your project: find_package(napa) and target_link_libraries(app PRIVATE napa::napa).
cmake_minimum_required(VERSION 3.15)
file(STRINGS src/napa_vision/_version.py NAPA_VERSION_LINE REGEX "^__version__")
string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" NAPA_VERSION "${NAPA_VERSION_LINE}")
project(napa VERSION ${NAPA_VERSION} LANGUAGES C)

option(BUILD_SHARED_LIBS "Build a shared library" OFF)
option(NAPA_BUILD_EXAMPLES "Build the example program" OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(NAPA_NATIVE ${CMAKE_CURRENT_SOURCE_DIR}/src/napa_vision/_native)
add_library(napa ${NAPA_NATIVE}/napa_core.c)
add_library(napa::napa ALIAS napa)
target_include_directories(napa PUBLIC $<BUILD_INTERFACE:${NAPA_NATIVE}> $<INSTALL_INTERFACE:include>)
set_target_properties(napa PROPERTIES
  C_STANDARD 99
  C_STANDARD_REQUIRED ON
  C_VISIBILITY_PRESET hidden
  POSITION_INDEPENDENT_CODE ON
  PUBLIC_HEADER ${NAPA_NATIVE}/napa_core.h)
if(MSVC)
  target_compile_options(napa PRIVATE $<$<CONFIG:Release>:/O2>)
else()
  # Signed overflow isn't relied on; -fno-wrapv keeps compilers free to optimise the inner loops.
  target_compile_options(napa PRIVATE $<$<CONFIG:Release>:-O3> -fno-wrapv)
  target_link_libraries(napa PRIVATE m)
endif()

include(GNUInstallDirs)
install(TARGETS napa EXPORT napaTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES LICENSE THIRD_PARTY_NOTICES DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/napa)
install(EXPORT napaTargets NAMESPACE napa:: FILE napaConfig.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/napa)

if(NAPA_BUILD_EXAMPLES)
  add_executable(napa_detect_pgm examples/c/detect_pgm.c)
  target_link_libraries(napa_detect_pgm PRIVATE napa)
endif()
