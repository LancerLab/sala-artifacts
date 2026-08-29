include(FetchContent)

set(GOOGLETEST_DIR "" CACHE STRING "Location of local GoogleTest repo to build against")

# SALA artifact: the googletest release-1.12.1 tarball is bundled in
# third_party/ (github.com is the fork build's only remaining flaky
# download besides json, which is bundled too). When present, extract
# and use it — the git clone is skipped entirely.
if(NOT GOOGLETEST_DIR)
  set(_gt_bundle "${CMAKE_CURRENT_LIST_DIR}/../../third_party/googletest-release-1.12.1.tar.gz")
  if(EXISTS "${_gt_bundle}")
    set(GOOGLETEST_DIR "${CMAKE_BINARY_DIR}/_deps/googletest-src/googletest-release-1.12.1")
    if(NOT EXISTS "${GOOGLETEST_DIR}/CMakeLists.txt")
      file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/_deps/googletest-src")
      execute_process(COMMAND ${CMAKE_COMMAND} -E tar xf "${_gt_bundle}"
                      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/_deps/googletest-src")
    endif()
  endif()
endif()

# FORCE: on a re-configure (pip retries reuse the in-tree build dir) the
# cache value from the previous attempt must be re-applied, otherwise
# FetchContent falls back to the git clone.
set(FETCHCONTENT_SOURCE_DIR_GOOGLETEST ${GOOGLETEST_DIR} CACHE STRING "GoogleTest source directory override" FORCE)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
  )

FetchContent_GetProperties(googletest)

if(NOT googletest_POPULATED)
  FetchContent_Populate(googletest)
  if (MSVC)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  endif()
  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()
