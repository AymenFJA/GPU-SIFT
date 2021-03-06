# Install script for directory: /home/aymen/SummerRadical/SIFT-GPU

# Set the install prefix
IF(NOT DEFINED CMAKE_INSTALL_PREFIX)
  SET(CMAKE_INSTALL_PREFIX "/usr/local")
ENDIF(NOT DEFINED CMAKE_INSTALL_PREFIX)
STRING(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
IF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  IF(BUILD_TYPE)
    STRING(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  ELSE(BUILD_TYPE)
    SET(CMAKE_INSTALL_CONFIG_NAME "")
  ENDIF(BUILD_TYPE)
  MESSAGE(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
ENDIF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)

# Set the component getting installed.
IF(NOT CMAKE_INSTALL_COMPONENT)
  IF(COMPONENT)
    MESSAGE(STATUS "Install component: \"${COMPONENT}\"")
    SET(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  ELSE(COMPONENT)
    SET(CMAKE_INSTALL_COMPONENT)
  ENDIF(COMPONENT)
ENDIF(NOT CMAKE_INSTALL_COMPONENT)

# Install shared libraries without execute permission?
IF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  SET(CMAKE_INSTALL_SO_NO_EXE "0")
ENDIF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/." TYPE FILE FILES
    "/home/aymen/SummerRadical/SIFT-GPU/cudaImage.cu"
    "/home/aymen/SummerRadical/SIFT-GPU/cudaImage.h"
    "/home/aymen/SummerRadical/SIFT-GPU/cudaSiftH.cu"
    "/home/aymen/SummerRadical/SIFT-GPU/cudaSiftH.h"
    "/home/aymen/SummerRadical/SIFT-GPU/matching.cu"
    "/home/aymen/SummerRadical/SIFT-GPU/cudaSiftD.h"
    "/home/aymen/SummerRadical/SIFT-GPU/cudaSift.h"
    "/home/aymen/SummerRadical/SIFT-GPU/cudautils.h"
    "/home/aymen/SummerRadical/SIFT-GPU/geomFuncs.cpp"
    "/home/aymen/SummerRadical/SIFT-GPU/mainSift.cpp"
    "/home/aymen/SummerRadical/SIFT-GPU/cudaSiftD.cu"
    "/home/aymen/SummerRadical/SIFT-GPU/CMakeLists.txt"
    "/home/aymen/SummerRadical/SIFT-GPU/Copyright.txt"
    )
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/data" TYPE FILE FILES
    "/home/aymen/SummerRadical/SIFT-GPU/data/left.pgm"
    "/home/aymen/SummerRadical/SIFT-GPU/data/righ.pgm"
    )
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

IF(CMAKE_INSTALL_COMPONENT)
  SET(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
ELSE(CMAKE_INSTALL_COMPONENT)
  SET(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
ENDIF(CMAKE_INSTALL_COMPONENT)

FILE(WRITE "/home/aymen/SummerRadical/SIFT-GPU/${CMAKE_INSTALL_MANIFEST}" "")
FOREACH(file ${CMAKE_INSTALL_MANIFEST_FILES})
  FILE(APPEND "/home/aymen/SummerRadical/SIFT-GPU/${CMAKE_INSTALL_MANIFEST}" "${file}\n")
ENDFOREACH(file)
