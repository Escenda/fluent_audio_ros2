#pragma once

/**
 * @file cv_bridge_compat.hpp
 * @brief Compatibility header for cv_bridge across ROS2 distributions
 * 
 * ROS2 Humble uses cv_bridge.h, while ROS2 Jazzy uses cv_bridge.hpp
 * This header automatically selects the correct header based on ROS_DISTRO
 */

// ROS2 distribution detection
// Try to detect ROS_DISTRO from environment or preprocessor
#if !defined(ROS_DISTRO)
    // ROS_DISTRO is not defined as a preprocessor macro
    // Try to detect using __has_include (C++17 feature)
    #ifdef __has_include
        // Try .hpp first (Jazzy/Rolling), then .h (Humble)
        #if __has_include(<cv_bridge/cv_bridge.hpp>)
            #include <cv_bridge/cv_bridge.hpp>
        #elif __has_include(<cv_bridge/cv_bridge.h>)
            #include <cv_bridge/cv_bridge.h>
        #else
            #error "cv_bridge header not found. Please install ros-<distro>-cv-bridge"
        #endif
    #else
        // Fallback: try .hpp first (Jazzy/Rolling), then .h (Humble)
        // This will fail at compile time if wrong, but that's acceptable
        #include <cv_bridge/cv_bridge.hpp>
    #endif
#else
    // ROS_DISTRO is defined as a preprocessor macro
    // Note: String comparison in preprocessor requires special handling
    // We'll use a simpler approach: try .hpp first, then .h
    #ifdef __has_include
        #if __has_include(<cv_bridge/cv_bridge.hpp>)
            #include <cv_bridge/cv_bridge.hpp>
        #elif __has_include(<cv_bridge/cv_bridge.h>)
            #include <cv_bridge/cv_bridge.h>
        #else
            #error "cv_bridge header not found. Please install ros-<distro>-cv-bridge"
        #endif
    #else
        // Simple fallback: assume Jazzy/Rolling (.hpp) for newer ROS_DISTRO values
        // This is a heuristic and may need adjustment
        #include <cv_bridge/cv_bridge.hpp>
    #endif
#endif
