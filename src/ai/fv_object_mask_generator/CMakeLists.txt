# =============================================================================
# Fluent Vision - オブジェクトマスク生成ノード CMakeLists.txt
# =============================================================================
# 
# 概要：
#   - UNetを使用したセグメンテーションマスク生成ノードのビルド設定
#   - OpenVINOによる高速推論対応
#   - ROS2 Humble対応のCMake設定
#   - 依存関係の管理とインストール設定
#
# 作者：Takashi Otsuka
# 作成日：2024年
# バージョン：1.0
# =============================================================================

cmake_minimum_required(VERSION 3.8)
project(fv_object_mask_generator)

# ===== コンパイラ警告設定 =====
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)  # 厳格な警告設定
endif()

# ===== 依存関係パッケージ検索 =====
find_package(ament_cmake REQUIRED)           # ROS2ビルドシステム
find_package(rclcpp REQUIRED)                # ROS2 C++クライアントライブラリ
find_package(rclcpp_components REQUIRED)     # ROS2コンポーネント
find_package(sensor_msgs REQUIRED)           # センサーメッセージ型
find_package(std_msgs REQUIRED)              # 標準メッセージ型
find_package(cv_bridge REQUIRED)             # OpenCV-ROS2橋渡し
find_package(tf2 REQUIRED)                   # TF2座標変換ライブラリ
find_package(tf2_ros REQUIRED)               # TF2 ROS2統合
find_package(vision_msgs REQUIRED)           # コンピュータビジョンメッセージ
find_package(OpenCV REQUIRED)                # OpenCVコンピュータビジョン
find_package(fluent_lib REQUIRED)            # Fluent Vision共通ライブラリ
find_package(OpenVINO QUIET)                 # Intel OpenVINO推論エンジン（オプション）
find_package(PkgConfig REQUIRED)             # pkg-config

# OpenVINOサポートの有効/無効を設定
option(ENABLE_OPENVINO "Enable OpenVINO support" ${OpenVINO_FOUND})
if(ENABLE_OPENVINO AND NOT OpenVINO_FOUND)
    message(WARNING "OpenVINO not found. Disabling OpenVINO support.")
    set(ENABLE_OPENVINO OFF)
endif()
pkg_check_modules(JSON QUIET nlohmann_json)  # JSON処理ライブラリ（オプション）
find_package(TIFF QUIET)
find_package(CURL QUIET)
find_package(GDAL QUIET)

# ===== インクルードディレクトリ設定 =====
include_directories(
  include                    # ヘッダーファイルディレクトリ
  ${OpenCV_INCLUDE_DIRS}     # OpenCVインクルードディレクトリ
)
if(JSON_FOUND)
  include_directories(${JSON_INCLUDE_DIRS})  # JSONライブラリインクルードディレクトリ
endif()

# ===== 実行ファイル作成 =====
add_executable(fv_object_mask_generator_node
  src/fv_object_mask_generator_node.cpp  # メインノードソース
)

# ===== ROS2依存関係リンク =====
ament_target_dependencies(fv_object_mask_generator_node
  rclcpp                    # ROS2 C++クライアントライブラリ
  rclcpp_components         # ROS2コンポーネント
  sensor_msgs               # センサーメッセージ型
  std_msgs                  # 標準メッセージ型
  cv_bridge                 # OpenCV-ROS2橋渡し
  tf2                       # TF2座標変換ライブラリ
  tf2_ros                   # TF2 ROS2統合
  vision_msgs               # コンピュータビジョンメッセージ
  fluent_lib                # Fluent Vision共通ライブラリ
)

# ===== 外部ライブラリリンク =====
target_link_libraries(fv_object_mask_generator_node
  ${OpenCV_LIBRARIES}       # OpenCVライブラリ
)

# JSONライブラリ（オプション）
if(JSON_FOUND)
  target_link_libraries(fv_object_mask_generator_node ${JSON_LIBRARIES})
  target_compile_definitions(fv_object_mask_generator_node PRIVATE ENABLE_JSON=1)
else()
  target_compile_definitions(fv_object_mask_generator_node PRIVATE ENABLE_JSON=0)
  message(STATUS "nlohmann_json not found. Building without JSON support.")
endif()

# TIFFライブラリ（オプション）
if(TIFF_FOUND)
  target_link_libraries(fv_object_mask_generator_node TIFF::TIFF)
endif()

# CURLライブラリ（オプション）
if(CURL_FOUND)
  target_link_libraries(fv_object_mask_generator_node CURL::libcurl)
endif()

# GDALライブラリ（オプション）
if(GDAL_FOUND)
  target_link_libraries(fv_object_mask_generator_node GDAL::GDAL)
endif()

# OpenVINOサポートが有効な場合のみリンク
if(ENABLE_OPENVINO)
  target_link_libraries(fv_object_mask_generator_node
    openvino::runtime         # OpenVINO推論エンジン
  )
  target_compile_definitions(fv_object_mask_generator_node
    PRIVATE ENABLE_OPENVINO=1
  )
else()
  target_compile_definitions(fv_object_mask_generator_node
    PRIVATE ENABLE_OPENVINO=0
  )
  message(STATUS "OpenVINO support disabled. Building without OpenVINO.")
endif()

# ===== インストール設定 =====
# 実行ファイルのインストール
install(TARGETS
  fv_object_mask_generator_node
  DESTINATION lib/${PROJECT_NAME}  # インストール先：lib/fv_object_mask_generator/
)

# ディレクトリのインストール
install(DIRECTORY
  launch                    # launchディレクトリ
  DESTINATION share/${PROJECT_NAME}  # インストール先：share/fv_object_mask_generator/
)

# 設定ファイルのインストール
install(DIRECTORY
  config                    # configディレクトリ
  DESTINATION share/${PROJECT_NAME}  # インストール先：share/fv_object_mask_generator/
)

# ヘッダーファイルのインストール
install(DIRECTORY
  include/                  # ヘッダーファイルディレクトリ
  DESTINATION include       # インストール先：include/
)

# ===== テスト設定 =====
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)                    # 自動リントツール
  ament_lint_auto_find_test_dependencies()                  # テスト依存関係検索
endif()

# ===== パッケージ設定 =====
ament_package()  # ROS2パッケージ設定の完了
