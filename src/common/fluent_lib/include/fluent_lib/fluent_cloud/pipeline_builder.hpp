#pragma once

// 最小互換のパイプライン実装
// - fv_pointcloud_pipeline が参照する API を提供
// - YAML の steps / filters を読み取り、簡易フィルタを適用

#include <yaml-cpp/yaml.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <Eigen/Core>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>

#include "fluent_lib/fluent_cloud/filters.hpp"

#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>

namespace fluent_cloud::pipeline {

using Cloud = pcl::PointCloud<pcl::PointXYZRGB>;
using CloudPtr = Cloud::Ptr;

struct StepConfig {
    std::string name;
    std::string type;           // "voxel_grid" / "sor" / "z_range" など
    std::string debug_topic;    // 任意
    std::unordered_map<std::string, double> num;     // 数値パラメータ
    std::unordered_map<std::string, std::string> str; // 文字列パラメータ
};

struct PipelineConfig {
    std::vector<StepConfig> steps;
};

struct FilterContext {
    std::unordered_map<std::string, double> scalars; // 例: root_depth, roi_width_px
};

struct PipelineOptions {
    std::function<void(std::size_t, const std::string&, const CloudPtr&)> on_step; // デバッグ出力用
};

inline PipelineConfig loadPipelineConfig(const YAML::Node &root) {
    PipelineConfig cfg;
    if (!root || !root.IsMap()) return cfg;
    // steps / filters の両対応
    YAML::Node steps = root["steps"];
    if (!steps || !steps.IsSequence()) {
        steps = root["filters"]; // alias
    }
    if (!steps || !steps.IsSequence()) return cfg;
    for (const auto &s : steps) {
        StepConfig sc;
        if (s["name"]) sc.name = s["name"].as<std::string>("");
        if (s["type"]) sc.type = s["type"].as<std::string>("");
        if (s["debug_topic"]) sc.debug_topic = s["debug_topic"].as<std::string>("");
        // 1) 互換のため params: ブロックを読み取り
        if (s["params"] && s["params"].IsMap()) {
            for (auto it = s["params"].begin(); it != s["params"].end(); ++it) {
                const std::string key = it->first.as<std::string>("");
                const auto &val = it->second;
                if (val.IsScalar()) {
                    try { sc.num[key] = val.as<double>(); }
                    catch (...) { sc.str[key] = val.as<std::string>(""); }
                } else if (val.IsSequence() && key == "axis") {
                    // axis: [x,y,z]
                    if (val.size() >= 3) {
                        try {
                            sc.num["axis_x"] = val[0].as<double>();
                            sc.num["axis_y"] = val[1].as<double>();
                            sc.num["axis_z"] = val[2].as<double>();
                        } catch (...) {}
                    }
                }
            }
        }
        // 2) トップレベルに直接書かれたキーも読み取り（後方互換）
        if (s.IsMap()) {
            for (auto it = s.begin(); it != s.end(); ++it) {
                const std::string key = it->first.as<std::string>("");
                if (key == "name" || key == "type" || key == "debug_topic" || key == "params") continue;
                const auto &val = it->second;
                if (val.IsScalar()) {
                    // 数値 or 文字列
                    try {
                        sc.num[key] = val.as<double>();
                    } catch (...) {
                        sc.str[key] = val.as<std::string>("");
                    }
                } else if (val.IsSequence() && key == "axis") {
                    if (val.size() >= 3) {
                        try {
                            sc.num["axis_x"] = val[0].as<double>();
                            sc.num["axis_y"] = val[1].as<double>();
                            sc.num["axis_z"] = val[2].as<double>();
                        } catch (...) {}
                    }
                }
            }
        }
        cfg.steps.emplace_back(std::move(sc));
    }
    return cfg;
}

// 文字列パラメータに $var ± number を許容し、簡易評価する
inline double evalParam(const std::unordered_map<std::string, double> &num,
                        const std::unordered_map<std::string, std::string> &str,
                        const std::string &key,
                        const FilterContext &ctx,
                        double fallback)
{
    auto n = num.find(key);
    if (n != num.end()) return n->second;
    auto it = str.find(key);
    if (it == str.end()) return fallback;
    const std::string &expr = it->second;
    // 例: "$root_depth - 0.08" / "$root_depth+0.08" / "0.25"
    // 数値そのもの
    try { return std::stod(expr); } catch (...) {}
    // $var [+-] const
    double sign = +1.0; double c = 0.0;
    size_t pos_plus = expr.find('+');
    size_t pos_minus = expr.find('-');
    size_t pos = std::string::npos;
    if (pos_plus != std::string::npos) { pos = pos_plus; sign = +1.0; }
    else if (pos_minus != std::string::npos && pos_minus > 0) { pos = pos_minus; sign = -1.0; }
    std::string lhs = expr;
    if (pos != std::string::npos) {
        lhs = expr.substr(0, pos);
        try { c = std::stod(expr.substr(pos+1)); } catch (...) { c = 0.0; }
    }
    // lhs は $var 形式を想定
    std::string var = lhs;
    // トリム
    auto trim = [](std::string s){
        size_t b=s.find_first_not_of(" \t"); size_t e=s.find_last_not_of(" \t");
        if (b==std::string::npos) return std::string();
        return s.substr(b, e-b+1);
    };
    var = trim(var);
    if (!var.empty() && var[0] == '$') var = var.substr(1);
    auto v = ctx.scalars.find(var);
    double base = (v != ctx.scalars.end()) ? v->second : fallback;
    return base + sign * c;
}

inline CloudPtr apply(const PipelineConfig &cfg, CloudPtr cloud, const FilterContext &ctx, const PipelineOptions &opt={}) {
    if (!cloud) cloud.reset(new Cloud);
    for (std::size_t i=0; i<cfg.steps.size(); ++i) {
        const auto &st = cfg.steps[i];
        if (st.type == "voxel_grid") {
            const double leaf = st.num.count("leaf") ? st.num.at("leaf") : (st.num.count("leaf_size") ? st.num.at("leaf_size") : 0.005);
            cloud = fluent_cloud::filters::VoxelGrid<pcl::PointXYZRGB>().setLeafSize(leaf).filter(cloud);
        } else if (st.type == "sor" || st.type == "statistical_outlier_removal" || st.type == "statistical_outlier") {
            const int mean_k = st.num.count("mean_k") ? static_cast<int>(st.num.at("mean_k")) : 50;
            const double stddev = st.num.count("stddev_mul") ? st.num.at("stddev_mul") : 1.0;
            cloud = fluent_cloud::filters::StatisticalOutlierRemoval<pcl::PointXYZRGB>()
                        .setMeanK(mean_k).setStddevMulThresh(stddev).filter(cloud);
        } else if (st.type == "z_range" || st.type == "pass_through") {
            const double vmin = evalParam(st.num, st.str, "min", ctx, st.num.count("zmin") ? st.num.at("zmin") : 0.0);
            const double vmax = evalParam(st.num, st.str, "max", ctx, st.num.count("zmax") ? st.num.at("zmax") : 10.0);
            std::string field = "z";
            if (st.str.count("field")) field = st.str.at("field");
            // 安全性: 未知フィールドは z にフォールバック
            if (!(field == "x" || field == "y" || field == "z")) field = "z";
            pcl::PassThrough<pcl::PointXYZRGB> pass; pass.setInputCloud(cloud); pass.setFilterFieldName(field);
            pass.setFilterLimits(static_cast<float>(vmin), static_cast<float>(vmax));
            CloudPtr out(new Cloud); pass.filter(*out); cloud = out;
        } else if (st.type == "ground_remove") {
            // RANSAC平面 + 法線角度/深度/インライヤ基準で地面判定し除去
            const double dist = st.num.count("distance_threshold") ? st.num.at("distance_threshold") : 0.015;
            const int max_iter = st.num.count("max_iterations") ? static_cast<int>(st.num.at("max_iterations")) : 100;
            const double axis_x = st.num.count("axis_x") ? st.num.at("axis_x") : 0.0;
            const double axis_y = st.num.count("axis_y") ? st.num.at("axis_y") : 0.0;
            const double axis_z = st.num.count("axis_z") ? st.num.at("axis_z") : 1.0;
            const double ang_deg = st.num.count("angular_threshold_deg") ? st.num.at("angular_threshold_deg") : 25.0;
            const double min_inliers_ratio = st.num.count("min_inliers_ratio") ? st.num.at("min_inliers_ratio") : 0.0;
            const int min_inliers = st.num.count("min_inliers") ? static_cast<int>(st.num.at("min_inliers")) : 0;
            const double min_remaining_ratio = st.num.count("min_remaining_ratio") ? st.num.at("min_remaining_ratio") : 0.0;
            const int min_remaining_points = st.num.count("min_remaining_points") ? static_cast<int>(st.num.at("min_remaining_points")) : 0;
            const bool keep_largest_cluster = st.str.count("keep_largest_cluster") ? (st.str.at("keep_largest_cluster") == "true" || st.str.at("keep_largest_cluster") == "1")
                                                                                    : (st.num.count("keep_largest_cluster") ? (st.num.at("keep_largest_cluster") != 0.0) : false);
            const double min_plane_depth = evalParam(st.num, st.str, "min_plane_depth", ctx, -1.0);

            pcl::SACSegmentation<pcl::PointXYZRGB> seg; seg.setOptimizeCoefficients(true);
            seg.setModelType(pcl::SACMODEL_PLANE); seg.setMethodType(pcl::SAC_RANSAC);
            seg.setMaxIterations(max_iter); seg.setDistanceThreshold(static_cast<float>(dist));
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
            pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients);
            seg.setInputCloud(cloud);
            seg.segment(*inliers, *coeff);

            bool remove_ok = false;
            if (!inliers->indices.empty() && coeff && coeff->values.size() >= 4) {
                // 法線角度チェック
                Eigen::Vector3d normal(coeff->values[0], coeff->values[1], coeff->values[2]);
                double nn = normal.norm();
                if (nn > 1e-9) normal /= nn;
                Eigen::Vector3d axis(axis_x, axis_y, axis_z);
                double aa = axis.norm();
                if (aa > 1e-9) axis /= aa; else axis = Eigen::Vector3d(0,0,1);
                double cosang = std::abs(normal.dot(axis)); // ±方向どちらでも可
                double cos_thr = std::cos(ang_deg * M_PI / 180.0);

                // インライヤ条件
                const int npts = static_cast<int>(cloud->points.size());
                const int ninl = static_cast<int>(inliers->indices.size());
                const double ratio = npts > 0 ? static_cast<double>(ninl) / static_cast<double>(npts) : 0.0;

                // 平面の平均Z（カメラ前方の深度）
                double mean_z = 0.0; int cnt_z = 0;
                for (int idx : inliers->indices) {
                    if (idx >= 0 && idx < static_cast<int>(cloud->points.size())) {
                        const auto &p = cloud->points[idx];
                        if (std::isfinite(p.z) && p.z > 0.0f) { mean_z += p.z; ++cnt_z; }
                    }
                }
                if (cnt_z > 0) mean_z /= static_cast<double>(cnt_z); else mean_z = 0.0;

                bool angle_ok = (cosang >= cos_thr);
                bool inliers_ok = (ninl >= min_inliers) && (ratio >= min_inliers_ratio);
                bool depth_ok = (min_plane_depth < 0.0) || (mean_z >= min_plane_depth);
                remove_ok = angle_ok && inliers_ok && depth_ok;
            }

            if (remove_ok) {
                pcl::ExtractIndices<pcl::PointXYZRGB> ex; ex.setInputCloud(cloud); ex.setIndices(inliers); ex.setNegative(true);
                CloudPtr out(new Cloud); ex.filter(*out);
                // セーフガード: 残量が小さすぎる場合は除去無効
                const int n_before = static_cast<int>(cloud->points.size());
                const int n_after = static_cast<int>(out->points.size());
                const double rem_ratio = (n_before > 0) ? static_cast<double>(n_after) / static_cast<double>(n_before) : 0.0;
                bool remain_ok = (n_after >= min_remaining_points) && (rem_ratio >= min_remaining_ratio);
                if (remain_ok) {
                    // オプション: 残った点群から最大クラスタのみ保持
                    if (keep_largest_cluster && !out->empty()) {
                        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
                        tree->setInputCloud(out);
                        std::vector<pcl::PointIndices> clusters;
                        pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
                        ec.setClusterTolerance(0.02f);
                        ec.setMinClusterSize(10);
                        ec.setMaxClusterSize(250000);
                        ec.setSearchMethod(tree);
                        ec.setInputCloud(out);
                        ec.extract(clusters);
                        if (!clusters.empty()) {
                            auto largest = std::max_element(clusters.begin(), clusters.end(),
                                [](const auto &a, const auto &b){ return a.indices.size() < b.indices.size(); });
                            pcl::ExtractIndices<pcl::PointXYZRGB> ex2; ex2.setInputCloud(out);
                            pcl::PointIndices::Ptr idx2(new pcl::PointIndices(*largest)); ex2.setIndices(idx2); ex2.setNegative(false);
                            CloudPtr out2(new Cloud); ex2.filter(*out2); out = out2;
                        }
                    }
                    cloud = out;
                }
            }
        } else if (st.type == "cluster_keep_largest") {
            const double tol = st.num.count("tolerance") ? st.num.at("tolerance") : 0.02;
            const int min_sz = st.num.count("min_size") ? static_cast<int>(st.num.at("min_size")) : 30;
            const int max_sz = st.num.count("max_size") ? static_cast<int>(st.num.at("max_size")) : 250000;
            if (!cloud->empty()) {
                pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
                tree->setInputCloud(cloud);
                std::vector<pcl::PointIndices> clusters;
                pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
                ec.setClusterTolerance(static_cast<float>(tol));
                ec.setMinClusterSize(min_sz);
                ec.setMaxClusterSize(max_sz);
                ec.setSearchMethod(tree);
                ec.setInputCloud(cloud);
                ec.extract(clusters);
                if (!clusters.empty()) {
                    auto largest = std::max_element(clusters.begin(), clusters.end(),
                        [](const auto &a, const auto &b){ return a.indices.size() < b.indices.size(); });
                    pcl::ExtractIndices<pcl::PointXYZRGB> ex; ex.setInputCloud(cloud);
                    pcl::PointIndices::Ptr idx(new pcl::PointIndices(*largest)); ex.setIndices(idx); ex.setNegative(false);
                    CloudPtr out(new Cloud); ex.filter(*out); cloud = out;
                }
            }
        } else {
            // 未知ステップはスキップ
        }

        if (opt.on_step) {
            opt.on_step(i, st.debug_topic, cloud);
        }
    }
    return cloud;
}

} // namespace fluent_cloud::pipeline
