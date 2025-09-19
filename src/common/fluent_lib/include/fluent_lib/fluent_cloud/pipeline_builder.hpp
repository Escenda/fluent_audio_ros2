#pragma once

// 最小互換のパイプライン実装
// - fv_pointcloud_pipeline が参照する API を提供
// - YAML の steps / filters を読み取り、簡易フィルタを適用

#include <yaml-cpp/yaml.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

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
        if (s["params"] && s["params"].IsMap()) {
            for (auto it = s["params"].begin(); it != s["params"].end(); ++it) {
                const std::string key = it->first.as<std::string>("");
                const auto &val = it->second;
                if (val.IsScalar()) {
                    try { sc.num[key] = val.as<double>(); }
                    catch (...) { sc.str[key] = val.as<std::string>(""); }
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
        } else if (st.type == "sor" || st.type == "statistical_outlier_removal") {
            const int mean_k = st.num.count("mean_k") ? static_cast<int>(st.num.at("mean_k")) : 50;
            const double stddev = st.num.count("stddev_mul") ? st.num.at("stddev_mul") : 1.0;
            cloud = fluent_cloud::filters::StatisticalOutlierRemoval<pcl::PointXYZRGB>()
                        .setMeanK(mean_k).setStddevMulThresh(stddev).filter(cloud);
        } else if (st.type == "z_range" || st.type == "pass_through") {
            const double zmin = evalParam(st.num, st.str, "min", ctx, st.num.count("zmin") ? st.num.at("zmin") : 0.0);
            const double zmax = evalParam(st.num, st.str, "max", ctx, st.num.count("zmax") ? st.num.at("zmax") : 10.0);
            pcl::PassThrough<pcl::PointXYZRGB> pass; pass.setInputCloud(cloud); pass.setFilterFieldName("z");
            pass.setFilterLimits(static_cast<float>(zmin), static_cast<float>(zmax));
            CloudPtr out(new Cloud); pass.filter(*out); cloud = out;
        } else if (st.type == "ground_remove") {
            // 単純なRANSAC平面除去 + 法線制約（近似）
            const double dist = st.num.count("distance_threshold") ? st.num.at("distance_threshold") : 0.015;
            const int max_iter = st.num.count("max_iterations") ? static_cast<int>(st.num.at("max_iterations")) : 100;
            pcl::SACSegmentation<pcl::PointXYZRGB> seg; seg.setOptimizeCoefficients(true);
            seg.setModelType(pcl::SACMODEL_PLANE); seg.setMethodType(pcl::SAC_RANSAC);
            seg.setMaxIterations(max_iter); seg.setDistanceThreshold(static_cast<float>(dist));
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
            pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients);
            seg.setInputCloud(cloud);
            seg.segment(*inliers, *coeff);
            if (!inliers->indices.empty()) {
                pcl::ExtractIndices<pcl::PointXYZRGB> ex; ex.setInputCloud(cloud); ex.setIndices(inliers); ex.setNegative(true);
                CloudPtr out(new Cloud); ex.filter(*out); cloud = out;
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
