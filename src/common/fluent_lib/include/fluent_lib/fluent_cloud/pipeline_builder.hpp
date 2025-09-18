#pragma once

// 最小互換のパイプライン実装
// - fv_pointcloud_pipeline が参照する API を提供
// - YAML の steps を読み取り、分かる範囲の簡易フィルタを適用

#include <yaml-cpp/yaml.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "fluent_lib/fluent_cloud/filters.hpp"

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
    YAML::Node steps = root["steps"];
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

inline CloudPtr apply(const PipelineConfig &cfg, CloudPtr cloud, const FilterContext &, const PipelineOptions &opt={}) {
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
            const double zmin = st.num.count("min") ? st.num.at("min") : (st.num.count("zmin") ? st.num.at("zmin") : 0.0);
            const double zmax = st.num.count("max") ? st.num.at("max") : (st.num.count("zmax") ? st.num.at("zmax") : 10.0);
            pcl::PassThrough<pcl::PointXYZRGB> pass; pass.setInputCloud(cloud); pass.setFilterFieldName("z");
            pass.setFilterLimits(static_cast<float>(zmin), static_cast<float>(zmax));
            CloudPtr out(new Cloud); pass.filter(*out); cloud = out;
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

