#include "fa_kws/backends/factory.hpp"

#include "fa_kws/backend_config_validation.hpp"
#include "fa_kws/backends/sherpa_onnx_kws_backend.hpp"

namespace fa_kws
{

std::unique_ptr<KwsBackend> buildKwsBackend(const KwsBackendSettings &settings)
{
  validation::requireSupportedBackendName(settings.name);

  if (settings.name == validation::kBackendSherpaOnnxKws) {
    SherpaOnnxKwsBackendConfig config;
    config.target_sample_rate = settings.target_sample_rate;
    config.model_num_threads = settings.model_num_threads;
    config.execution_provider = settings.execution_provider;
    config.encoder_path = settings.encoder_path;
    config.decoder_path = settings.decoder_path;
    config.joiner_path = settings.joiner_path;
    config.tokens_path = settings.tokens_path;
    config.keywords_path = settings.keywords_path;
    config.max_active_paths = settings.max_active_paths;
    config.num_trailing_blanks = settings.num_trailing_blanks;
    config.keywords_score = settings.keywords_score;
    config.keywords_threshold = settings.keywords_threshold;
    config.vad_threshold = settings.vad_threshold;
    config.cooldown = settings.cooldown;
    config.command = settings.command;
    config.args = settings.args;
    config.health_args = settings.health_args;
    config.timeout_sec = settings.timeout_sec;
    config.workspace_dir = settings.workspace_dir;
    config.cleanup_audio_files = settings.cleanup_audio_files;

    return std::make_unique<SherpaOnnxKwsBackend>(config);
  }

  throw std::runtime_error("unsupported fa_kws backend.name: " + settings.name);
}

}  // namespace fa_kws
