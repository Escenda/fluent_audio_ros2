#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "fa_kws/backends/sherpa_onnx_kws_backend.hpp"

namespace {

std::vector<float> resample_linear(const std::vector<float> &samples,
                                   std::int32_t src_rate,
                                   std::int32_t dst_rate)
{
  if (samples.empty() || src_rate <= 0 || dst_rate <= 0 || src_rate == dst_rate) {
    if (src_rate <= 0 || dst_rate <= 0) {
      throw std::invalid_argument("sample rates must be positive");
    }
    return samples;
  }
  const double ratio = static_cast<double>(dst_rate) / static_cast<double>(src_rate);
  const std::size_t dst_len =
    std::max<std::size_t>(1, static_cast<std::size_t>(std::floor(samples.size() * ratio)));

  std::vector<float> out(dst_len);
  const double inv_ratio = 1.0 / ratio;
  for (std::size_t i = 0; i < dst_len; ++i) {
    const double pos = static_cast<double>(i) * inv_ratio;
    const std::size_t idx0 = static_cast<std::size_t>(std::floor(pos));
    const std::size_t idx1 = std::min(idx0 + 1, samples.size() - 1);
    const double t = pos - static_cast<double>(idx0);
    out[i] = static_cast<float>((1.0 - t) * samples[idx0] + t * samples[idx1]);
  }
  return out;
}

struct WavData
{
  std::int32_t sample_rate{0};
  std::int32_t channels{0};
  std::int32_t bit_depth{0};
  std::vector<float> samples;
};

std::uint32_t read_u32(std::ifstream &ifs)
{
  std::uint32_t v = 0;
  ifs.read(reinterpret_cast<char *>(&v), sizeof(v));
  return v;
}

std::uint16_t read_u16(std::ifstream &ifs)
{
  std::uint16_t v = 0;
  ifs.read(reinterpret_cast<char *>(&v), sizeof(v));
  return v;
}

WavData load_wav(const std::string &path)
{
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) {
    throw std::runtime_error("Failed to open wav: " + path);
  }

  char riff[4];
  ifs.read(riff, 4);
  if (ifs.gcount() != 4 || std::string(riff, 4) != "RIFF") {
    throw std::runtime_error("Not a RIFF file: " + path);
  }
  read_u32(ifs);  // chunk size
  char wave[4];
  ifs.read(wave, 4);
  if (ifs.gcount() != 4 || std::string(wave, 4) != "WAVE") {
    throw std::runtime_error("Not a WAVE file: " + path);
  }

  std::int32_t sample_rate = 0;
  std::int32_t channels = 0;
  std::int32_t bit_depth = 0;
  std::vector<std::uint8_t> data_bytes;

  while (ifs && !ifs.eof()) {
    char chunk_id[4];
    ifs.read(chunk_id, 4);
    if (ifs.gcount() != 4) {
      break;
    }
    const std::string id(chunk_id, 4);
    std::uint32_t chunk_size = read_u32(ifs);

    if (id == "fmt ") {
      std::uint16_t audio_format = read_u16(ifs);
      channels = static_cast<std::int32_t>(read_u16(ifs));
      sample_rate = static_cast<std::int32_t>(read_u32(ifs));
      read_u32(ifs);                 // byte_rate
      read_u16(ifs);                 // block_align
      bit_depth = read_u16(ifs);     // bits per sample
      if (chunk_size > 16) {
        ifs.seekg(chunk_size - 16, std::ios::cur);  // skip extra fmt bytes
      }
      if (audio_format != 1) {
        throw std::runtime_error("Only PCM wav is supported");
      }
    } else if (id == "data") {
      data_bytes.resize(chunk_size);
      ifs.read(reinterpret_cast<char *>(data_bytes.data()), chunk_size);
    } else {
      // skip other chunks
      ifs.seekg(chunk_size, std::ios::cur);
    }
  }

  if (sample_rate <= 0 || channels <= 0 || bit_depth <= 0 || data_bytes.empty()) {
    throw std::runtime_error("Incomplete wav header: " + path);
  }

  WavData out;
  out.sample_rate = sample_rate;
  out.channels = channels;
  out.bit_depth = bit_depth;

  if (bit_depth == 16) {
    const std::size_t count = data_bytes.size() / sizeof(std::int16_t);
    out.samples.resize(count);
    const auto *raw = reinterpret_cast<const std::int16_t *>(data_bytes.data());
    constexpr float kScale = 1.0f / 32768.0f;
    for (std::size_t i = 0; i < count; ++i) {
      out.samples[i] = static_cast<float>(raw[i]) * kScale;
    }
  } else if (bit_depth == 32) {
    const std::size_t count = data_bytes.size() / sizeof(float);
    out.samples.resize(count);
    const auto *raw = reinterpret_cast<const float *>(data_bytes.data());
    for (std::size_t i = 0; i < count; ++i) {
      out.samples[i] = raw[i];
    }
  } else {
    throw std::runtime_error("Unsupported bit depth in wav: " + std::to_string(bit_depth));
  }

  if (channels > 1) {
    const std::size_t frames = out.samples.size() / static_cast<std::size_t>(channels);
    std::vector<float> mono(frames, 0.0f);
    for (std::size_t i = 0; i < frames; ++i) {
      float sum = 0.0f;
      for (std::size_t ch = 0; ch < static_cast<std::size_t>(channels); ++ch) {
        sum += out.samples[i * static_cast<std::size_t>(channels) + ch];
      }
      mono[i] = sum / static_cast<float>(channels);
    }
    out.samples = std::move(mono);
    out.channels = 1;
  }

  return out;
}

std::vector<std::string> list_wav_files(const std::string &dir_path)
{
  std::vector<std::string> files;
  DIR *dir = opendir(dir_path.c_str());
  if (!dir) {
    throw std::runtime_error("Cannot open directory: " + dir_path);
  }

  struct dirent *entry;
  while ((entry = readdir(dir)) != nullptr) {
    std::string name = entry->d_name;
    if (name.size() > 4 && name.substr(name.size() - 4) == ".wav") {
      files.push_back(dir_path + "/" + name);
    }
  }
  closedir(dir);
  std::sort(files.begin(), files.end());
  return files;
}

struct Args
{
  std::string wav_path;
  std::string wav_dir;  // batch mode: directory of WAV files
  std::string encoder;
  std::string decoder;
  std::string joiner;
  std::string tokens;
  std::string keywords;
  std::string provider;
  float keywords_threshold{0.25f};
  int target_sample_rate{16000};
  int chunk_ms{20};
  int max_active_paths{4};
  bool batch_mode{false};
};

Args parse_args(int argc, char **argv)
{
  if (argc < 15) {
    throw std::runtime_error(
      "Usage: kws_wav_tool --wav <path> --encoder <path> --decoder <path> "
      "--joiner <path> --tokens <path> --keywords <path> --provider <provider> [--threshold 0.25] "
      "[--sample_rate 16000] [--chunk_ms 20]\n"
      "   or: kws_wav_tool --batch <dir> --encoder <path> --decoder <path> "
      "--joiner <path> --tokens <path> --keywords <path> --provider <provider> [--threshold 0.25]");
  }
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--wav" && i + 1 < argc) {
      args.wav_path = argv[++i];
    } else if (arg == "--batch" && i + 1 < argc) {
      args.wav_dir = argv[++i];
      args.batch_mode = true;
    } else if (arg == "--encoder" && i + 1 < argc) {
      args.encoder = argv[++i];
    } else if (arg == "--decoder" && i + 1 < argc) {
      args.decoder = argv[++i];
    } else if (arg == "--joiner" && i + 1 < argc) {
      args.joiner = argv[++i];
    } else if (arg == "--tokens" && i + 1 < argc) {
      args.tokens = argv[++i];
    } else if (arg == "--keywords" && i + 1 < argc) {
      args.keywords = argv[++i];
    } else if (arg == "--provider" && i + 1 < argc) {
      args.provider = argv[++i];
    } else if (arg == "--threshold" && i + 1 < argc) {
      args.keywords_threshold = std::stof(argv[++i]);
    } else if (arg == "--sample_rate" && i + 1 < argc) {
      args.target_sample_rate = std::stoi(argv[++i]);
    } else if (arg == "--chunk_ms" && i + 1 < argc) {
      args.chunk_ms = std::stoi(argv[++i]);
    } else if (arg == "--max_active_paths" && i + 1 < argc) {
      args.max_active_paths = std::stoi(argv[++i]);
    }
  }
  if (args.batch_mode) {
    if (args.wav_dir.empty() || args.encoder.empty() || args.decoder.empty() ||
        args.joiner.empty() || args.tokens.empty() || args.keywords.empty() ||
        args.provider.empty()) {
      throw std::runtime_error("Missing required arguments for batch mode");
    }
  } else {
    if (args.wav_path.empty() || args.encoder.empty() || args.decoder.empty() ||
        args.joiner.empty() || args.tokens.empty() || args.keywords.empty() ||
        args.provider.empty()) {
      throw std::runtime_error("Missing required arguments");
    }
  }
  if (!fa_kws::isSupportedSherpaOnnxExecutionProvider(args.provider)) {
    throw std::runtime_error(
      "unsupported backend.execution_provider: " + args.provider +
      "; supported providers: " + fa_kws::supportedSherpaOnnxExecutionProvidersForMessage());
  }
  return args;
}

}  // namespace

// Process a single WAV file with an existing engine
// Returns true if keyword was detected
bool process_single_wav(fa_kws::SherpaOnnxKwsBackend &engine,
                        const std::string &wav_path,
                        const Args &args,
                        bool verbose)
{
  const WavData wav = load_wav(wav_path);
  const auto resampled =
    resample_linear(wav.samples, wav.sample_rate, args.target_sample_rate);
  const std::size_t chunk =
    static_cast<std::size_t>(args.target_sample_rate * args.chunk_ms / 1000);

  if (verbose) {
    std::cout << "Loaded wav: " << wav_path
              << " sr=" << wav.sample_rate
              << " frames=" << wav.samples.size()
              << " resampled=" << resampled.size()
              << " chunk=" << chunk
              << " threshold=" << args.keywords_threshold
              << std::endl;
  }

  std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
  for (std::size_t offset = 0; offset < resampled.size(); offset += chunk) {
    const std::size_t len = std::min(chunk, resampled.size() - offset);
    std::vector<float> slice(resampled.begin() + static_cast<std::ptrdiff_t>(offset),
                             resampled.begin() + static_cast<std::ptrdiff_t>(offset + len));
    now += std::chrono::milliseconds(args.chunk_ms);
    auto det = engine.process(slice,
                              static_cast<std::int32_t>(args.target_sample_rate),
                              /*vad_prob=*/1.0f,
                              now);
    if (det) {
      if (verbose) {
        std::cout << "DETECTED keyword=" << det->keyword
                  << " start_time=" << det->start_time_sec << std::endl;
      }
      return true;
    }
  }
  return false;
}

int main(int argc, char **argv)
{
  try {
    const Args args = parse_args(argc, argv);

    fa_kws::SherpaOnnxKwsBackendConfig cfg;
    cfg.target_sample_rate = args.target_sample_rate;
    cfg.encoder_path = args.encoder;
    cfg.decoder_path = args.decoder;
    cfg.joiner_path = args.joiner;
    cfg.tokens_path = args.tokens;
    cfg.keywords_path = args.keywords;
    cfg.execution_provider = args.provider;
    cfg.keywords_threshold = args.keywords_threshold;
    cfg.max_active_paths = args.max_active_paths;
    cfg.vad_threshold = 0.0f;  // disable gating
    cfg.cooldown = std::chrono::milliseconds{0};

    if (args.batch_mode) {
      // Batch mode: process all WAV files in directory with single model load
      const auto wav_files = list_wav_files(args.wav_dir);
      if (wav_files.empty()) {
        std::cerr << "No WAV files found in: " << args.wav_dir << std::endl;
        return 1;
      }

      std::cout << "Batch mode: " << wav_files.size() << " files, threshold="
                << args.keywords_threshold << std::endl;

      // Create engine once
      fa_kws::SherpaOnnxKwsBackend engine(cfg);

      int detected_count = 0;
      for (std::size_t i = 0; i < wav_files.size(); ++i) {
        const auto &wav_path = wav_files[i];
        // Extract just filename for output
        std::string filename = wav_path;
        auto pos = filename.rfind('/');
        if (pos != std::string::npos) {
          filename = filename.substr(pos + 1);
        }

        bool detected = process_single_wav(engine, wav_path, args, false);
        std::cout << filename << "\t" << (detected ? "OK" : "MISS") << std::endl;
        if (detected) {
          ++detected_count;
        }
        // Hard reset: destroy and recreate stream for clean state
        engine.resetHard();
      }

      // Summary
      std::cout << "---" << std::endl;
      std::cout << "Total: " << wav_files.size()
                << " Detected: " << detected_count
                << " Missed: " << (wav_files.size() - detected_count)
                << " Rate: " << (100.0 * detected_count / wav_files.size()) << "%"
                << std::endl;

      return (detected_count > 0) ? 0 : 1;
    } else {
      // Single file mode
      fa_kws::SherpaOnnxKwsBackend engine(cfg);

      const WavData wav = load_wav(args.wav_path);
      const auto resampled =
        resample_linear(wav.samples, wav.sample_rate, args.target_sample_rate);
      const std::size_t chunk =
        static_cast<std::size_t>(args.target_sample_rate * args.chunk_ms / 1000);

      std::cout << "Loaded wav: " << args.wav_path
                << " sr=" << wav.sample_rate
                << " frames=" << wav.samples.size()
                << " resampled=" << resampled.size()
                << " chunk=" << chunk
                << " threshold=" << args.keywords_threshold
                << std::endl;

      std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
      bool detected = false;
      for (std::size_t offset = 0; offset < resampled.size(); offset += chunk) {
        const std::size_t len = std::min(chunk, resampled.size() - offset);
        std::vector<float> slice(resampled.begin() + static_cast<std::ptrdiff_t>(offset),
                                 resampled.begin() + static_cast<std::ptrdiff_t>(offset + len));
        now += std::chrono::milliseconds(args.chunk_ms);
        auto det = engine.process(slice,
                                  static_cast<std::int32_t>(args.target_sample_rate),
                                  /*vad_prob=*/1.0f,
                                  now);
        if (det) {
          std::cout << "DETECTED keyword=" << det->keyword
                    << " start_time=" << det->start_time_sec << std::endl;
          detected = true;
        }
      }

      if (!detected) {
        std::cout << "No keyword detected" << std::endl;
        return 1;
      }
      return 0;
    }
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
