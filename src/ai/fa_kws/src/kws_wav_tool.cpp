#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "fa_kws/backends/sherpa_onnx_kws_backend.hpp"

namespace {

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
  std::int32_t audio_format = 0;
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
      if (chunk_size < 16) {
        throw std::runtime_error("WAV fmt chunk is too small: " + path);
      }
      audio_format = static_cast<std::int32_t>(read_u16(ifs));
      channels = static_cast<std::int32_t>(read_u16(ifs));
      sample_rate = static_cast<std::int32_t>(read_u32(ifs));
      read_u32(ifs);                 // byte_rate
      read_u16(ifs);                 // block_align
      bit_depth = read_u16(ifs);     // bits per sample
      if (chunk_size > 16) {
        ifs.seekg(chunk_size - 16, std::ios::cur);  // skip extra fmt bytes
      }
    } else if (id == "data") {
      data_bytes.resize(chunk_size);
      ifs.read(reinterpret_cast<char *>(data_bytes.data()), chunk_size);
      if (ifs.gcount() != static_cast<std::streamsize>(chunk_size)) {
        throw std::runtime_error("Failed to read WAV data chunk: " + path);
      }
    } else {
      // skip other chunks
      ifs.seekg(chunk_size, std::ios::cur);
    }
  }

  if (sample_rate <= 0 || channels <= 0 || bit_depth <= 0 || data_bytes.empty()) {
    throw std::runtime_error("Incomplete wav header: " + path);
  }
  if (audio_format != 3) {
    throw std::runtime_error("WAV must be IEEE float format");
  }
  if (channels != 1) {
    throw std::runtime_error("WAV channels must be 1");
  }
  if (bit_depth != 32) {
    throw std::runtime_error("WAV bit_depth must be 32");
  }
  if (data_bytes.size() % sizeof(float) != 0) {
    throw std::runtime_error("WAV float32 data length is not byte-aligned");
  }

  WavData out;
  out.sample_rate = sample_rate;
  out.channels = channels;
  out.bit_depth = bit_depth;

  const std::size_t count = data_bytes.size() / sizeof(float);
  out.samples.resize(count);
  for (std::size_t i = 0; i < count; ++i) {
    float sample = 0.0f;
    std::memcpy(&sample, data_bytes.data() + (i * sizeof(float)), sizeof(float));
    if (!std::isfinite(sample)) {
      throw std::runtime_error("WAV contains non-finite samples");
    }
    if (sample < -1.0f || sample > 1.0f) {
      throw std::runtime_error("WAV samples must be normalized to [-1.0, 1.0]");
    }
    out.samples[i] = sample;
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
  std::string command;
  std::string workspace_dir;
  double timeout_sec{std::numeric_limits<double>::quiet_NaN()};
  float keywords_threshold{std::numeric_limits<float>::quiet_NaN()};
  int target_sample_rate{0};
  int chunk_ms{20};
  int model_num_threads{0};
  int max_active_paths{0};
  int num_trailing_blanks{-1};
  float keywords_score{std::numeric_limits<float>::quiet_NaN()};
  bool batch_mode{false};
  bool assume_vad_speech{false};
};

Args parse_args(int argc, char **argv)
{
  if (argc < 15) {
    throw std::runtime_error(
      "Usage: kws_wav_tool --wav <path> --encoder <path> --decoder <path> "
      "--joiner <path> --tokens <path> --keywords <path> --provider <provider> "
      "--command <path> --workspace_dir <dir> --timeout_sec <sec> "
      "--threshold <float> --sample_rate <hz> --num_threads <n> "
      "--max_active_paths <n> --num_trailing_blanks <n> --keywords_score <float> "
      "[--chunk_ms 20] --assume-vad-speech\n"
      "   or: kws_wav_tool --batch <dir> --encoder <path> --decoder <path> "
      "--joiner <path> --tokens <path> --keywords <path> --provider <provider> "
      "--command <path> --workspace_dir <dir> --timeout_sec <sec> "
      "--threshold <float> --sample_rate <hz> --num_threads <n> "
      "--max_active_paths <n> --num_trailing_blanks <n> --keywords_score <float> "
      "[--chunk_ms 20] --assume-vad-speech");
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
    } else if (arg == "--command" && i + 1 < argc) {
      args.command = argv[++i];
    } else if (arg == "--workspace_dir" && i + 1 < argc) {
      args.workspace_dir = argv[++i];
    } else if (arg == "--timeout_sec" && i + 1 < argc) {
      args.timeout_sec = std::stod(argv[++i]);
    } else if (arg == "--threshold" && i + 1 < argc) {
      args.keywords_threshold = std::stof(argv[++i]);
    } else if (arg == "--sample_rate" && i + 1 < argc) {
      args.target_sample_rate = std::stoi(argv[++i]);
    } else if (arg == "--chunk_ms" && i + 1 < argc) {
      args.chunk_ms = std::stoi(argv[++i]);
    } else if (arg == "--num_threads" && i + 1 < argc) {
      args.model_num_threads = std::stoi(argv[++i]);
    } else if (arg == "--max_active_paths" && i + 1 < argc) {
      args.max_active_paths = std::stoi(argv[++i]);
    } else if (arg == "--num_trailing_blanks" && i + 1 < argc) {
      args.num_trailing_blanks = std::stoi(argv[++i]);
    } else if (arg == "--keywords_score" && i + 1 < argc) {
      args.keywords_score = std::stof(argv[++i]);
    } else if (arg == "--assume-vad-speech") {
      args.assume_vad_speech = true;
    }
  }
  if (args.batch_mode) {
    if (args.wav_dir.empty() || args.encoder.empty() || args.decoder.empty() ||
        args.joiner.empty() || args.tokens.empty() || args.keywords.empty() ||
        args.provider.empty() || args.command.empty() || args.workspace_dir.empty()) {
      throw std::runtime_error("Missing required arguments for batch mode");
    }
  } else {
    if (args.wav_path.empty() || args.encoder.empty() || args.decoder.empty() ||
        args.joiner.empty() || args.tokens.empty() || args.keywords.empty() ||
        args.provider.empty() || args.command.empty() || args.workspace_dir.empty()) {
      throw std::runtime_error("Missing required arguments");
    }
  }
  if (!fa_kws::isSupportedSherpaOnnxExecutionProvider(args.provider)) {
    throw std::runtime_error(
      "unsupported backend.execution_provider: " + args.provider +
      "; supported providers: " + fa_kws::supportedSherpaOnnxExecutionProvidersForMessage());
  }
  if (args.target_sample_rate <= 0) {
    throw std::runtime_error("sample_rate must be positive");
  }
  if (args.chunk_ms <= 0) {
    throw std::runtime_error("chunk_ms must be positive");
  }
  if (args.model_num_threads <= 0) {
    throw std::runtime_error("num_threads must be positive");
  }
  if (args.max_active_paths <= 0) {
    throw std::runtime_error("max_active_paths must be positive");
  }
  if (args.num_trailing_blanks < 0) {
    throw std::runtime_error("num_trailing_blanks must be zero or positive");
  }
  if (!std::isfinite(args.keywords_score) || args.keywords_score <= 0.0f) {
    throw std::runtime_error("keywords_score must be finite and positive");
  }
  if (!std::isfinite(args.keywords_threshold) || args.keywords_threshold <= 0.0f) {
    throw std::runtime_error("threshold must be finite and positive");
  }
  if (!std::isfinite(args.timeout_sec) || args.timeout_sec <= 0.0) {
    throw std::runtime_error("timeout_sec must be finite and positive");
  }
  if (!args.assume_vad_speech) {
    throw std::runtime_error(
      "kws_wav_tool requires explicit VAD policy: "
      "pass --assume-vad-speech for canonical speech fixtures");
  }
  return args;
}

void validate_wav_contract(const WavData &wav,
                           const Args &args,
                           const std::string &wav_path)
{
  if (wav.sample_rate != args.target_sample_rate) {
    throw std::runtime_error(
      "WAV sample_rate must match --sample_rate for " + wav_path +
      ": got " + std::to_string(wav.sample_rate) +
      " expected " + std::to_string(args.target_sample_rate));
  }
  if (wav.channels != 1) {
    throw std::runtime_error("WAV channels must be 1 for " + wav_path);
  }
  if (wav.bit_depth != 32) {
    throw std::runtime_error("WAV bit_depth must be 32 for " + wav_path);
  }
}

fa_kws::SherpaOnnxKwsBackendConfig build_backend_config(const Args &args)
{
  fa_kws::SherpaOnnxKwsBackendConfig cfg;
  cfg.target_sample_rate = args.target_sample_rate;
  cfg.model_num_threads = args.model_num_threads;
  cfg.execution_provider = args.provider;
  cfg.encoder_path = args.encoder;
  cfg.decoder_path = args.decoder;
  cfg.joiner_path = args.joiner;
  cfg.tokens_path = args.tokens;
  cfg.keywords_path = args.keywords;
  cfg.max_active_paths = args.max_active_paths;
  cfg.num_trailing_blanks = args.num_trailing_blanks;
  cfg.keywords_score = args.keywords_score;
  cfg.keywords_threshold = args.keywords_threshold;
  cfg.vad_threshold = 1.0f;
  cfg.cooldown = std::chrono::milliseconds{0};
  cfg.command = args.command;
  cfg.args = {
    "detect",
    "--audio",
    "{audio}",
    "--encoder",
    "{encoder}",
    "--decoder",
    "{decoder}",
    "--joiner",
    "{joiner}",
    "--tokens",
    "{tokens}",
    "--keywords",
    "{keywords}",
    "--provider",
    "{provider}",
    "--sample-rate",
    "{sample_rate}",
    "--num-threads",
    "{num_threads}",
    "--max-active-paths",
    "{max_active_paths}",
    "--num-trailing-blanks",
    "{num_trailing_blanks}",
    "--keywords-score",
    "{keywords_score}",
    "--keywords-threshold",
    "{keywords_threshold}",
  };
  cfg.health_args = {
    "health",
    "--encoder",
    "{encoder}",
    "--decoder",
    "{decoder}",
    "--joiner",
    "{joiner}",
    "--tokens",
    "{tokens}",
    "--keywords",
    "{keywords}",
    "--provider",
    "{provider}",
  };
  cfg.timeout_sec = args.timeout_sec;
  cfg.workspace_dir = args.workspace_dir;
  cfg.cleanup_audio_files = true;
  return cfg;
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
  validate_wav_contract(wav, args, wav_path);
  const std::size_t chunk =
    static_cast<std::size_t>(args.target_sample_rate * args.chunk_ms / 1000);
  if (chunk == 0) {
    throw std::runtime_error("chunk size must be at least one sample");
  }

  if (verbose) {
    std::cout << "Loaded wav: " << wav_path
              << " sr=" << wav.sample_rate
              << " frames=" << wav.samples.size()
              << " chunk=" << chunk
              << " threshold=" << args.keywords_threshold
              << " vad_policy=assume_speech"
              << std::endl;
  }

  std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
  for (std::size_t offset = 0; offset < wav.samples.size(); offset += chunk) {
    const std::size_t len = std::min(chunk, wav.samples.size() - offset);
    std::vector<float> slice(wav.samples.begin() + static_cast<std::ptrdiff_t>(offset),
                             wav.samples.begin() + static_cast<std::ptrdiff_t>(offset + len));
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
    const fa_kws::SherpaOnnxKwsBackendConfig cfg = build_backend_config(args);

    if (args.batch_mode) {
      // Batch mode: process all WAV files in directory with single model load
      const auto wav_files = list_wav_files(args.wav_dir);
      if (wav_files.empty()) {
        std::cerr << "No WAV files found in: " << args.wav_dir << std::endl;
        return 1;
      }
      for (const auto &wav_path : wav_files) {
        validate_wav_contract(load_wav(wav_path), args, wav_path);
      }

      std::cout << "Batch mode: " << wav_files.size() << " files, threshold="
                << args.keywords_threshold
                << " vad_policy=assume_speech"
                << std::endl;

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
      const WavData wav = load_wav(args.wav_path);
      validate_wav_contract(wav, args, args.wav_path);
      const std::size_t chunk =
        static_cast<std::size_t>(args.target_sample_rate * args.chunk_ms / 1000);
      if (chunk == 0) {
        throw std::runtime_error("chunk size must be at least one sample");
      }

      fa_kws::SherpaOnnxKwsBackend engine(cfg);

      std::cout << "Loaded wav: " << args.wav_path
                << " sr=" << wav.sample_rate
                << " frames=" << wav.samples.size()
                << " chunk=" << chunk
                << " threshold=" << args.keywords_threshold
                << " vad_policy=assume_speech"
                << std::endl;

      std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
      bool detected = false;
      for (std::size_t offset = 0; offset < wav.samples.size(); offset += chunk) {
        const std::size_t len = std::min(chunk, wav.samples.size() - offset);
        std::vector<float> slice(wav.samples.begin() + static_cast<std::ptrdiff_t>(offset),
                                 wav.samples.begin() + static_cast<std::ptrdiff_t>(offset + len));
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
