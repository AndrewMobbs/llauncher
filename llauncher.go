package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"os/signal"
	"reflect"
	"strconv"
	"syscall"

	"gopkg.in/yaml.v3"
)

// LlamaConfig defines the structure of the YAML configuration file.
// The `yaml` struct tags map YAML keys to the struct fields.
// The `arg` struct tags define the corresponding command-line flag, which is
// now the single source of truth for argument generation.
type LlamaConfig struct {
	// Basic server configuration
	ModelPath      string   `yaml:"model" arg:"--model"`
	ModelUrl       string   `yaml:"model-url" arg:"--model-url"`
	Host           string   `yaml:"host" arg:"--host"`
	Port           int      `yaml:"port" arg:"--port"`
	Path           string   `yaml:"path" arg:"--path"`
	ApiPrefix      string   `yaml:"api-prefix" arg:"--api-prefix"`
	NoWebUI        bool     `yaml:"no-webui" arg:"--no-webui"`
	Timeout        int      `yaml:"timeout" arg:"--timeout"`
	ThreadsHTTP    int      `yaml:"threads-http" arg:"--threads-http"`
	
	// Model loading and configuration
	HfRepo         string   `yaml:"hf-repo" arg:"--hf-repo"`
	HfFile         string   `yaml:"hf-file" arg:"--hf-file"`
	HfToken        string   `yaml:"hf-token" arg:"--hf-token"`
	Offline        bool     `yaml:"offline" arg:"--offline"`
	
	// Performance and resource configuration
	Threads        int      `yaml:"threads" arg:"--threads"`
	ThreadsBatch   int      `yaml:"threads-batch" arg:"--threads-batch"`
	CpuMask        string   `yaml:"cpu-mask" arg:"--cpu-mask"`
	CpuRange       string   `yaml:"cpu-range" arg:"--cpu-range"`
	CpuStrict      int      `yaml:"cpu-strict" arg:"--cpu-strict"`
	Priority       int      `yaml:"prio" arg:"--prio"`
	Poll           int      `yaml:"poll" arg:"--poll"`
	ContextSize    int      `yaml:"n-ctx" arg:"--ctx-size"`
	BatchSize      int      `yaml:"batch-size" arg:"--batch-size"`
	UBatchSize     int      `yaml:"ubatch-size" arg:"--ubatch-size"`
	GpuLayers      int      `yaml:"n-gpu-layers" arg:"--n-gpu-layers"`
	SplitMode      string   `yaml:"split-mode" arg:"--split-mode"`
	TensorSplit    string   `yaml:"tensor-split" arg:"--tensor-split"`
	MainGPU        int      `yaml:"main-gpu" arg:"--main-gpu"`
	Numa           string   `yaml:"numa" arg:"--numa"`
	Device         string   `yaml:"device" arg:"--device"`
	
	// Memory management
	Mlock          bool     `yaml:"mlock" arg:"--mlock"`
	NoMMap         bool     `yaml:"no-mmap" arg:"--no-mmap"`
	CacheTypeK     string   `yaml:"cache-type-k" arg:"--cache-type-k"`
	CacheTypeV     string   `yaml:"cache-type-v" arg:"--cache-type-v"`
	CacheReuse     int      `yaml:"cache-reuse" arg:"--cache-reuse"`
	SwaFull        bool     `yaml:"swa-full" arg:"--swa-full"`
	KvUnified      bool     `yaml:"kv-unified" arg:"--kv-unified"`
	
	// RoPE configuration
	RopeScaling    string   `yaml:"rope-scaling" arg:"--rope-scaling"`
	RopeScale      float64  `yaml:"rope-scale" arg:"--rope-scale"`
	RopeFreqBase   float64  `yaml:"rope-freq-base" arg:"--rope-freq-base"`
	RopeFreqScale  float64  `yaml:"rope-freq-scale" arg:"--rope-freq-scale"`
	
	// YaRN configuration
	YarnOrigCtx    int      `yaml:"yarn-orig-ctx" arg:"--yarn-orig-ctx"`
	YarnExtFactor  float64  `yaml:"yarn-ext-factor" arg:"--yarn-ext-factor"`
	YarnAttnFactor float64  `yaml:"yarn-attn-factor" arg:"--yarn-attn-factor"`
	YarnBetaSlow   float64  `yaml:"yarn-beta-slow" arg:"--yarn-beta-slow"`
	YarnBetaFast   float64  `yaml:"yarn-beta-fast" arg:"--yarn-beta-fast"`
	
	// Sampling parameters
	Seed           int      `yaml:"seed" arg:"--seed"`
	Samplers       string   `yaml:"samplers" arg:"--samplers"`
	SamplerSeq     string   `yaml:"sampler-seq" arg:"--sampling-seq"`
	IgnoreEOS      bool     `yaml:"ignore-eos" arg:"--ignore-eos"`
	Temperature    float64  `yaml:"temp" arg:"--temp"`
	TopK           int      `yaml:"top-k" arg:"--top-k"`
	TopP           float64  `yaml:"top-p" arg:"--top-p"`
	MinP           float64  `yaml:"min-p" arg:"--min-p"`
	TopNSigma      float64  `yaml:"top-nsigma" arg:"--top-nsigma"`
	Typical        float64  `yaml:"typical" arg:"--typical"`
	RepeatLastN    int      `yaml:"repeat-last-n" arg:"--repeat-last-n"`
	RepeatPenalty  float64  `yaml:"repeat-penalty" arg:"--repeat-penalty"`
	PresencePenalty float64 `yaml:"presence-penalty" arg:"--presence-penalty"`
	FrequencyPenalty float64 `yaml:"frequency-penalty" arg:"--frequency-penalty"`
	Mirostat       int      `yaml:"mirostat" arg:"--mirostat"`
	MirostatLR     float64  `yaml:"mirostat-lr" arg:"--mirostat-lr"`
	MirostatEnt    float64  `yaml:"mirostat-ent" arg:"--mirostat-ent"`
	
	// Grammar and constraints
	Grammar        string   `yaml:"grammar" arg:"--grammar"`
	GrammarFile    string   `yaml:"grammar-file" arg:"--grammar-file"`
	JsonSchema     string   `yaml:"json-schema" arg:"--json-schema"`
	JsonSchemaFile string   `yaml:"json-schema-file" arg:"--json-schema-file"`
	
	// Adapters and extensions
	LoraAdapters   []string `yaml:"lora" arg:"--lora"`
	LoraScaled     []string `yaml:"lora-scaled" arg:"--lora-scaled"`
	MmProj         string   `yaml:"mmproj" arg:"--mmproj"`
	MmProjUrl      string   `yaml:"mmproj-url" arg:"--mmproj-url"`
	NoMmProj       bool     `yaml:"no-mmproj" arg:"--no-mmproj"`
	NoMmProjOffload bool    `yaml:"no-mmproj-offload" arg:"--no-mmproj-offload"`
	
	// Server features
	ContBatching   bool     `yaml:"cont-batching" arg:"--cont-batching"`
	NoContBatching bool     `yaml:"no-cont-batching" arg:"--no-cont-batching"`
	Metrics        bool     `yaml:"metrics" arg:"--metrics"`
	Slots          bool     `yaml:"slots" arg:"--slots"`
	NoSlots        bool     `yaml:"no-slots" arg:"--no-slots"`
	SlotSavePath   string   `yaml:"slot-save-path" arg:"--slot-save-path"`
	SlotPromptSimilarity float64 `yaml:"slot-prompt-similarity" arg:"--slot-prompt-similarity"`
	SwaCheckpoints int      `yaml:"swa-checkpoints" arg:"--swa-checkpoints"`
	
	// Authentication and security
	ApiKey         string   `yaml:"api-key" arg:"--api-key"`
	ApiKeyFile     string   `yaml:"api-key-file" arg:"--api-key-file"`
	SslKeyFile     string   `yaml:"ssl-key-file" arg:"--ssl-key-file"`
	SslCertFile    string   `yaml:"ssl-cert-file" arg:"--ssl-cert-file"`
	
	// Chat and template configuration
	ChatTemplate   string   `yaml:"chat-template" arg:"--chat-template"`
	ChatTemplateFile string `yaml:"chat-template-file" arg:"--chat-template-file"`
	ChatTemplateKwargs string `yaml:"chat-template-kwargs" arg:"--chat-template-kwargs"`
	Jinja          bool     `yaml:"jinja" arg:"--jinja"`
	NoPrefillAssistant bool `yaml:"no-prefill-assistant" arg:"--no-prefill-assistant"`
	ReasoningFormat string `yaml:"reasoning-format" arg:"--reasoning-format"`
	ReasoningBudget int    `yaml:"reasoning-budget" arg:"--reasoning-budget"`
	
	// Special use cases
	Embedding      bool     `yaml:"embedding" arg:"--embedding"`
	Reranking      bool     `yaml:"reranking" arg:"--reranking"`
	Pooling        string   `yaml:"pooling" arg:"--pooling"`
	
	// Logging
	Verbose        bool     `yaml:"verbose" arg:"--verbose"`
	LogDisable     bool     `yaml:"log-disable" arg:"--log-disable"`
	LogFile        string   `yaml:"log-file" arg:"--log-file"`
	LogColors      bool     `yaml:"log-colors" arg:"--log-colors"`
	LogVerbosity   int      `yaml:"log-verbosity" arg:"--log-verbosity"`
	LogPrefix      bool     `yaml:"log-prefix" arg:"--log-prefix"`
	LogTimestamps  bool     `yaml:"log-timestamps" arg:"--log-timestamps"`
	
	// Prediction and generation
	Predict        int      `yaml:"n-predict" arg:"--predict"`
	ReversePrompt  string   `yaml:"reverse-prompt" arg:"--reverse-prompt"`
	Special        bool     `yaml:"special" arg:"--special"`
	NoWarmup       bool     `yaml:"no-warmup" arg:"--no-warmup"`
	NoContextShift bool     `yaml:"no-context-shift" arg:"--no-context-shift"`
	ContextShift   bool     `yaml:"context-shift" arg:"--context-shift"`
	Keep           int      `yaml:"keep" arg:"--keep"`
	
	// Speculative decoding
	ModelDraft     string   `yaml:"model-draft" arg:"--model-draft"`
	ThreadsDraft   int      `yaml:"threads-draft" arg:"--threads-draft"`
	ThreadsBatchDraft int   `yaml:"threads-batch-draft" arg:"--threads-batch-draft"`
	ContextSizeDraft int    `yaml:"ctx-size-draft" arg:"--ctx-size-draft"`
	DeviceDraft    string   `yaml:"device-draft" arg:"--device-draft"`
	GpuLayersDraft int      `yaml:"n-gpu-layers-draft" arg:"--gpu-layers-draft"`
	DraftMax       int      `yaml:"draft-max" arg:"--draft-max"`
	DraftMin       int      `yaml:"draft-min" arg:"--draft-min"`
	DraftPMin      float64  `yaml:"draft-p-min" arg:"--draft-p-min"`
}

// showHelp displays usage information for the launcher
func showHelp() {
	fmt.Println("llauncher - A launcher for llama-server")
	fmt.Println("\nUsage:")
	fmt.Println("  llauncher [--config <config_file>] [--help]")
	fmt.Println("\nOptions:")
	fmt.Println("  --config <file>    Path to YAML configuration file")
	fmt.Println("  --help             Show this help message")
	fmt.Println("\nEnvironment Variables:")
	fmt.Println("  LLAMA_CONFIG_PATH  Path to YAML configuration file (overridden by --config)")
	fmt.Println("\nConfiguration File Format (YAML):")
	fmt.Println("  # Basic server configuration")
	fmt.Println("  model: path/to/model.gguf       # Path to the model file")
	fmt.Println("  model-url: url                  # URL to download model from")
	fmt.Println("  host: 127.0.0.1                 # Host to bind to")
	fmt.Println("  port: 8080                      # Port to listen on")
	fmt.Println("  path: /static                   # Path to serve static files from")
	fmt.Println("  api-prefix: /api                # API prefix path")
	fmt.Println("  timeout: 600                    # Server timeout in seconds")
	fmt.Println("  threads-http: 4                 # Number of HTTP threads")
	fmt.Println("")
	fmt.Println("  # Performance configuration")
	fmt.Println("  threads: 4                      # Number of threads to use")
	fmt.Println("  threads-batch: 4                # Number of batch threads")
	fmt.Println("  n-ctx: 2048                     # Context size")
	fmt.Println("  batch-size: 512                 # Batch size")
	fmt.Println("  n-gpu-layers: 0                 # Number of GPU layers")
	fmt.Println("  device: cpu                     # Device to use (cpu, cuda, etc)")
	fmt.Println("  numa: none                      # NUMA configuration")
	fmt.Println("")
	fmt.Println("  # Memory management")
	fmt.Println("  mlock: true                     # Lock memory to prevent swapping")
	fmt.Println("  cache-type-k: f16               # KV cache type for K")
	fmt.Println("  cache-type-v: f16               # KV cache type for V")
	fmt.Println("  cache-reuse: 10                 # KV cache reuse threshold")
	fmt.Println("")
	fmt.Println("  # Model extensions")
	fmt.Println("  lora:                           # LoRA adapters")
	fmt.Println("    - adapter1.bin")
	fmt.Println("    - adapter2.bin")
	fmt.Println("  mmproj: vision.bin              # Multimodal projector file")
	fmt.Println("")
	fmt.Println("  # Sampling parameters")
	fmt.Println("  seed: 42                        # RNG seed")
	fmt.Println("  temp: 0.8                       # Temperature")
	fmt.Println("  top-k: 40                       # Top-K sampling")
	fmt.Println("  top-p: 0.9                      # Top-P sampling")
	fmt.Println("")
	fmt.Println("  # Server features")
	fmt.Println("  cont-batching: true             # Enable continuous batching")
	fmt.Println("  metrics: false                  # Enable metrics endpoint")
	fmt.Println("  slots: true                     # Enable slots monitoring")
	fmt.Println("  slot-save-path: ./slots         # Path to save slots")
	fmt.Println("")
	fmt.Println("  # Authentication")
	fmt.Println("  api-key: secret                 # API key for authentication")
	fmt.Println("  ssl-key-file: key.pem           # SSL key file")
	fmt.Println("  ssl-cert-file: cert.pem         # SSL certificate file")
	fmt.Println("")
	fmt.Println("  # Logging")
	fmt.Println("  verbose: true                   # Enable verbose output")
	fmt.Println("  log-file: server.log            # Log to file")
	os.Exit(0)
}

func main() {
	// Check if help is requested
	if len(os.Args) > 1 && os.Args[1] == "--help" {
		showHelp()
	}

	// 1. Determine the configuration file path.
	// Priority: --config flag > LLAMA_CONFIG_PATH env var > default path.
	configFile := "./config.yaml" // Default path
	if val, ok := os.LookupEnv("LLAMA_CONFIG_PATH"); ok {
		configFile = val
	}
	if len(os.Args) > 2 && os.Args[1] == "--config" {
		configFile = os.Args[2]
	}
	log.Printf("Loading configuration from: %s", configFile)

	// 2. Read and parse the YAML configuration file.
	config, err := loadConfig(configFile)
	if err != nil {
		log.Printf("Failed to load configuration: %v", err)
		showHelp()
	}

	// 3. Build the command-line arguments for llama-server using reflection.
	args, err := buildArgs(config)
	if err != nil {
		log.Fatalf("Failed to build arguments: %v", err)
	}
	log.Printf("Starting llama-server with arguments: %v", args)

	// 4. Set up the command to execute llama-server.
	// Assumes 'llama-server' is in the PATH.
	cmd := exec.Command("llama-server", args...)

	// 5. Connect stdout and stderr of the child process to the parent.
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	// 6. Set up signal handling to forward signals to the child process.
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		sig := <-sigChan
		log.Printf("Received signal: %v. Forwarding to llama-server...", sig)
		if cmd.Process != nil {
			cmd.Process.Signal(sig)
		}
	}()

	// 7. Start and wait for the command to complete.
	err = cmd.Run()

	// 8. Handle the exit code.
	if err != nil {
		if exitError, ok := err.(*exec.ExitError); ok {
			waitStatus := exitError.Sys().(syscall.WaitStatus)
			log.Printf("llama-server exited with status: %d", waitStatus.ExitStatus())
			os.Exit(waitStatus.ExitStatus())
		} else {
			log.Fatalf("Failed to run llama-server: %v", err)
		}
	}

	log.Println("llama-server exited successfully.")
}

// loadConfig reads a YAML file and unmarshals it into a LlamaConfig struct.
func loadConfig(path string) (*LlamaConfig, error) {
	yamlFile, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("could not read yaml file: %w", err)
	}

	var config LlamaConfig
	err = yaml.Unmarshal(yamlFile, &config)
	if err != nil {
		return nil, fmt.Errorf("could not unmarshal yaml: %w", err)
	}

	return &config, nil
}

// buildArgs uses reflection to dynamically build command-line arguments
// from the LlamaConfig struct's `arg` tags.
func buildArgs(config *LlamaConfig) ([]string, error) {
	var args []string
	val := reflect.ValueOf(config).Elem() // Get the value of the struct
	typ := val.Type()                     // Get the type of the struct

	// Iterate over all the fields of the struct.
	for i := 0; i < val.NumField(); i++ {
		field := val.Field(i)
		fieldType := typ.Field(i)
		argTag := fieldType.Tag.Get("arg")

		// Skip fields that don't have an `arg` tag.
		if argTag == "" {
			continue
		}

		// Skip fields with zero values (e.g., empty strings, 0, false),
		// so we don't pass empty flags to the server.
		if field.IsZero() {
			continue
		}

		// Handle different field types.
		switch field.Kind() {
		case reflect.Bool:
			// For booleans, if true, just add the flag.
			if field.Bool() {
				args = append(args, argTag)
			}
		case reflect.String:
			args = append(args, argTag, field.String())
		case reflect.Int:
			args = append(args, argTag, strconv.FormatInt(field.Int(), 10))
		case reflect.Slice:
			// For slices (like --lora), add the flag for each item.
			if field.Type().Elem().Kind() == reflect.String {
				for j := 0; j < field.Len(); j++ {
					args = append(args, argTag, field.Index(j).String())
				}
			}
		default:
			return nil, fmt.Errorf("unsupported config field type: %s", field.Kind())
		}
	}

	return args, nil
}

