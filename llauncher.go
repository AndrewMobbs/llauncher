package main

import (
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"reflect"
	"strconv"
	"strings"
	"syscall"
	"unicode"

	"gopkg.in/yaml.v3"
)

// execCommand is a wrapper around exec.Command that can be swapped out in tests.
// By default it points at the real exec.Command implementation.
var execCommand = exec.Command

// LlamaConfig defines the structure of the YAML configuration file.
// The `yaml` struct tags map YAML keys to the struct fields.
// The `arg` struct tags define the corresponding command-line flag, which is
// now the single source of truth for argument generation.
type LlamaConfig struct {
	// Basic server configuration
	Host        string `yaml:"host" arg:"--host"`
	Port        int    `yaml:"port" arg:"--port"`
	Path        string `yaml:"path" arg:"--path"`
	ApiPrefix   string `yaml:"api-prefix" arg:"--api-prefix"`
	NoWebUI     bool   `yaml:"no-webui" arg:"--no-webui"`
	Timeout     int    `yaml:"timeout" arg:"--timeout"`
	ThreadsHTTP int    `yaml:"threads-http" arg:"--threads-http"`
	Props       bool   `yaml:"props" arg:"--props"`

	// Model loading and configuration
	ModelPath   string `yaml:"model" arg:"--model"`
	ModelUrl    string `yaml:"model-url" arg:"--model-url"`
	HfRepo      string `yaml:"hf-repo" arg:"--hf-repo"`
	HfRepoDraft string `yaml:"hf-repo-draft" arg:"--hf-repo-draft"`
	HfRepoV     string `yaml:"hf-repo-v" arg:"--hf-repo-v"`
	HfFile      string `yaml:"hf-file" arg:"--hf-file"`
	HfFileV     string `yaml:"hf-file-v" arg:"--hf-file-v"`
	HfToken     string `yaml:"hf-token" arg:"--hf-token"`
	Offline     bool   `yaml:"offline" arg:"--offline"`
	Alias       string `yaml:"alias" arg:"--alias"`

	// Performance and resource configuration
	Threads        int    `yaml:"threads" arg:"--threads"`
	ThreadsBatch   int    `yaml:"threads-batch" arg:"--threads-batch"`
	CpuMask        string `yaml:"cpu-mask" arg:"--cpu-mask"`
	CpuMaskBatch   string `yaml:"cpu-mask-batch" arg:"--cpu-mask-batch"`
	CpuRange       string `yaml:"cpu-range" arg:"--cpu-range"`
	CpuRangeBatch  string `yaml:"cpu-range-batch" arg:"--cpu-range-batch"`
	CpuStrict      int    `yaml:"cpu-strict" arg:"--cpu-strict"`
	CpuStrictBatch int    `yaml:"cpu-strict-batch" arg:"--cpu-strict-batch"`
	Priority       int    `yaml:"prio" arg:"--prio"`
	PriorityBatch  int    `yaml:"prio-batch" arg:"--prio-batch"`
	Poll           int    `yaml:"poll" arg:"--poll"`
	PollBatch      int    `yaml:"poll-batch" arg:"--poll-batch"`
	BatchSize      int    `yaml:"batch-size" arg:"--batch-size"`
	UBatchSize     int    `yaml:"ubatch-size" arg:"--ubatch-size"`
	GpuLayers      int    `yaml:"n-gpu-layers" arg:"--n-gpu-layers"`
	SplitMode      string `yaml:"split-mode" arg:"--split-mode"`
	TensorSplit    string `yaml:"tensor-split" arg:"--tensor-split"`
	MainGPU        int    `yaml:"main-gpu" arg:"--main-gpu"`
	Numa           string `yaml:"numa" arg:"--numa"`
	Device         string `yaml:"device" arg:"--device"`
	NoPerf         bool   `yaml:"no-perf" arg:"--no-perf"`
	Parallel       int    `yaml:"parallel" arg:"--parallel"`

	// Memory management
	ContextSize     int    `yaml:"ctx-size" arg:"--ctx-size"`
	FlashAttn       bool   `yaml:"flash-attn" arg:"--flash-attn"`
	Mlock           bool   `yaml:"mlock" arg:"--mlock"`
	NoMMap          bool   `yaml:"no-mmap" arg:"--no-mmap"`
	CacheTypeK      string `yaml:"cache-type-k" arg:"--cache-type-k"`
	CacheTypeKDraft string `yaml:"cache-type-k-draft" arg:"--cache-type-k-draft"`
	CacheTypeV      string `yaml:"cache-type-v" arg:"--cache-type-v"`
	CacheTypeVDraft string `yaml:"cache-type-v-draft" arg:"--cache-type-v-draft"`
	CacheReuse      int    `yaml:"cache-reuse" arg:"--cache-reuse"`
	SwaFull         bool   `yaml:"swa-full" arg:"--swa-full"`
	KvUnified       bool   `yaml:"kv-unified" arg:"--kv-unified"`
	NoKvOffload     bool   `yaml:"no-kv-offload" arg:"--no-kv-offload"`
	NoRepack        bool   `yaml:"no-repack" arg:"--no-repack"`
	NoOpOffload     bool   `yaml:"no-op-offload" arg:"--no-op-offload"`

	// RoPE configuration
	RopeScaling   string  `yaml:"rope-scaling" arg:"--rope-scaling"`
	RopeScale     float64 `yaml:"rope-scale" arg:"--rope-scale"`
	RopeFreqBase  float64 `yaml:"rope-freq-base" arg:"--rope-freq-base"`
	RopeFreqScale float64 `yaml:"rope-freq-scale" arg:"--rope-freq-scale"`

	// YaRN configuration
	YarnOrigCtx    int     `yaml:"yarn-orig-ctx" arg:"--yarn-orig-ctx"`
	YarnExtFactor  float64 `yaml:"yarn-ext-factor" arg:"--yarn-ext-factor"`
	YarnAttnFactor float64 `yaml:"yarn-attn-factor" arg:"--yarn-attn-factor"`
	YarnBetaSlow   float64 `yaml:"yarn-beta-slow" arg:"--yarn-beta-slow"`
	YarnBetaFast   float64 `yaml:"yarn-beta-fast" arg:"--yarn-beta-fast"`

	// MoE offload and Tensor Override
	CpuMoe              bool   `yaml:"cpu-moe" arg:"--cpu-moe"`
	NCpuMoe             string `yaml:"n-cpu-moe" arg:"--n-cpu-moe"`
	OverrideTensor      string `yaml:"override-tensor" arg:"--override-tensor"`
	OverrideTensorDraft string `yaml:"override-tensor-draft" arg:"--override-tensor-draft"`
	OverrideKv          string `yaml:"override-kv" arg:"--override-kv"`

	// Sampling parameters
	Seed               int     `yaml:"seed" arg:"--seed"`
	Samplers           string  `yaml:"samplers" arg:"--samplers"`
	SamplerSeq         string  `yaml:"sampler-seq" arg:"--sampler-seq"`
	IgnoreEOS          bool    `yaml:"ignore-eos" arg:"--ignore-eos"`
	Temperature        float64 `yaml:"temp" arg:"--temp"`
	TopK               int     `yaml:"top-k" arg:"--top-k"`
	TopP               float64 `yaml:"top-p" arg:"--top-p"`
	MinP               float64 `yaml:"min-p" arg:"--min-p"`
	TopNSigma          float64 `yaml:"top-nsigma" arg:"--top-nsigma"`
	Typical            float64 `yaml:"typical" arg:"--typical"`
	RepeatLastN        int     `yaml:"repeat-last-n" arg:"--repeat-last-n"`
	RepeatPenalty      float64 `yaml:"repeat-penalty" arg:"--repeat-penalty"`
	PresencePenalty    float64 `yaml:"presence-penalty" arg:"--presence-penalty"`
	FrequencyPenalty   float64 `yaml:"frequency-penalty" arg:"--frequency-penalty"`
	Mirostat           int     `yaml:"mirostat" arg:"--mirostat"`
	MirostatLR         float64 `yaml:"mirostat-lr" arg:"--mirostat-lr"`
	MirostatEnt        float64 `yaml:"mirostat-ent" arg:"--mirostat-ent"`
	XtcProbability     float64 `yaml:"xtc-probability" arg:"--xtc-probability"`
	XtcThreshold       float64 `yaml:"xtc-threshold" arg:"--xtc-threshold"`
	DryMultiplier      float64 `yaml:"dry-multiplier" arg:"--dry-multiplier"`
	DryBase            float64 `yaml:"dry-base" arg:"--dry-base"`
	DryAllowedLen      int     `yaml:"dry-allowed-length" arg:"--dry-allowed-length"`
	DryPenaltyLastN    int     `yaml:"dry-penalty-last-n" arg:"--dry-penalty-last-n"`
	DrySequenceBreaker string  `yaml:"dry-sequence-breaker" arg:"--dry-sequence-breaker"`
	DynaTempRange      float64 `yaml:"dynatemp-range" arg:"--dynatemp-range"`
	DynaTempExp        float64 `yaml:"dynatemp-exp" arg:"--dynatemp-exp"`

	// Grammar and constraints
	Grammar        string `yaml:"grammar" arg:"--grammar"`
	GrammarFile    string `yaml:"grammar-file" arg:"--grammar-file"`
	JsonSchema     string `yaml:"json-schema" arg:"--json-schema"`
	JsonSchemaFile string `yaml:"json-schema-file" arg:"--json-schema-file"`
	Escape         bool   `yaml:"escape" arg:"--escape"`
	NoEscape       bool   `yaml:"no-escape" arg:"--no-escape"`
	SpmInfill      bool   `yaml:"spm-infill" arg:"--spm-infill"`

	// Adapters and extensions
	LoraAdapters            []string `yaml:"lora" arg:"--lora"`
	LoraScaled              []string `yaml:"lora-scaled" arg:"--lora-scaled"`
	LoraInitWithoutApply    bool     `yaml:"lora-init-without-apply" arg:"--lora-init-without-apply"`
	ControlVector           []string `yaml:"control-vector" arg:"--control-vector"`
	ControlVectorScaled     []string `yaml:"control-vector-scaled" arg:"--control-vector-scaled"`
	ControlVectorLayerRange string   `yaml:"control-vector-layer-range" arg:"--control-vector-layer-range"`
	MmProj                  string   `yaml:"mmproj" arg:"--mmproj"`
	MmProjUrl               string   `yaml:"mmproj-url" arg:"--mmproj-url"`
	NoMmProj                bool     `yaml:"no-mmproj" arg:"--no-mmproj"`
	NoMmProjOffload         bool     `yaml:"no-mmproj-offload" arg:"--no-mmproj-offload"`

	// Server features
	ContBatching         bool    `yaml:"cont-batching" arg:"--cont-batching"`
	NoContBatching       bool    `yaml:"no-cont-batching" arg:"--no-cont-batching"`
	Metrics              bool    `yaml:"metrics" arg:"--metrics"`
	Slots                bool    `yaml:"slots" arg:"--slots"`
	NoSlots              bool    `yaml:"no-slots" arg:"--no-slots"`
	SlotSavePath         string  `yaml:"slot-save-path" arg:"--slot-save-path"`
	SlotPromptSimilarity float64 `yaml:"slot-prompt-similarity" arg:"--slot-prompt-similarity"`
	SwaCheckpoints       int     `yaml:"swa-checkpoints" arg:"--swa-checkpoints"`

	// Authentication and security
	ApiKey      string `yaml:"api-key" arg:"--api-key"`
	ApiKeyFile  string `yaml:"api-key-file" arg:"--api-key-file"`
	SslKeyFile  string `yaml:"ssl-key-file" arg:"--ssl-key-file"`
	SslCertFile string `yaml:"ssl-cert-file" arg:"--ssl-cert-file"`

	// Chat and template configuration
	ChatTemplate       string `yaml:"chat-template" arg:"--chat-template"`
	ChatTemplateFile   string `yaml:"chat-template-file" arg:"--chat-template-file"`
	ChatTemplateKwargs string `yaml:"chat-template-kwargs" arg:"--chat-template-kwargs"`
	Jinja              bool   `yaml:"jinja" arg:"--jinja"`
	NoPrefillAssistant bool   `yaml:"no-prefill-assistant" arg:"--no-prefill-assistant"`
	ReasoningFormat    string `yaml:"reasoning-format" arg:"--reasoning-format"`
	ReasoningBudget    int    `yaml:"reasoning-budget" arg:"--reasoning-budget"`

	// Special use cases
	Embeddings        bool   `yaml:"embeddings" arg:"--embeddings"`
	Reranking         bool   `yaml:"reranking" arg:"--reranking"`
	Pooling           string `yaml:"pooling" arg:"--pooling"`
	CheckTensors      bool   `yaml:"check-tensors" arg:"--check-tensors"`
	LogitBias         string `yaml:"logit-bias" arg:"--logit-bias"`
	ModelVocoder      string `yaml:"model-vocoder" arg:"--model-vocoder"`
	TTSUseGuideTokens bool   `yaml:"tts-use-guide-tokens" arg:"--tts-use-guide-tokens"`

	// Logging
	Verbose       bool   `yaml:"log-verbose" arg:"--log-verbose"`
	LogDisable    bool   `yaml:"log-disable" arg:"--log-disable"`
	LogFile       string `yaml:"log-file" arg:"--log-file"`
	LogColors     bool   `yaml:"log-colors" arg:"--log-colors"`
	LogVerbosity  int    `yaml:"log-verbosity" arg:"--log-verbosity"`
	LogPrefix     bool   `yaml:"log-prefix" arg:"--log-prefix"`
	LogTimestamps bool   `yaml:"log-timestamps" arg:"--log-timestamps"`

	// Prediction and generation
	Predict        int    `yaml:"n-predict" arg:"--n-predict"`
	ReversePrompt  string `yaml:"reverse-prompt" arg:"--reverse-prompt"`
	Special        bool   `yaml:"special" arg:"--special"`
	NoWarmup       bool   `yaml:"no-warmup" arg:"--no-warmup"`
	NoContextShift bool   `yaml:"no-context-shift" arg:"--no-context-shift"`
	ContextShift   bool   `yaml:"context-shift" arg:"--context-shift"`
	Keep           int    `yaml:"keep" arg:"--keep"`

	// Speculative decoding
	ModelDraft        string  `yaml:"model-draft" arg:"--model-draft"`
	ThreadsDraft      int     `yaml:"threads-draft" arg:"--threads-draft"`
	ThreadsBatchDraft int     `yaml:"threads-batch-draft" arg:"--threads-batch-draft"`
	ContextSizeDraft  int     `yaml:"ctx-size-draft" arg:"--ctx-size-draft"`
	DeviceDraft       string  `yaml:"device-draft" arg:"--device-draft"`
	GpuLayersDraft    int     `yaml:"n-gpu-layers-draft" arg:"--n-gpu-layers-draft"`
	DraftMax          int     `yaml:"draft-max" arg:"--draft-max"`
	DraftMin          int     `yaml:"draft-min" arg:"--draft-min"`
	DraftPMin         float64 `yaml:"draft-p-min" arg:"--draft-p-min"`
	SpecReplace       string  `yaml:"spec-replace" arg:"--spec-replace"`
}

// showHelp displays usage information for the launcher
func showHelp() {
	fmt.Println("llauncher - A launcher for llama-server")
	fmt.Println("\nUsage:")
	fmt.Println("  llauncher [--config <config_file>] [--help] [--debug]")
	fmt.Println("\nOptions:")
	fmt.Println("  --config <file>    Path to YAML configuration file")
	fmt.Println("  --help             Show this help message")
	fmt.Println("  --debug            Print debug information including the full command")
	fmt.Println("\nEnvironment Variables:")
	fmt.Println("  LLAMA_CONFIG_PATH  Path to YAML configuration file (overridden by --config)")
	os.Exit(0)
}

func main() {
	// Check if help is requested
	if len(os.Args) > 1 && os.Args[1] == "--help" {
		showHelp()
	}

	// Check if debug mode is enabled
	debugMode := false
	for _, arg := range os.Args {
		if arg == "--debug" {
			debugMode = true
			break
		}
	}

	// 1. Determine the configuration file path.
	// Priority: --config flag > XDG_CONFIG_HOME/llauncher/config.yaml > XDG_CONFIG_HOME/llauncher.yaml > HOME/.config/llauncher/config.yaml > HOME/.config/llauncher.yaml > LLAMA_CONFIG_PATH env var > default path.
	// Default path (fallback) remains "./config.yaml" for backward compatibility.
	const defaultPath = "./config.yaml"
	// Resolve XDG base directories.
	xdgConfigHome := os.Getenv("XDG_CONFIG_HOME")
	if xdgConfigHome == "" {
		homeDir, err := os.UserHomeDir()
		if err == nil && homeDir != "" {
			xdgConfigHome = homeDir + "/.config"
		}
	}
	// Build candidate paths.
	var candidatePaths []string
	if xdgConfigHome != "" {
		candidatePaths = append(candidatePaths, xdgConfigHome+"/llauncher/config.yaml")
		candidatePaths = append(candidatePaths, xdgConfigHome+"/llauncher.yaml")
	}
	// Environment variable override.
	if val, ok := os.LookupEnv("LLAMA_CONFIG_PATH"); ok && val != "" {
		candidatePaths = []string{val}
	}
	// Command‑line flag override.
	for i := 1; i < len(os.Args)-1; i++ {
		if os.Args[i] == "--config" {
			candidatePaths = []string{os.Args[i+1]}
			break
		}
	}
	// Determine the first existing file among candidates.
	configFile := defaultPath
	for _, p := range candidatePaths {
		if _, err := os.Stat(p); err == nil {
			configFile = p
			break
		}
	}
	if debugMode {
		fmt.Printf("Loading configuration from: %s\n", configFile)
	}

	// 2. Read and parse the YAML configuration file.
	config, err := loadConfig(configFile)
	if err != nil {
		if debugMode {
			fmt.Printf("Failed to load configuration: %v\n", err)
		}
		showHelp()
	}

	// 3. Build the command-line arguments for llama-server using reflection.
	args, err := buildArgs(config)
	if err != nil {
		if debugMode {
			fmt.Printf("Failed to build arguments: %v\n", err)
		}
		os.Exit(1)
	}

	// In debug mode, print the full command that will be executed
	if debugMode {
		fmt.Printf("DEBUG: Configuration file: %s\n", configFile)
		fmt.Println("DEBUG: Full command that will be executed:")
		fmt.Printf("DEBUG: llama-server %s\n", formatArgsForDisplay(args))
	}

	// 4. Set up the command to execute llama-server.
	// Assumes 'llama-server' is in the PATH.
	// Use the injectable execCommand so tests can replace it.
	cmd := execCommand("llama-server", args...)

	// 5. Connect stdout and stderr of the child process to the parent.
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	// 6. Set up signal handling to forward signals to the child process.
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		sig := <-sigChan
		if debugMode {
			fmt.Printf("Received signal: %v. Forwarding to llama-server...\n", sig)
		}
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
			if debugMode {
				fmt.Printf("llama-server exited with status: %d\n", waitStatus.ExitStatus())
			}
			os.Exit(waitStatus.ExitStatus())
		} else {
			if debugMode {
				fmt.Printf("Failed to run llama-server: %v\n", err)
			}
			os.Exit(1)
		}
	}

	if debugMode {
		fmt.Println("llama-server exited successfully.")
	}
}

// loadConfig reads a YAML file and unmarshals it into a LlamaConfig struct.
func loadConfig(path string) (*LlamaConfig, error) {
	yamlFile, err := os.ReadFile(path)
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
		case reflect.Float64:
			args = append(args, argTag, fmt.Sprintf("%g", field.Float()))
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

// formatArgsForDisplay formats the arguments array for better readability
func formatArgsForDisplay(args []string) string {
	var result string
	for i, arg := range args {
		// Add quotes around arguments that contain spaces
		if containsSpace(arg) {
			arg = "\"" + arg + "\""
		}

		// Add a newline and indentation for each flag (arguments starting with --)
		if i > 0 && strings.HasPrefix(arg, "--") {
			result += " \\\n    "
		} else if i > 0 {
			result += " "
		}

		result += arg
	}
	return result
}

// containsSpace checks if a string contains any whitespace
func containsSpace(s string) bool {
	for _, r := range s {
		if unicode.IsSpace(r) {
			return true
		}
	}
	return false
}
