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
	ModelPath      string   `yaml:"model" arg:"--model"`
	Host           string   `yaml:"host" arg:"--host"`
	Port           int      `yaml:"port" arg:"--port"`
	Threads        int      `yaml:"threads" arg:"--threads"`
	ThreadsBatch   int      `yaml:"threads-batch" arg:"--threads-batch"`
	ContextSize    int      `yaml:"n-ctx" arg:"--n-ctx"`
	GpuLayers      int      `yaml:"n-gpu-layers" arg:"--n-gpu-layers"`
	LoraAdapters   []string `yaml:"lora" arg:"--lora"`
	Verbose        bool     `yaml:"verbose" arg:"--verbose"`
	ContBatching   bool     `yaml:"cont-batching" arg:"--cont-batching"`
	Metrics        bool     `yaml:"metrics" arg:"--metrics"`
	SlotSavePath   string   `yaml:"slot-save-path" arg:"--slot-save-path"`
	Numa           string   `yaml:"numa" arg:"--numa"`
}

func main() {
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
		log.Fatalf("Failed to load configuration: %v", err)
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

