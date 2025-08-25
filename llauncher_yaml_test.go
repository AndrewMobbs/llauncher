package main

import (
	"os"
	"testing"
)

// TestYAMLEdgeCases tests various edge cases in YAML parsing
func TestYAMLEdgeCases(t *testing.T) {
	tests := []struct {
		name     string
		yaml     string
		wantErr  bool
		validate func(*LlamaConfig) bool
	}{
		{
			name: "Empty YAML",
			yaml: "",
			wantErr: false,
			validate: func(c *LlamaConfig) bool {
				// An empty YAML should result in a config with all default values
				return c.ModelPath == "" && c.Host == "" && c.Port == 0
			},
		},
		{
			name: "YAML with comments",
			yaml: `
# This is a comment
model: /path/to/model.gguf  # This is an inline comment
# Another comment
host: 0.0.0.0
`,
			wantErr: false,
			validate: func(c *LlamaConfig) bool {
				return c.ModelPath == "/path/to/model.gguf" && c.Host == "0.0.0.0"
			},
		},
		{
			name: "YAML with nested structures",
			yaml: `
model: /path/to/model.gguf
advanced:
  option1: value1
  option2: value2
lora:
  - adapter1.bin
  - adapter2.bin
`,
			wantErr: false,
			validate: func(c *LlamaConfig) bool {
				return c.ModelPath == "/path/to/model.gguf" && 
					len(c.LoraAdapters) == 2 &&
					c.LoraAdapters[0] == "adapter1.bin" &&
					c.LoraAdapters[1] == "adapter2.bin"
			},
		},
		{
			name: "YAML with type mismatches",
			yaml: `
model: /path/to/model.gguf
threads: "four"  # Should be a number
`,
			wantErr: true,
			validate: func(c *LlamaConfig) bool {
				return true // Not used when wantErr is true
			},
		},
		{
			name: "YAML with boolean values",
			yaml: `
model: /path/to/model.gguf
verbose: true
mlock: yes  # Alternative way to specify true
no-mmap: false
metrics: no  # Alternative way to specify false
`,
			wantErr: false,
			validate: func(c *LlamaConfig) bool {
				return c.ModelPath == "/path/to/model.gguf" && 
					c.Verbose == true && 
					c.Mlock == true && 
					c.NoMMap == false && 
					c.Metrics == false
			},
		},
		{
			name: "YAML with floating point values",
			yaml: `
model: /path/to/model.gguf
temp: 0.8
top-p: 0.9
rope-scale: 1.5
`,
			wantErr: false,
			validate: func(c *LlamaConfig) bool {
				return c.ModelPath == "/path/to/model.gguf" && 
					c.Temperature == 0.8 && 
					c.TopP == 0.9 && 
					c.RopeScale == 1.5
			},
		},
		{
			name: "YAML with invalid indentation",
			yaml: `
model: /path/to/model.gguf
lora:
- adapter1.bin
    - invalid: structure  # This creates invalid YAML structure
      that: will fail
`,
			wantErr: true,
			validate: func(c *LlamaConfig) bool {
				return true // Not used when wantErr is true
			},
		},
		{
			name: "YAML with special characters",
			yaml: `
model: "/path/with \"quotes\"/model.gguf"
host: "server-name:with:colons"
path: "/path/with/special/chars/!@#$%^&*()"
`,
			wantErr: false,
			validate: func(c *LlamaConfig) bool {
				return c.ModelPath == `/path/with "quotes"/model.gguf` && 
					c.Host == "server-name:with:colons" && 
					c.Path == "/path/with/special/chars/!@#$%^&*()"
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a temporary file with the test YAML content
			tmpfile := createTempFile(t, tt.yaml)
			defer os.Remove(tmpfile)
			
			// Load the config from the temporary file
			config, err := loadConfig(tmpfile)
			if (err != nil) != tt.wantErr {
				t.Errorf("loadConfig() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			
			// If we don't expect an error, validate the config
			if !tt.wantErr && !tt.validate(config) {
				t.Errorf("loadConfig() returned incorrect config: %+v", config)
			}
		})
	}
}

// TestYAMLFieldMapping tests that YAML fields map correctly to struct fields
func TestYAMLFieldMapping(t *testing.T) {
	yaml := `
# Basic server configuration
model: /path/to/model.gguf
model-url: https://example.com/model.gguf
host: 0.0.0.0
port: 8080
path: /static
api-prefix: /api
no-webui: true
timeout: 600
threads-http: 8

# Performance configuration
threads: 4
threads-batch: 8
n-ctx: 2048
batch-size: 512
n-gpu-layers: 0
device: cpu
numa: none

# Memory management
mlock: true
cache-type-k: f16
cache-type-v: f16
cache-reuse: 10

# Sampling parameters
seed: 42
temp: 0.8
top-k: 40
top-p: 0.9

# Server features
cont-batching: true
metrics: true
slots: true
slot-save-path: ./slots
`

	tmpfile := createTempFile(t, yaml)
	defer os.Remove(tmpfile)
	
	config, err := loadConfig(tmpfile)
	if err != nil {
		t.Fatalf("loadConfig() error = %v", err)
	}
	
	// Check that all fields were mapped correctly
	if config.ModelPath != "/path/to/model.gguf" {
		t.Errorf("ModelPath = %v, want %v", config.ModelPath, "/path/to/model.gguf")
	}
	if config.ModelUrl != "https://example.com/model.gguf" {
		t.Errorf("ModelUrl = %v, want %v", config.ModelUrl, "https://example.com/model.gguf")
	}
	if config.Host != "0.0.0.0" {
		t.Errorf("Host = %v, want %v", config.Host, "0.0.0.0")
	}
	if config.Port != 8080 {
		t.Errorf("Port = %v, want %v", config.Port, 8080)
	}
	if config.Path != "/static" {
		t.Errorf("Path = %v, want %v", config.Path, "/static")
	}
	if config.ApiPrefix != "/api" {
		t.Errorf("ApiPrefix = %v, want %v", config.ApiPrefix, "/api")
	}
	if !config.NoWebUI {
		t.Errorf("NoWebUI = %v, want %v", config.NoWebUI, true)
	}
	if config.Timeout != 600 {
		t.Errorf("Timeout = %v, want %v", config.Timeout, 600)
	}
	if config.ThreadsHTTP != 8 {
		t.Errorf("ThreadsHTTP = %v, want %v", config.ThreadsHTTP, 8)
	}
	if config.Threads != 4 {
		t.Errorf("Threads = %v, want %v", config.Threads, 4)
	}
	if config.ThreadsBatch != 8 {
		t.Errorf("ThreadsBatch = %v, want %v", config.ThreadsBatch, 8)
	}
	if config.ContextSize != 2048 {
		t.Errorf("ContextSize = %v, want %v", config.ContextSize, 2048)
	}
	if config.BatchSize != 512 {
		t.Errorf("BatchSize = %v, want %v", config.BatchSize, 512)
	}
	if config.GpuLayers != 0 {
		t.Errorf("GpuLayers = %v, want %v", config.GpuLayers, 0)
	}
	if config.Device != "cpu" {
		t.Errorf("Device = %v, want %v", config.Device, "cpu")
	}
	if config.Numa != "none" {
		t.Errorf("Numa = %v, want %v", config.Numa, "none")
	}
	if !config.Mlock {
		t.Errorf("Mlock = %v, want %v", config.Mlock, true)
	}
	if config.CacheTypeK != "f16" {
		t.Errorf("CacheTypeK = %v, want %v", config.CacheTypeK, "f16")
	}
	if config.CacheTypeV != "f16" {
		t.Errorf("CacheTypeV = %v, want %v", config.CacheTypeV, "f16")
	}
	if config.CacheReuse != 10 {
		t.Errorf("CacheReuse = %v, want %v", config.CacheReuse, 10)
	}
	if config.Seed != 42 {
		t.Errorf("Seed = %v, want %v", config.Seed, 42)
	}
	if config.Temperature != 0.8 {
		t.Errorf("Temperature = %v, want %v", config.Temperature, 0.8)
	}
	if config.TopK != 40 {
		t.Errorf("TopK = %v, want %v", config.TopK, 40)
	}
	if config.TopP != 0.9 {
		t.Errorf("TopP = %v, want %v", config.TopP, 0.9)
	}
	if !config.ContBatching {
		t.Errorf("ContBatching = %v, want %v", config.ContBatching, true)
	}
	if !config.Metrics {
		t.Errorf("Metrics = %v, want %v", config.Metrics, true)
	}
	if !config.Slots {
		t.Errorf("Slots = %v, want %v", config.Slots, true)
	}
	if config.SlotSavePath != "./slots" {
		t.Errorf("SlotSavePath = %v, want %v", config.SlotSavePath, "./slots")
	}
}
