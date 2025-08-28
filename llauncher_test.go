package main

import (
	"os"
	"reflect"
	"testing"
)

// TestLoadConfig tests the loadConfig function with various inputs
func TestLoadConfig(t *testing.T) {
	// Create a temporary valid YAML file
	validYaml := `
model: /path/to/model.gguf
host: 0.0.0.0
port: 8080
threads: 4
n-gpu-layers: 0
lora:
  - adapter1.bin
  - adapter2.bin
log-verbose: true
ctx-size: 2048
`
	validFile := createTempFile(t, validYaml)
	defer os.Remove(validFile)

	// Create a temporary invalid YAML file
	invalidYaml := `
model: /path/to/model.gguf
host: 0.0.0.0
port: not-a-number  # This should cause an error
`
	invalidFile := createTempFile(t, invalidYaml)
	defer os.Remove(invalidFile)

	// Create a non-existent file path
	nonExistentFile := "/tmp/non-existent-file-" + randomString(8) + ".yaml"

	tests := []struct {
		name     string
		path     string
		wantErr  bool
		validate func(*LlamaConfig) bool
	}{
		{
			name:    "Valid YAML file",
			path:    validFile,
			wantErr: false,
			validate: func(c *LlamaConfig) bool {
				return c.ModelPath == "/path/to/model.gguf" &&
					c.Host == "0.0.0.0" &&
					c.Port == 8080 &&
					c.Threads == 4 &&
					c.GpuLayers == 0 &&
					len(c.LoraAdapters) == 2 &&
					c.LoraAdapters[0] == "adapter1.bin" &&
					c.LoraAdapters[1] == "adapter2.bin" &&
					c.Verbose == true
			},
		},
		{
			name:    "Invalid YAML file",
			path:    invalidFile,
			wantErr: true,
			validate: func(c *LlamaConfig) bool {
				return true // Not used when wantErr is true
			},
		},
		{
			name:    "Non-existent file",
			path:    nonExistentFile,
			wantErr: true,
			validate: func(c *LlamaConfig) bool {
				return true // Not used when wantErr is true
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config, err := loadConfig(tt.path)
			if (err != nil) != tt.wantErr {
				t.Errorf("loadConfig() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && !tt.validate(config) {
				t.Errorf("loadConfig() returned incorrect config: %+v", config)
			}
		})
	}
}

// TestBuildArgs tests the buildArgs function
func TestBuildArgs(t *testing.T) {
	tests := []struct {
		name    string
		config  *LlamaConfig
		want    []string
		wantErr bool
	}{
		{
			name: "Basic config",
			config: &LlamaConfig{
				ModelPath:    "/path/to/model.gguf",
				Host:         "0.0.0.0",
				Port:         8080,
				Threads:      4,
				GpuLayers:    0,
				LoraAdapters: []string{"adapter1.bin", "adapter2.bin"},
				Verbose:      true,
			},
			want: []string{
				"--host", "0.0.0.0",
				"--port", "8080",
				"--model", "/path/to/model.gguf",
				"--threads", "4",
				"--lora", "adapter1.bin",
				"--lora", "adapter2.bin",
				"--log-verbose",
			},
			wantErr: false,
		},
		{
			name:   "Empty config",
			config: &LlamaConfig{
				// All fields are zero values
			},
			want:    nil,
			wantErr: false,
		},
		{
			name: "Config with boolean flags",
			config: &LlamaConfig{
				ModelPath:      "/path/to/model.gguf",
				Verbose:        true,
				Mlock:          true,
				NoMMap:         false, // Should be skipped
				ContBatching:   true,
				NoContBatching: false, // Should be skipped
			},
			want: []string{
				"--model", "/path/to/model.gguf",
				"--mlock",
				"--cont-batching",
				"--log-verbose",
			},
			wantErr: false,
		},
		{
			name: "Config with numeric values",
			config: &LlamaConfig{
				ModelPath:    "/path/to/model.gguf",
				Threads:      4,
				ThreadsBatch: 8,
				ContextSize:  2048,
				TopK:         40,
				TopP:         0.95,
			},
			want: []string{
				"--model", "/path/to/model.gguf",
				"--threads", "4",
				"--threads-batch", "8",
				"--ctx-size", "2048",
				"--top-k", "40",
				"--top-p", "0.95",
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := buildArgs(tt.config)
			if (err != nil) != tt.wantErr {
				t.Errorf("buildArgs() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("buildArgs() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestFormatArgsForDisplay tests the formatArgsForDisplay function
func TestFormatArgsForDisplay(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want string
	}{
		{
			name: "Simple args",
			args: []string{"--model", "/path/to/model.gguf", "--threads", "4"},
			want: "--model /path/to/model.gguf \\\n    --threads 4",
		},
		{
			name: "Args with spaces",
			args: []string{"--model", "/path with spaces/model.gguf", "--host", "localhost"},
			want: "--model \"/path with spaces/model.gguf\" \\\n    --host localhost",
		},
		{
			name: "Multiple flags",
			args: []string{
				"--model", "/path/to/model.gguf",
				"--host", "0.0.0.0",
				"--port", "8080",
				"--threads", "4",
			},
			want: "--model /path/to/model.gguf \\\n    --host 0.0.0.0 \\\n    --port 8080 \\\n    --threads 4",
		},
		{
			name: "Empty args",
			args: []string{},
			want: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := formatArgsForDisplay(tt.args)
			if got != tt.want {
				t.Errorf("formatArgsForDisplay() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestContainsSpace tests the containsSpace function
func TestContainsSpace(t *testing.T) {
	tests := []struct {
		name string
		s    string
		want bool
	}{
		{
			name: "String with spaces",
			s:    "hello world",
			want: true,
		},
		{
			name: "String with tabs",
			s:    "hello\tworld",
			want: true,
		},
		{
			name: "String with newlines",
			s:    "hello\nworld",
			want: true,
		},
		{
			name: "String without spaces",
			s:    "hello_world",
			want: false,
		},
		{
			name: "Empty string",
			s:    "",
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := containsSpace(tt.s)
			if got != tt.want {
				t.Errorf("containsSpace() = %v, want %v", got, tt.want)
			}
		})
	}
}

// Helper function to create a temporary file with content
func createTempFile(t *testing.T, content string) string {
	t.Helper()
	tmpfile, err := os.CreateTemp("", "llauncher-test-*.yaml")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}

	if _, err := tmpfile.Write([]byte(content)); err != nil {
		t.Fatalf("Failed to write to temp file: %v", err)
	}

	if err := tmpfile.Close(); err != nil {
		t.Fatalf("Failed to close temp file: %v", err)
	}

	return tmpfile.Name()
}

// Helper function to generate a random string
func randomString(n int) string {
	const letters = "abcdefghijklmnopqrstuvwxyz0123456789"
	b := make([]byte, n)
	for i := range b {
		b[i] = letters[i%len(letters)]
	}
	return string(b)
}

// TestMain tests the main function with various command line arguments
func TestMainHelp(t *testing.T) {
	// Save original args and restore them after the test
	oldArgs := os.Args
	defer func() { os.Args = oldArgs }()

	// Test --help flag
	os.Args = []string{"llauncher", "--help"}

	// Since main() calls os.Exit(), we need to run it in a subprocess.
	// The subprocess will set TEST_MAIN_HELP=1 and invoke main().
	cmd := exec.Command(os.Args[0], "-test.run=TestMainHelp")
	cmd.Env = append(os.Environ(), "TEST_MAIN_HELP=1")
	// Suppress output; we only care about the exit code.
	cmd.Stdout = nil
	cmd.Stderr = nil

	if err := cmd.Run(); err != nil {
		// The subprocess should exit with code 0 (showHelp calls os.Exit(0)).
		if exitErr, ok := err.(*exec.ExitError); ok {
			if status := exitErr.ExitCode(); status != 0 {
				t.Fatalf("showHelp exited with %d, want 0", status)
			}
		} else {
			t.Fatalf("unexpected error running subprocess: %v", err)
		}
	}
}

// TestConfigFileHandling tests how the program handles different config file scenarios
func TestConfigFileHandling(t *testing.T) {
	// Create a valid config file
	validConfig := `
model: /path/to/model.gguf
host: 0.0.0.0
port: 8080
`
	validFile := createTempFile(t, validConfig)
	defer os.Remove(validFile)

	// Helper to run main in a subprocess with given args and env.
	runMain := func(args []string, env []string) error {
		cmd := exec.Command(os.Args[0], "-test.run=TestConfigFileHandling")
		cmd.Env = append(os.Environ(), env...)
		cmd.Args = append([]string{os.Args[0]}, args...)
		// Suppress output; we only care about exit code.
		cmd.Stdout = nil
		cmd.Stderr = nil
		return cmd.Run()
	}

	// Test with environment variable
	t.Run("Environment variable config path", func(t *testing.T) {
		oldEnv, exists := os.LookupEnv("LLAMA_CONFIG_PATH")
		if exists {
			defer os.Setenv("LLAMA_CONFIG_PATH", oldEnv)
		} else {
			defer os.Unsetenv("LLAMA_CONFIG_PATH")
		}
		os.Setenv("LLAMA_CONFIG_PATH", validFile)

		if err := runMain([]string{"llauncher"}, []string{"TEST_CONFIG_ENV=1"}); err != nil {
			if exitErr, ok := err.(*exec.ExitError); ok && exitErr.ExitCode() != 0 {
				t.Fatalf("main exited with %d, want 0 (env config)", exitErr.ExitCode())
			}
			t.Fatalf("error running main with env config: %v", err)
		}
	})

	// Test with --config flag (should override env var)
	t.Run("Config flag overrides env", func(t *testing.T) {
		// Set a bogus env var to ensure flag takes precedence
		os.Setenv("LLAMA_CONFIG_PATH", "/nonexistent.yaml")
		if err := runMain([]string{"llauncher", "--config", validFile}, []string{"TEST_CONFIG_FLAG=1"}); err != nil {
			if exitErr, ok := err.(*exec.ExitError); ok && exitErr.ExitCode() != 0 {
				t.Fatalf("main exited with %d, want 0 (config flag)", exitErr.ExitCode())
			}
			t.Fatalf("error running main with config flag: %v", err)
		}
	})
}
