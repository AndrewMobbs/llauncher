package main

import (
	"bytes"
	"io"
	"os"
	"strings"
	"testing"
	"time"
	"syscall"
	"os/exec"
	"os/signal"
)

// TestDebugMode tests the --debug flag functionality
func TestDebugMode(t *testing.T) {
	// Save original args and restore them after the test
	oldArgs := os.Args
	defer func() { os.Args = oldArgs }()

	// Create a valid config file
	validConfig := `
model: /path/to/model.gguf
host: 0.0.0.0
port: 8080
`
	validFile := createTempFile(t, validConfig)
	defer os.Remove(validFile)

	// Test with --debug flag
	t.Run("Debug flag", func(t *testing.T) {
		// Set up args with debug flag
		os.Args = []string{"llauncher", "--debug", "--config", validFile}

		// Capture stdout to check debug output
		oldStdout := os.Stdout
		r, w, _ := os.Pipe()
		os.Stdout = w

		// Run main in a subprocess so os.Exit does not kill the test.
		cmd := exec.Command("llauncher", "--debug", "--config", validFile)
		// Suppress output; we will read from the pipe.
		cmd.Stdout = w
		cmd.Stderr = w

		if err := cmd.Run(); err != nil {
			if exitErr, ok := err.(*exec.ExitError); ok && exitErr.ExitCode() != 0 {
				t.Fatalf("main exited with %d, want 0 (debug mode)", exitErr.ExitCode())
			}
			t.Fatalf("error running main in debug mode: %v", err)
		}

		// Restore stdout
		os.Stdout = oldStdout
		w.Close()

		// Read captured output
		var buf bytes.Buffer
		io.Copy(&buf, r)

		output := buf.String()
		if !strings.Contains(output, "DEBUG:") {
			t.Fatalf("expected debug output, got: %s", output)
		}
	})
}

// TestSignalHandling tests the signal handling functionality
func TestSignalHandling(t *testing.T) {
	// Helper to capture and discard stdout/stderr.
	captureOutput := func(f func()) {
		devNull, _ := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
		origStdout := os.Stdout
		origStderr := os.Stderr
		os.Stdout = devNull
		os.Stderr = devNull
		defer func() {
			os.Stdout = origStdout
			os.Stderr = origStderr
			_ = devNull.Close()
		}()
		f()
	}

	captureOutput(func() {
		// This test verifies that signals are forwarded to the child process.
		// We use a mock command that blocks until it receives a SIGTERM.
		// The mock is provided via the execCommand variable.

		// Create a temporary config file (minimal, just model)
		cfg := `
model: /tmp/dummy.gguf
`
		cfgFile := createTempFile(t, cfg)
		defer os.Remove(cfgFile)

		// Prepare a mock command that waits for a signal.
		// The mock will be a separate Go test binary that exits with code 0
		// when it receives SIGTERM.
		mockCmd := func(name string, args ...string) *exec.Cmd {
			// Use the same test binary with a special env var.
			cmd := exec.Command(os.Args[0], "-test.run=MockSignalReceiver")
			cmd.Env = append(os.Environ(), "MOCK_SIGNAL=1")
			return cmd
		}
		originalExec := execCommand
		execCommand = mockCmd
		defer func() { execCommand = originalExec }()

		// Run main in a goroutine so we can send it a signal.
		done := make(chan struct{})
		go func() {
			// Set args to include config file.
			os.Args = []string{"llauncher", "--config", cfgFile}
			main()
			close(done)
		}()

		// Give the child process a moment to start.
		time.Sleep(100 * time.Millisecond)

		// Send SIGTERM to the current process; it should be forwarded.
		syscall.Kill(syscall.Getpid(), syscall.SIGTERM)

		// Wait for main to exit.
		select {
		case <-done:
		case <-time.After(2 * time.Second):
			t.Fatalf("main did not exit after signal")
		}
	})
}

// MockSignalReceiver is executed in a subprocess to act as the child process.
func TestMockSignalReceiver(t *testing.T) {
	if os.Getenv("MOCK_SIGNAL") != "1" {
		return
	}
	// Wait for a signal.
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGTERM)
	<-sigChan
	// Received signal, exit with success.
	os.Exit(0)
}
