package main

import (
	"os"
	"os/exec"
	"testing"
)

// mockExecCommand is used to mock the exec.Command function for testing
func mockExecCommand(command string, args ...string) *exec.Cmd {
	cs := []string{"-test.run=TestHelperProcess", "--", command}
	cs = append(cs, args...)
	cmd := exec.Command(os.Args[0], cs...)
	cmd.Env = []string{"GO_WANT_HELPER_PROCESS=1"}
	return cmd
}

// TestHelperProcess isn't a real test. It's used as a helper process for TestMain.
func TestHelperProcess(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "1" {
		return
	}
	
	// Get the command and arguments that were passed to exec.Command
	args := os.Args
	for len(args) > 0 {
		if args[0] == "--" {
			args = args[1:]
			break
		}
		args = args[1:]
	}
	
	if len(args) == 0 {
		// No command was provided
		os.Exit(2)
	}
	
	// Check which command we're mocking
	cmd, args := args[0], args[1:]
	
	switch cmd {
	case "llama-server":
		// Mock successful execution of llama-server
		os.Exit(0)
	default:
		// Unknown command
		os.Exit(1)
	}
}

 // TestMainWithMock tests the main function with a mocked exec.Command.
 // It verifies that `main` runs to completion without invoking the real
 // “llama‑server” binary.
 func TestMainWithMock(t *testing.T) {
	 // -----------------------------------------------------------------
	 // 1️⃣  Save the original execCommand and restore it after the test.
	 // -----------------------------------------------------------------
	 origExecCommand := execCommand
	 defer func() { execCommand = origExecCommand }()

	 // -----------------------------------------------------------------
	 // 2️⃣  Replace execCommand with the mock implementation.
	 // -----------------------------------------------------------------
	 execCommand = mockExecCommand

	 // -----------------------------------------------------------------
	 // 3️⃣  Create a minimal temporary config file.
	 // -----------------------------------------------------------------
	 const yaml = "model: /tmp/dummy.gguf\n"
	 cfgFile := createTempFile(t, yaml)
	 defer os.Remove(cfgFile)

	 // -----------------------------------------------------------------
	 // 4️⃣  Set up the command‑line arguments for the test run.
	 // -----------------------------------------------------------------
	 oldArgs := os.Args
	 defer func() { os.Args = oldArgs }()
	 os.Args = []string{"llauncher", "--config", cfgFile}

	 // -----------------------------------------------------------------
	 // 5️⃣  Suppress any output that `main()` (or the code it calls) writes
	 //     to stdout / stderr.  This keeps the test runner silent while
	 //     preserving the existing behaviour and coverage.
	 // -----------------------------------------------------------------
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

	 // -----------------------------------------------------------------
	 // 5️⃣  Run main().  The mock will cause the spawned “llama‑server”
	 //    process to exit immediately with status 0, so main should return
	 //    without calling os.Exit or panicking.
	 // -----------------------------------------------------------------
	 main()
	 // If we reach this point the test succeeded.
 }
