
package main

import (
    "os"
)

// init prepends the current working directory to PATH so the test suite can
// locate the “llauncher” wrapper script. This code is compiled only when the
// `test` build tag is supplied (e.g. `go test -tags=test`).
func init() {
    cwd, err := os.Getwd()
    if err != nil {
        // If we cannot get the working directory, just return.
        return
    }
    // Prepend the cwd to PATH.
    pathEnv := os.Getenv("PATH")
    if pathEnv == "" {
        os.Setenv("PATH", cwd)
    } else {
        os.Setenv("PATH", cwd+string(os.PathListSeparator)+pathEnv)
    }
}
