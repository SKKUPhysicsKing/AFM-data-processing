// Build on any Go host with GOOS=windows GOARCH=amd64 CGO_ENABLED=0.
package main

import (
	"context"
	"embed"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"time"
	"unsafe"
)

//go:embed *.py requirements.txt
var sources embed.FS

// Set at build time to the commit of the bundled application.
var seedRevision string

const appName = "AFM XE7 ANALYZER"
const createNoWindow = 0x08000000

func showError(text string) {
	title, _ := syscall.UTF16PtrFromString(appName)
	message, _ := syscall.UTF16PtrFromString(text)
	proc := syscall.NewLazyDLL("user32.dll").NewProc("MessageBoxW")
	proc.Call(0, uintptr(unsafe.Pointer(message)), uintptr(unsafe.Pointer(title)), 0x10)
}

func installedPython() []string {
	probe := "import pathlib, sys, tkinter; assert sys.version_info >= (3, 10); " +
		"assert pathlib.Path(sys.executable).with_name('pythonw.exe').is_file(); tkinter.Tcl()"
	for _, candidate := range [][]string{{"py.exe", "-3"}, {"python.exe"}} {
		ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
		args := append(append([]string{}, candidate[1:]...), "-I", "-c", probe)
		cmd := exec.CommandContext(ctx, candidate[0], args...)
		cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: createNoWindow}
		err := cmd.Run()
		cancel()
		if err == nil {
			return candidate
		}
	}
	return nil
}

func main() {
	python := installedPython()
	if python == nil {
		showError("Setup needs Python 3.10 or later with Tcl/Tk. Install the Windows Python distribution from python.org, then run Setup again.")
		return
	}
	dir, err := os.MkdirTemp("", "afm-xe7-setup-")
	if err != nil {
		showError(err.Error())
		return
	}
	defer os.RemoveAll(dir)
	entries, err := sources.ReadDir(".")
	if err != nil {
		showError(err.Error())
		return
	}
	for _, entry := range entries {
		content, err := sources.ReadFile(entry.Name())
		if err == nil {
			err = os.WriteFile(filepath.Join(dir, entry.Name()), content, 0600)
		}
		if err != nil {
			showError(err.Error())
			return
		}
	}
	args := append(append([]string{}, python[1:]...), "-I", filepath.Join(dir, "launcher.py"), "--install")
	if seedRevision != "" {
		args = append(args, "--seed", seedRevision)
	}
	cmd := exec.Command(python[0], args...)
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: createNoWindow}
	if err := cmd.Start(); err != nil {
		showError(err.Error())
		return
	}
	// The setup window reports installation errors directly.
	_ = cmd.Wait()
}
