# Llauncher

Llauncher will read a YAML configuration file, and then run llama-server from PATH with the command line options taken from the YAML configuration. It is intended to act as the Entrypoint in a container to reduce the burden of managing llama-server configuration options.

In general [Llama-Swap](https://github.com/mostlygeek/llama-swap/) is a better option, even for a single model. It has superior config management, is more widely used and better maintained. This was written to meet a peculiar set of constraints that made llama-swap awkward to deploy.

In general, the format of the YAML file is simply the longest version of the option given by `llama-server --help` without the double-dash (e.g. "n-gpu-layers" is used rather than "gpu-layers" or "ngl"). There is no validation of the options beyond the parser checking types and valid strings in the YAML.

All llama-server options should be supported as of 20250825 - but this is likely to degrade over time.


## Example YAML Config
```yaml
model: /var/lib/models/gpt-oss-120b.gguf
alias: GPT-OSS-120b
port: 9000
ctx-size: 131072
jinja: true
cache-type-k: q8_0
cache-type-v: q8_0
n-gpu-layers: 99
n-cpu-moe: 36
flash-attn: true
chat-template-kwargs: |-
  '{"reasoning_effort": "high"}'
top-p: 1.00
min-p: 0.05
temp: 1.0
top-k: 40
```