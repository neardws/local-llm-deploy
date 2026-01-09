#!/bin/bash
# Configure HuggingFace domestic mirror for faster downloads in China

HF_MIRROR="https://hf-mirror.com"

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --temp       Set mirror for current session only"
    echo "  --persist    Add mirror config to shell profile (default)"
    echo "  --remove     Remove mirror config from shell profile"
    echo "  --status     Show current mirror configuration"
    echo "  -h, --help   Show this help message"
}

get_shell_profile() {
    if [ -n "$ZSH_VERSION" ]; then
        echo "$HOME/.zshrc"
    elif [ -n "$BASH_VERSION" ]; then
        echo "$HOME/.bashrc"
    else
        echo "$HOME/.profile"
    fi
}

show_status() {
    echo "Current HF_ENDPOINT: ${HF_ENDPOINT:-<not set>}"
    
    profile=$(get_shell_profile)
    if grep -q "HF_ENDPOINT" "$profile" 2>/dev/null; then
        echo "Persisted in: $profile"
        grep "HF_ENDPOINT" "$profile"
    else
        echo "Not persisted in shell profile"
    fi
}

set_temp() {
    export HF_ENDPOINT="$HF_MIRROR"
    echo "Mirror set for current session: $HF_MIRROR"
    echo ""
    echo "To use in your Python scripts, run:"
    echo "  source $(dirname "$0")/config_mirror.sh --temp"
}

set_persist() {
    profile=$(get_shell_profile)
    
    if grep -q "HF_ENDPOINT" "$profile" 2>/dev/null; then
        echo "HF_ENDPOINT already configured in $profile"
    else
        echo "" >> "$profile"
        echo "# HuggingFace Mirror (added by local-llm-deploy)" >> "$profile"
        echo "export HF_ENDPOINT=$HF_MIRROR" >> "$profile"
        echo "Added to $profile"
    fi
    
    export HF_ENDPOINT="$HF_MIRROR"
    echo "Mirror configured: $HF_MIRROR"
    echo ""
    echo "Run 'source $profile' or restart terminal to apply."
}

remove_config() {
    profile=$(get_shell_profile)
    
    if grep -q "HF_ENDPOINT" "$profile" 2>/dev/null; then
        sed -i '/# HuggingFace Mirror/d' "$profile"
        sed -i '/HF_ENDPOINT/d' "$profile"
        echo "Removed HF_ENDPOINT from $profile"
    else
        echo "No HF_ENDPOINT config found in $profile"
    fi
    
    unset HF_ENDPOINT
}

case "${1:-}" in
    --temp)
        set_temp
        ;;
    --persist|"")
        set_persist
        ;;
    --remove)
        remove_config
        ;;
    --status)
        show_status
        ;;
    -h|--help)
        usage
        ;;
    *)
        echo "Unknown option: $1"
        usage
        exit 1
        ;;
esac
