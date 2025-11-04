#!/bin/bash

# =============================================================================
# BITRECS SETUP - Minimal & Clean
# =============================================================================

# Colors
GRAY='\033[0;90m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# Config
LOGFILE="bitrecs_setup_$(date +%Y%m%d_%H%M%S).log"

# =============================================================================
# SPINNER
# =============================================================================

spinner_pid=""

show_spinner() {
    local msg="$1"
    local delay=0.1
    local spinstr='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'

    tput civis # Hide cursor

    while true; do
        local temp=${spinstr#?}
        printf " ${GRAY}%c${NC}  ${DIM}%s${NC}\r" "$spinstr" "$msg"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
    done
}

start_spinner() {
    show_spinner "$1" &
    spinner_pid=$!
}

stop_spinner() {
    if [ -n "$spinner_pid" ]; then
        kill "$spinner_pid" 2>/dev/null
        wait "$spinner_pid" 2>/dev/null
        spinner_pid=""
        printf "\r\033[K" # Clear line
    fi
    tput cnorm # Show cursor
}

# =============================================================================
# OUTPUT
# =============================================================================

print_step() {
    echo -e "\n${BOLD}$1${NC}"
}

print_ok() {
    echo -e " ${GREEN}✓${NC}  $1"
}

print_skip() {
    echo -e " ${GRAY}−${NC}  ${DIM}$1${NC}"
}

print_error() {
    echo -e " ${RED}✗${NC}  $1"
}

print_dim() {
    echo -e "    ${DIM}$1${NC}"
}

# =============================================================================
# COMMAND EXECUTION
# =============================================================================

run_silent() {
    local msg="$1"
    local cmd="$2"

    start_spinner "$msg"

    if eval "$cmd" >> "$LOGFILE" 2>&1; then
        stop_spinner
        print_ok "$msg"
        return 0
    else
        stop_spinner
        print_error "$msg"
        echo "$(date '+%Y-%m-%d %H:%M:%S') ERROR: $cmd" >> "$LOGFILE"
        return 1
    fi
}

# =============================================================================
# INPUT
# =============================================================================

ask() {
    local prompt="$1"
    local default="$2"

    if [[ "$default" == "y" ]]; then
        echo -n -e "  ${prompt} ${DIM}(Y/n)${NC} "
    else
        echo -n -e "  ${prompt} ${DIM}(y/N)${NC} "
    fi

    read -r response

    if [[ -z "$response" ]]; then
        response="$default"
    fi

    response=$(echo "$response" | tr '[:upper:]' '[:lower:]')

    [[ "$response" == "y" || "$response" == "yes" ]]
}

ask_port() {
    local prompt="$1"
    local default="$2"

    while true; do
        echo -n -e "  ${prompt} ${DIM}($default)${NC} "
        read -r port

        if [[ -z "$port" ]]; then
            echo "$default"
            return 0
        fi

        if [[ "$port" =~ ^[0-9]+$ ]] && [ "$port" -ge 1 ] && [ "$port" -le 65535 ]; then
            echo "$port"
            return 0
        fi

        print_error "Invalid port"
    done
}

# =============================================================================
# MAIN
# =============================================================================

trap 'stop_spinner; tput cnorm' EXIT

main() {
    clear

    # Header
    echo ""
    echo -e "${BOLD}bitrecs${NC} ${DIM}setup${NC}"
    echo ""

    # Root check
    if [ "$EUID" -ne 0 ]; then
        print_error "Must run as root"
        echo ""
        print_dim "sudo bash $0"
        echo ""
        exit 1
    fi

    # Config
    if ask "Use default setup?" "y"; then
        MODE="default"
        UFW_PORT=8091
        AUTO_UFW="y"
    else
        MODE="manual"
        echo ""
        if ask "Configure firewall?" "y"; then
            UFW_PORT=$(ask_port "Port number:" "8091")
            AUTO_UFW="n"
        else
            UFW_PORT=""
            AUTO_UFW="n"
        fi
    fi

    # Start
    print_step "Installing"

    # Swap
    if [ -f /swapfile ]; then
        print_skip "swap already exists"
    else
        run_silent "swap 4GB" "fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile && grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab"
    fi

    # Firewall
    if [[ -n "$UFW_PORT" ]]; then
        run_silent "ufw install" "apt install ufw -y"
        run_silent "ufw allow ssh" "ufw allow 22"
        run_silent "ufw allow $UFW_PORT" "ufw allow proto tcp to 0.0.0.0/0 port $UFW_PORT"

        if [[ "$AUTO_UFW" == "y" ]]; then
            run_silent "ufw enable" "yes | ufw enable && ufw reload"
        else
            echo ""
            if ask "Enable firewall now?" "y"; then
                run_silent "ufw enable" "yes | ufw enable && ufw reload"
            else
                print_skip "firewall not enabled"
            fi
        fi
    fi

    # System
    run_silent "system update" "apt-get update && apt-get upgrade -y"
    run_silent "cleanup disk" "apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* ~/.cache/pip"

    # Node
    run_silent "node.js deps" "apt install -y curl gnupg"
    run_silent "node.js repo" "curl -fsSL https://deb.nodesource.com/setup_18.x | bash -"
    run_silent "node.js install" "apt install -y nodejs"
    run_silent "pm2 install" "npm install -g pm2"

    # Python
    run_silent "python install" "apt install -y python3-pip python3.12-venv"
    run_silent "python venv" "mkdir -p /root/pip_tmp && python3.12 -m venv \$HOME/bt/bt_venv"
    run_silent "bashrc update" "grep -qxF 'source \$HOME/bt/bt_venv/bin/activate' ~/.bashrc || echo 'source \$HOME/bt/bt_venv/bin/activate' >> ~/.bashrc"

    # Bitrecs
    print_step "Bitrecs"
    run_silent "clone repo" "mkdir -p \$HOME/bt && cd \$HOME/bt && rm -rf bitrecs-subnet || true && git clone https://github.com/bitrecs/bitrecs-subnet.git"
    run_silent "install packages" "cd \$HOME/bt/bitrecs-subnet && source \$HOME/bt/bt_venv/bin/activate && TMPDIR=/root/pip_tmp pip install -e . --no-cache-dir"

    # Done
    echo ""
    print_step "Complete"
    print_dim "repo: ~/bt/bitrecs-subnet"
    print_dim "logs: $LOGFILE"
    echo ""
    print_dim "Next: Open new terminal, configure wallet & .env"

    if [[ -z "$UFW_PORT" || "$AUTO_UFW" == "n" ]]; then
        print_dim "      Enable firewall: sudo ufw enable"
    fi

    echo ""
}

main "$@"
