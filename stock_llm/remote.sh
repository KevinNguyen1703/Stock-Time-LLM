#!/bin/bash
# Offload Time-LLM+ runs to the second GPU host (ssh alias "gpu2", see ~/.ssh/config).
#   bash stock_llm/remote.sh sync                      # push code + data (not .venv/checkpoints/logs/reports)
#   bash stock_llm/remote.sh run <name> <train_plus args>   # start one run in the background on gpu2
#   bash stock_llm/remote.sh wait                      # block until gpu2 runs finish, then pull results back
set -e
HOST=${REMOTE_HOST:-gpu2}
DIR=Gumiho/Stock-Time-LLM
cd "$(dirname "$0")/.."
case "$1" in
  sync)
    rsync -az --exclude .venv --exclude checkpoints --exclude logs --exclude reports --exclude .git \
      --exclude '*.zip' --exclude dataset --exclude report ./ "$HOST:$DIR/" ;;
  run)
    name=$2; shift 2
    ssh "$HOST" "cd $DIR && mkdir -p logs && nohup .venv/bin/python -m stock_llm.train_plus $* > logs/$name.log 2>&1 < /dev/null &"
    echo "started $name on $HOST" ;;
  wait)
    while ssh "$HOST" 'pgrep -f "^.venv/bin/python -m stock_llm.train_plus" > /dev/null'; do sleep 60; done
    rsync -az "$HOST:$DIR/reports/" reports/
    rsync -az "$HOST:$DIR/checkpoints/" checkpoints/
    rsync -az "$HOST:$DIR/logs/" logs/gpu2/
    echo "gpu2 idle; results pulled" ;;
  *) echo "usage: $0 sync|run <name> <args>|wait"; exit 1 ;;
esac
