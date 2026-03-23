#!/usr/bin/env zsh

# Monitor sequential benchmark runs
# Usage: zsh monitor.sh [interval_seconds]
# Default: checks every 60 seconds

INTERVAL=${1:-60}
TMP_HOME=/data/vgupta345/prowl_related_data/prowl-open-source

while true; do
    clear
    echo "========================================"
    echo "  BENCHMARK MONITOR — $(date)"
    echo "========================================"

    # Find the latest sequential log
    LOGFILE=$(ls -t ${TMP_HOME}/results/logs/all_sequential_full_*.log 2>/dev/null | head -1)
    if [ -z "$LOGFILE" ]; then
        LOGFILE=$(ls -t ${TMP_HOME}/results/logs/all_sequential_*.log 2>/dev/null | head -1)
    fi

    if [ -z "$LOGFILE" ]; then
        echo "  No sequential run log found."
    else
        LOG_MOD=$(stat -c '%y' "$LOGFILE" 2>/dev/null | cut -d. -f1)
        COMPLETED=$(grep -c 'Done:' "$LOGFILE" 2>/dev/null)
        TOTAL=$(grep -oP '\d+/\d+' "$LOGFILE" 2>/dev/null | tail -1)
        CURRENT=$(grep '\[.*/' "$LOGFILE" 2>/dev/null | tail -1 | sed 's/^  //')

        echo ""
        echo "  Log: $(basename $LOGFILE)"
        echo "  Last updated: ${LOG_MOD}"
        echo "  Completed: ${COMPLETED}/32"
        echo "  Current: ${CURRENT}"

        # Check if log is stale (>10 min since last update)
        LOG_EPOCH=$(stat -c '%Y' "$LOGFILE" 2>/dev/null)
        NOW_EPOCH=$(date +%s)
        STALE_SEC=$(( NOW_EPOCH - LOG_EPOCH ))
        if [ $STALE_SEC -gt 600 ]; then
            echo ""
            echo "  ⚠ WARNING: Log stale for $(( STALE_SEC / 60 )) minutes!"
        fi
    fi

    # Active processes
    echo ""
    echo "  ACTIVE PROCESSES:"
    PROCS=$(ps aux | grep vgupta | grep -E 'lm_eval_online_serve|vllm.*api_server' | grep -v grep)
    if [ -z "$PROCS" ]; then
        echo "    (none)"
    else
        echo "$PROCS" | while read line; do
            PID=$(echo $line | awk '{print $2}')
            CPU=$(echo $line | awk '{print $3}')
            START=$(echo $line | awk '{print $9}')
            # Extract key info
            if echo "$line" | grep -q 'lm_eval_online_serve'; then
                BENCH=$(echo "$line" | grep -oP '(?<=-b )\S+')
                MODEL=$(echo "$line" | grep -oP '(?<=-m )\S+' | xargs basename)
                LIMIT=$(echo "$line" | grep -oP '(?<=-l )\S+')
                CONF=$(echo "$line" | grep -oP '(?<=conf_)\S+(?=/)' || echo "$line" | grep -oP '(?<=-cf )\S+' | xargs basename | sed 's/.json//')
                echo "    PID ${PID} | ${MODEL} / ${BENCH} (limit=${LIMIT:-def}) | CPU=${CPU}% | since ${START}"
            elif echo "$line" | grep -q 'api_server'; then
                PORT=$(echo "$line" | grep -oP '(?<=--port )\d+')
                MODEL=$(echo "$line" | grep -oP '(?<=--model )\S+' | xargs basename)
                echo "    PID ${PID} | vllm server ${MODEL} :${PORT} | CPU=${CPU}% | since ${START}"
            fi
        done
    fi

    # Port status
    echo ""
    echo "  PORTS:"
    for p in 8019 8020 8030 8040; do
        HEALTH=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 localhost:${p}/health 2>/dev/null)
        if [ "$HEALTH" = "200" ]; then
            echo "    :${p} — UP"
        else
            echo "    :${p} — down"
        fi
    done

    # Check for stuck processes (sleeping, 0% CPU, >10 min old)
    echo ""
    STUCK=""
    ps aux | grep vgupta | grep -E 'lm_eval_online_serve' | grep -v grep | while read line; do
        PID=$(echo $line | awk '{print $2}')
        CPU=$(echo $line | awk '{print $3}')
        STATE=$(cat /proc/$PID/status 2>/dev/null | grep '^State' | awk '{print $2}')
        if [ "$CPU" = "0.0" ] && [ "$STATE" = "S" ]; then
            # Check how long it's been idle
            START_TIME=$(stat -c '%Y' /proc/$PID 2>/dev/null)
            if [ -n "$START_TIME" ]; then
                AGE=$(( $(date +%s) - START_TIME ))
                if [ $AGE -gt 600 ]; then
                    BENCH=$(echo "$line" | grep -oP '(?<=-b )\S+')
                    echo "  ⚠ STUCK: PID ${PID} (${BENCH}) — idle ${AGE}s, 0% CPU"
                    STUCK="yes"
                fi
            fi
        fi
    done
    if [ -z "$STUCK" ]; then
        echo "  No stuck processes detected."
    fi

    # Recent completions
    if [ -n "$LOGFILE" ]; then
        echo ""
        echo "  RECENT COMPLETIONS:"
        grep 'Done:' "$LOGFILE" 2>/dev/null | tail -5 | sed 's/^/    /'
        if [ $(grep -c 'Done:' "$LOGFILE" 2>/dev/null) -eq 0 ]; then
            echo "    (none yet)"
        fi
    fi

    echo ""
    echo "========================================"
    echo "  Refreshing in ${INTERVAL}s... (Ctrl+C to stop)"
    echo "========================================"
    sleep $INTERVAL
done
