#!/bin/bash
# 等待 SphereNet naive distillation 完成后，自动启动 SchNet EDG++ rMD17
cd /home/lzeng/workspace/GEOM3D

NAIVE_BASE=./experiments/run_rMD17_EDG_naive_lambda0.001/SphereNet/e1000_b128_lr5e-4_ed128_lsCosine/ED/ED0.001/rs42
WATCH_MOLECULES=(aspirin paracetamol azobenzene)

echo "[$(date)] Watcher started. Waiting for naive distillation to complete..."
echo "Monitoring: ${WATCH_MOLECULES[*]}"

while true; do
    all_done=true
    for mol in "${WATCH_MOLECULES[@]}"; do
        log="${NAIVE_BASE}/${mol}/logs.log"
        if [ -f "$log" ] && tail -1 "$log" | grep -q "best Force"; then
            : # done
        else
            all_done=false
            break
        fi
    done

    if $all_done; then
        echo ""
        echo "[$(date)] All 3 naive distillation experiments completed!"
        echo ""

        # 打印 naive distillation 结果
        for mol in "${WATCH_MOLECULES[@]}"; do
            echo "=== ${mol} ==="
            tail -2 "${NAIVE_BASE}/${mol}/logs.log"
        done
        echo ""

        # 启动 SchNet EDG++ rMD17（4卡并行，自动跳过已完成的 benzene）
        echo "[$(date)] Launching SchNet EDG++ rMD17 on GPUs 0,1,2,3 ..."
        bash /home/lzeng/workspace/GEOM3D/run_SchNet_rMD17_EDGpp.sh 0,1,2,3
        echo "[$(date)] All SchNet jobs launched. Watcher exiting."
        exit 0
    fi

    sleep 300  # 每 5 分钟检查一次
done
