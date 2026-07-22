
#!/bin/bash
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
else
    echo "Warning: CANN set_env.sh not found!"
fi

if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
elif [ -f /usr/local/Ascend/nnal/atb/latest/set_env.sh ]; then
    source /usr/local/Ascend/nnal/atb/latest/set_env.sh
else
    echo "Warning: ATB set_env.sh not found!"
fi


/usr/local/python3.12.13/bin/python3 /usr/local/python3.12.13/bin/vllm serve \
    /home/x30073879/minicpm/OpenBMB/MiniCPM-o-4_5/ \
    --omni \
    --port 8889 \
    --log-stats \
    --trust-remote-code \
    --stage-init-timeout 600 \
    --enforce-eager \
    --stage-overrides '{
      "0": {
        "profiler_config": {
          "profiler": "torch",
          "torch_profiler_dir": "/data/profiles/minicpmo45",
          "torch_profiler_with_stack": false,
          "torch_profiler_with_memory": false
        }
      },
      "1": {
        "profiler_config": {
          "profiler": "torch",
          "torch_profiler_dir": "/data/profiles/minicpmo45",
          "torch_profiler_with_stack": false,
          "torch_profiler_with_memory": false
        }
      }
    }'
    