
    export VIVARIUM_LOGGING_DIRECTORY=/share/costeffectiveness/results/vivarium_experiment_uk_econ/2019_10_03_17_39_40/logs/2019_10_03_17_39_40_run/worker_logs
    export PYTHONPATH=/ihme/costeffectiveness/results/vivarium_experiment_uk_econ/2019_10_03_17_39_40:$PYTHONPATH

    /homes/abie/.conda/envs/vivarium_experiment_uk_econ/bin/rq worker -c settings --name ${JOB_ID}.${SGE_TASK_ID} --burst -w "vivarium_cluster_tools.psimulate.distributed_worker.ResilientWorker" --exception-handler "vivarium_cluster_tools.psimulate.distributed_worker.retry_handler" vivarium

    