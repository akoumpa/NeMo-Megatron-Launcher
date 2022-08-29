DATA_DIR=/lustre/fsw/joc/big_nlp/gpt3/prepare_dataset/the_pile/train

HYDRA_FULL_ERROR=1 BIGNLP_CI=1 python3 main.py \
    export=gpt3/export_gpt3 \
    stages=["export"] \
    bignlp_path=${GIT_CLONE_PATH} \
    data_dir=${DATA_DIR} \
    base_results_dir=${BASE_RESULTS_DIR} \
    "container='${BUILD_IMAGE_NAME_SRUN}'" \
    cluster.partition=${SLURM_PARTITION} \
    cluster.account=${SLURM_ACCOUNT} \
    cluster.gpus_per_task=null \
    cluster.gpus_per_node=null \
    cluster.job_name_prefix="${SLURM_ACCOUNT}-bignlp_ci:" \
    export.run.name=${RUN_NAME} \
    export.run.time_limit="30:00" \
    export.run.model_train_name=gpt3_${RUN_MODEL_SIZE}_tp${TP_SIZE}_pp${PP_SIZE} \
    export.run.results_dir=${BASE_RESULTS_DIR}/${RUN_NAME} \
    export.model.checkpoint_path=${BASE_RESULTS_DIR}/train_gpt3_${RUN_MODEL_SIZE}_tp${TP_SIZE}_pp${PP_SIZE}_${PP_SIZE}node_100steps/results/checkpoints \
    export.model.weight_data_type=${FT_WEIGHT_DATA} \
    export.model.tensor_model_parallel_size=${FT_TP_SIZE} \
    export.triton_deployment.pipeline_model_parallel_size=${FT_PP_SIZE} \
    export.triton_deployment.data_type=${FT_DEPLOYMENT_DATA}
