training_script=dimweb_persona_bot.train_scripts.causal_training

train_status=0
cuda_device=-1

while getopts "d:c:" opt; do
	case $opt in
		d)
            train_status=${OPTARG};;
        c)
            cuda_device=${OPTARG};;
    esac
done

debug() {
    echo "start debugging"
    python -m $training_script --debug_status 1 --cuda_device $1
}

train() {
    echo "start training"
    git_status=$(git status -s) 

    if [ -n "$git_status" ]; then
        echo "You have uncommitted changes. Please commit them first."
        exit 1
    fi

    train_log_path=causal_model_$(date +"%d.%m.%Y_%H:%M:%S").log
    nohup python -m $training_script --cuda_device $1 > ./training_logs/causal_model/$train_log_path &
}

# clear dir
# rm -rf ./training_logs/causal_model/*

if [ $cuda_device -eq -1 ]; then
    echo "Please specify the cuda device"
    exit 1
# compare $cuda_device with string "all"
elif [ $cuda_device -eq -2 ]; then
    cuda_device=-1
fi


echo Your cuda device is $cuda_device

if [ $train_status -eq 1 ]; then
    debug $cuda_device
elif [ $train_status -eq 0 ]; then
    train $cuda_device
fi