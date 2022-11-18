training_script=src.train_scripts.causal_training

train_status=0
while getopts "d:" opt; do
	case $opt in
		d)
            train_status=${OPTARG}
    esac
done

debug() {
    echo "start debugging"
    python -m $training_script --debug_status 1
}

train() {
    echo "start training"
    git_status=$(git status -s) 

    if [ -n "$git_status" ]; then
        echo "You have uncommitted changes. Please commit them first."
        exit 1
    fi

    train_log_path=causal_model_$(date +"%d.%m.%Y_%H:%M:%S").log
    nohup python -m $training_script > ./training_logs/causal_model/$train_log_path &
}

# clear dir
rm -rf ./training_logs/causal_model/*

if [ $train_status -eq 1 ]; then
    debug
elif [ $train_status -eq 0 ]; then
    train
fi