env_name=$1
num_e=$2
date_str=`date +%Y.%m.%d_%H.%M.%S`
echo " program start time : " + $date_str
python main_mbpo.py --env_name ${env_name}'-v2' --num_epoch=${num_e} --use_algo 'flowrl'
