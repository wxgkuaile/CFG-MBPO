env_name=$1
num_e=$2
date_str=`date +%Y.%m.%d_%H.%M.%S`
for i in 1 2 3 4 5
do
  nohup python cfg_mbpo.py --env_name ${env_name}'-v2' --num_epoch=${num_e} --use_algo 'flowrl' 1> /dev/null 2> out/${env_name}_mbpo_error_$date_str.txt  &
  sleep 30
done
