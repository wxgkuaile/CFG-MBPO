# loss_list = []
import numpy as np
# reward = [1/(i+1) for i in range(100)]
#
# for index in range(100):
#     if index < 5:
#         loss_list.append(np.mean(reward[index]))
#     else:
#         now_index = index % 5
#         loss_list[now_index] = np.mean(reward[index])
#         min_value = min(loss_list)
#         min_index = loss_list.index(min_value)
#         if min_index == now_index:
#             print("True, min_index, index: ", min_index, index)
#     # reward.append(index)
def test(b,a=None):
    if a:
        b=3
    return b

from env import register_mbpo_environments

print(register_mbpo_environments())