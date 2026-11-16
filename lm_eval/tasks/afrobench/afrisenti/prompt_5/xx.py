# data = load_dataset('HausaNLP/AfriSenti-Twitter', 'yor', trust_remote_code=True)
# print(data)

import torch


print(torch.cuda.is_available())  # Should return True
print(torch.cuda.device_count())
