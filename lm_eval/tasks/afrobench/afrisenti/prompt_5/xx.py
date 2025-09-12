# data = load_dataset('HausaNLP/AfriSenti-Twitter', 'yor', trust_remote_code=True)
# print(data)

import torch
from torch.cuda import is_available, device_count

try:
    import torch_musa
    from torch_musa.core.device import is_available, device_count
except ModuleNotFoundError:
    torch_musa = None


print(is_available())  # Should return True
print(device_count())
