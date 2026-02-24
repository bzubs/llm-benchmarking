import torch 
print("Number of visible GPUs:", torch.cuda.device_count())
print("Current GPU:", torch.cuda.current_device())
print("GPU Name:", torch.cuda.get_device_name(0))


free, total = torch.cuda.mem_get_info(0)
print("Free (GiB):", free / 1024**3)
print("Total (GiB):", total / 1024**3)


if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
else:
    major, minor = (0, 0)

print(major, minor)