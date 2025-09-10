import torch

ckpt_path = r"XEUS\model\xeus_checkpoint_old.pth"
state = torch.load(ckpt_path, map_location="cpu")

args = state.get("train_args", state.get("args", None))
print(type(args))
print(args)

if args is not None:
    print("Has frontend_conf?", hasattr(args, "frontend_conf"))
    if hasattr(args, "frontend_conf"):
        print("frontend_conf:", args.frontend_conf)

if args and hasattr(args, "frontend_conf"):
    fc = args.frontend_conf
    if isinstance(fc, dict):
        print("Keys in frontend_conf:", fc.keys())
        if "normalize_output" in fc:
            print("normalize_output:", fc["normalize_output"])
