from transformers import AutoModelForImageTextToText, AutoConfig

# model_id = "./qwen3_weights" # 指向你下载的目录
model_id = "Qwen/Qwen3-VL-8B-Instruct" # 直接使用模型库中的模型
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="cpu", trust_remote_code=True)

print(model)