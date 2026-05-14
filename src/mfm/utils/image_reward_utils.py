import torch
import torch.nn.functional as F
from imscore.hps.model import HPSv2
from imscore.imreward.model import ImageReward
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
CLIP_RESCALE = 1.0 / 255.0
CLIP_SIZE = 224


def clip_preprocess_torch(images: torch.Tensor):
    x = images
    if x.detach().max() > 2.5:
        x = x * CLIP_RESCALE
    B, C, H, W = x.shape
    scale = CLIP_SIZE / min(H, W)
    new_h = int(round(H * scale))
    new_w = int(round(W * scale))
    x = F.interpolate(x, size=(new_h, new_w), mode="bicubic", align_corners=False)
    top = (new_h - CLIP_SIZE) // 2
    left = (new_w - CLIP_SIZE) // 2
    x = x[:, :, top : top + CLIP_SIZE, left : left + CLIP_SIZE]
    mean = CLIP_MEAN.to(x.device, x.dtype)
    std = CLIP_STD.to(x.device, x.dtype)
    x = (x - mean) / std
    return x


def clip_scores_per_image(processor, model, prompts, images, device):
    """
    images
    Returns: torch.float32 scores shape [N] in [0, 100] (clamped like CLIPScore).
    """
    images = torch.stack([img for img in images]) if type(images) is list else images
    images = clip_preprocess_torch(images)
    inputs = processor(text=prompts, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    image_features = model.get_image_features(pixel_values=images)  # [N,D]
    text_features = model.get_text_features(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
    )  # [N,D]
    image_features = F.normalize(image_features, dim=-1)  # [N,D]
    text_features = F.normalize(text_features, dim=-1)  # [N,D]
    sims = (image_features * text_features).sum(dim=-1)  # [N]
    scores = sims.clamp(min=0.0)
    return scores.float()


def load_image_reward_fn(cfg, device, model_name="ImageReward"):
    reward_model = get_image_reward_model(device, model_name=model_name)

    def reward_fn(images):
        imgs = [img for img in images]
        n = len(imgs)        
        batch_imgs = imgs
        batch_prompts = [cfg.prompt] * n
        all_scores = reward_model(batch_prompts, batch_imgs)
        pos_scores = all_scores[:n]
        pos_scores = cfg._lambda * pos_scores
        return pos_scores
    return reward_fn


def get_image_reward_model(
    device, model_name="ImageReward"
):  # takes in tensors from [0, 1]
    print(f"Loading image reward model: {model_name}")
    if model_name == "ImageReward":  # works
        model = ImageReward.from_pretrained(
            "RE-N-Y/ImageReward"
        )  # ImageReward aesthetic scorer
        model.to(device).eval()

        def reward_model(prompts, images):
            if type(images) is list:
                images = torch.stack([img for img in images])
            scores = model.score(images, prompts)
            return scores

        return reward_model
    elif model_name == "CLIP":  
        model_id = "openai/clip-vit-base-patch16"
        model = CLIPModel.from_pretrained(model_id).to(device).eval()
        processor = CLIPProcessor.from_pretrained(model_id)
        return lambda prompts, images: clip_scores_per_image(
            processor, model, prompts, images, device
        )
    elif model_name == "PickScore":  
        processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"
        processor = AutoProcessor.from_pretrained(processor_name_or_path)
        model = (
            AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)
        )

        def reward_model(prompts, images):
            images = (
                torch.stack([img for img in images]) if type(images) is list else images
            )
            images = clip_preprocess_torch(images)
            image_embs = model.get_image_features(pixel_values=images)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

            with torch.no_grad():
                text_inputs = processor(
                    text=prompts,
                    padding=True,
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                ).to(device)
                text_embs = model.get_text_features(**text_inputs)
                text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

            scores = model.logit_scale.exp() * torch.sum(text_embs * image_embs, dim=-1)
            return scores

        return reward_model
    elif model_name == "HPSv2":
        model = HPSv2.from_pretrained("RE-N-Y/hpsv21")  # HPSv2.1 preference scorer
        model.to(device).eval()

        def reward_model(prompts, images):
            if type(images) is list:
                images = torch.stack([img for img in images])
            scores = model.score(images, prompts)
            return scores

        return reward_model
    else:
        raise NotImplementedError(f"Unknown image reward model: {model_name}")
