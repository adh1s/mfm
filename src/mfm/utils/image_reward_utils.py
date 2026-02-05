import os
from typing import Union

import torch
from PIL import Image

"""
@File       :   ImageReward.py
@Time       :   2023/01/28 19:53:00
@Auther     :   Jiazheng Xu
@Contact    :   xjz22@mails.tsinghua.edu.cn
@Description:   ImageReward Reward model.
* Based on CLIP code base and improved-aesthetic-predictor code base
* https://github.com/openai/CLIP
* https://github.com/christophschuhmann/improved-aesthetic-predictor
"""

import torch.nn as nn

try:
    import ImageReward as RM
    from ImageReward.models.BLIP.blip_pretrain import BLIP_Pretrain
    from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,
                                        ToTensor)
except ImportError:
    pass

import torch.nn.functional as F
from imscore.hps.model import HPSv2
from imscore.imreward.model import ImageReward

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# import ImageReward as RM
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose(
        [
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def _transform_tensor():
    return Compose(
        [
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            # nn.ReLU(),
            nn.Linear(16, 1),
        )

        # initial MLP param
        for name, param in self.layers.named_parameters():
            if "weight" in name:
                nn.init.normal_(param, mean=0.0, std=1.0 / (self.input_size + 1))
            if "bias" in name:
                nn.init.constant_(param, val=0)

    def forward(self, input):
        return self.layers(input)


class IRSMC(nn.Module):
    def __init__(self, med_config, device="cpu"):
        super().__init__()
        self.device = device

        self.blip = BLIP_Pretrain(image_size=224, vit="large", med_config=med_config)
        self.preprocess = _transform(224)
        self.preprocess_tensor = _transform_tensor()
        self.mlp = MLP(768)

        self.mean = 0.16717362830052426
        self.std = 1.0333394966054072

    def score(self, prompt, image):
        if type(image).__name__ == "list":
            _, rewards = self.inference_rank(prompt, image)
            return rewards

        # text encode
        text_input = self.blip.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(self.device)

        # image encode
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, str) and os.path.isfile(image):
            pil_image = Image.open(image)
        else:
            raise TypeError(
                r"This image parameter type has not been supportted yet. Please pass PIL.Image or file path str."
            )

        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        image_embeds = self.blip.visual_encoder(image)

        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )
        text_output = self.blip.text_encoder(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        txt_features = text_output.last_hidden_state[:, 0, :].float()  # (feature_dim)
        rewards = self.mlp(txt_features)
        rewards = (rewards - self.mean) / self.std

        return rewards.detach().cpu().numpy().item()

    def score_batched(self, prompts, images):
        assert isinstance(prompts, list)
        assert isinstance(images, list)

        # text encode
        text_input = self.blip.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(self.device)

        # image encode
        images = [
            self.preprocess(image).unsqueeze(0).to(self.device) for image in images
        ]
        images = torch.cat(images, 0).to(self.device)

        image_embeds = self.blip.visual_encoder(images)

        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )
        text_output = self.blip.text_encoder(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        txt_features = text_output.last_hidden_state[:, 0, :].float()  # (feature_dim)
        rewards = self.mlp(txt_features)
        rewards = (rewards - self.mean) / self.std

        return rewards.view(txt_features.shape[0]).detach().cpu().numpy().tolist()

    def score_from_prompt_batched(self, prompts, images):
        assert isinstance(prompts, list)
        assert isinstance(images, list)

        # text encode
        text_input = self.blip.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(self.device)

        # image encode
        images = [
            self.preprocess_tensor(image).unsqueeze(0).to(self.device)
            for image in images
        ]
        images = torch.cat(images, 0).to(self.device).float()
        image_embeds = self.blip.visual_encoder(images)

        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )
        text_output = self.blip.text_encoder(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        txt_features = text_output.last_hidden_state[:, 0, :].float()  # (feature_dim)
        rewards = self.mlp(txt_features)
        rewards = (rewards - self.mean) / self.std

        return rewards.view(txt_features.shape[0])

    def inference_rank(self, prompt, generations_list):
        text_input = self.blip.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(self.device)
        txt_set = []
        for generation in generations_list:
            # image encode
            if isinstance(generation, Image.Image):
                pil_image = generation
            elif isinstance(generation, str):
                if os.path.isfile(generation):
                    pil_image = Image.open(generation)
            else:
                raise TypeError(
                    r"This image parameter type has not been supportted yet. Please pass PIL.Image or file path str."
                )
            image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            image_embeds = self.blip.visual_encoder(image)

            # text encode cross attention with image
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                self.device
            )
            text_output = self.blip.text_encoder(
                text_input.input_ids,
                attention_mask=text_input.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            txt_set.append(text_output.last_hidden_state[:, 0, :])

        txt_features = torch.cat(txt_set, 0).float()  # [image_num, feature_dim]
        rewards = self.mlp(txt_features)  # [image_num, 1]
        rewards = (rewards - self.mean) / self.std
        rewards = torch.squeeze(rewards)
        _, rank = torch.sort(rewards, dim=0, descending=True)
        _, indices = torch.sort(rank, dim=0)
        indices = indices + 1

        return (
            indices.detach().cpu().numpy().tolist(),
            rewards.detach().cpu().numpy().tolist(),
        )


def rm_load(
    name: str = "ImageReward-v1.0",
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    download_root: str = None,
    med_config: str = None,
):
    """Load a ImageReward model

    Parameters
    ----------
    name : str
        A model name listed by `ImageReward.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    download_root: str
        path to download the model files; by default, it uses "~/.cache/ImageReward"

    Returns
    -------
    model : torch.nn.Module
        The ImageReward model
    """
    if name in RM.utils._MODELS:
        model_path = RM.ImageReward_download(
            RM.utils._MODELS[name],
            download_root or os.path.expanduser("~/.cache/ImageReward"),
        )
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found;")

    print("load checkpoint from %s" % model_path)
    state_dict = torch.load(model_path, map_location="cpu")

    # med_config
    if med_config is None:
        med_config = RM.ImageReward_download(
            "https://huggingface.co/THUDM/ImageReward/blob/main/med_config.json",
            download_root or os.path.expanduser("~/.cache/ImageReward"),
        )

    model = IRSMC(device=device, med_config=med_config).to(device)
    msg = model.load_state_dict(state_dict, strict=False)
    print("checkpoint loaded")
    model.eval()

    return model


REWARDS_DICT = {
    "Clip-Score": None,
    "ImageReward": None,
}


def get_reward_function(reward_name, images, prompts):
    if reward_name == "ImageReward":
        return do_image_reward(images=images, prompts=prompts)


# Compute ImageReward
def do_image_reward(
    *, images, prompts, use_no_grad=True, use_score_from_prompt_batched=False
):
    global REWARDS_DICT
    if REWARDS_DICT["ImageReward"] is None:
        REWARDS_DICT["ImageReward"] = rm_load("ImageReward-v1.0")

    context = torch.no_grad() if use_no_grad else nullcontext()

    with context:
        if use_score_from_prompt_batched:
            image_reward_result = REWARDS_DICT["ImageReward"].score_from_prompt_batched(
                prompts, images
            )
        else:
            image_reward_result = REWARDS_DICT["ImageReward"].score_batched(
                prompts, images
            )

    return image_reward_result


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
