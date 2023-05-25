import clip
from imagenet_prompts_cn.standard_prompts_cn import standard_templates_cn
from CIFAR10_dataset import CIFAR10Dataset
import json
from tqdm import tqdm

import torch

import cn_clip.clip as cn_clip
from cn_clip.clip import load_from_name as load_cn_clip_from_name


def zero_shot_classifier_CN_clip(classnames, textnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        i = 0
        for classname in tqdm(classnames):
            texts = [template.format(textnames[i]) for template in templates]  # format with class
            texts = cn_clip.tokenize(texts).cuda()  # tokenize
            print(texts)
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)

            i += 1
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def zeroshot_classifier(classnames, textnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        i = 0
        for classname in tqdm(classnames):
            texts = [template.format(textnames[i]) for template in templates]  # format with class
            print(texts)
            texts = clip.tokenize(texts).cuda()  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)

            i += 1
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def zeroshot_classifier_gpt(classnames, textnames, templates, use_both, PATH_TO_PROMPTS, model):
    with open(PATH_TO_PROMPTS) as f:
        gpt3_prompts = json.load(f)

    with torch.no_grad():
        zeroshot_weights = []
        i = 0
        for classname in tqdm(classnames):
            if use_both:
                texts = [template.format(textnames[i]) for template in templates]
            else:
                texts = []

            for t in gpt3_prompts[textnames[i]]:
                texts.append(t)
            texts = clip.tokenize(texts, truncate=True).cuda()  # tokenize
            print(texts)
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
            i += 1

        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def zeroshot_classifier_gpt_CN(classnames, textnames, templates, use_both, PATH_TO_PROMPTS, model):
    with open(PATH_TO_PROMPTS, encoding="utf-8") as f:
        gpt3_prompts = json.load(f)

    with torch.no_grad():
        zeroshot_weights = []
        i = 0
        for classname in tqdm(classnames):
            if use_both:
                texts = [template.format(textnames[i]) for template in templates]
            else:
                texts = []

            for t in gpt3_prompts[textnames[i]]:
                texts.append(t)
            print(texts)
            texts = cn_clip.tokenize(texts).cuda()  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
            i += 1

        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def main():
    PATH_TO_CIFAR10 = "dataset/cifar-10-python/cifar-10-batches-py"
    PATH_TO_PROMPTS = "CIFAR10_prompts/CIFAR10_gpt_CN.json"

    # model, preprocess = clip.load("ViT-L/14")
    # model, preprocess = clip.load("RN50")
    # model.eval()
    model, preprocess = load_cn_clip_from_name("RN50", device="cuda")
    model.eval()

    all_images = CIFAR10Dataset(PATH_TO_CIFAR10, transform=preprocess)
    loader = torch.utils.data.DataLoader(all_images, batch_size=128, num_workers=4)

    print("\nCreating standard text embeddings...")
    # zeroshot_weights_base = zeroshot_classifier(all_images.idx_to_label, all_images.idx_to_text, imagenet_templates, model)
    zeroshot_weights_base = zero_shot_classifier_CN_clip(all_images.idx_to_label, all_images.idx_to_text,
                                                         CIFAR10_templates,
                                                         model)
    print("Done.\n")

    print("Creating CuPL text embeddings...")
    zeroshot_weights_cupl = zeroshot_classifier_gpt_CN(all_images.idx_to_label, all_images.idx_to_text,
                                                       CIFAR10_templates,
                                                       False, PATH_TO_PROMPTS, model)
    print("Done.\n")

    print("Creating combined text embeddings...")
    zeroshot_weights_gpt_both = zeroshot_classifier_gpt_CN(all_images.idx_to_label, all_images.idx_to_text,
                                                           CIFAR10_templates, True, PATH_TO_PROMPTS, model)
    print("Done.\n")

    total = 0.
    correct_base = 0.
    correct_cupl = 0.
    correct_both = 0.

    print("Classifying CIFAR...")
    with torch.no_grad():

        for i, (images, target, num) in enumerate(tqdm(loader)):
            images = images.cuda()
            target = target.cuda()

            # predict
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            logits_base = image_features @ zeroshot_weights_base
            logits_cupl = image_features @ zeroshot_weights_cupl
            logits_both = image_features @ zeroshot_weights_gpt_both

            pred_base = torch.argmax(logits_base, dim=1)
            pred_cupl = torch.argmax(logits_cupl, dim=1)
            pred_both = torch.argmax(logits_both, dim=1)

            for j in range(len(target)):
                total += 1.
                if pred_base[j] == target[j]:
                    correct_base += 1.
                if pred_cupl[j] == target[j]:
                    correct_cupl += 1.
                if pred_both[j] == target[j]:
                    correct_both += 1.

    print()
    top1 = (correct_base / total) * 100
    print(f"Top-1 accuracy standard: {top1:.2f}")

    top1 = (correct_cupl / total) * 100
    print(f"Top-1 accuracy CuPL: {top1:.2f}")

    top1 = (correct_both / total) * 100
    print(f"Top-1 accuracy both: {top1:.2f}")


if __name__ == "__main__":
    main()
