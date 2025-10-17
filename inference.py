

'''
from argparse import ArgumentParser
from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM
import PIL.Image

from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
from deepseek_vl2.serve.app_modules.utils import parse_ref_bbox
from PIL import Image
from typing import List

def load_pil_images(image_paths: List[str]) -> List[Image.Image]:
    """
    Args:
        image_paths (List[str]): list of image file paths.

    Returns:
        pil_images (List[Image.Image]): list of loaded PIL images.
    """
    pil_images = []

    for image_path in image_paths:
        pil_img = Image.open(image_path)
        pil_img = pil_img.convert("RGB")
        pil_images.append(pil_img)

    return pil_images
from PIL import Image
from typing import List

def load_pil_images(image_paths: List[str]) -> List[Image.Image]:
    """
    Args:
        image_paths (List[str]): list of image file paths.

    Returns:
        pil_images (List[Image.Image]): list of loaded PIL images.
    """
    pil_images = []

    for image_path in image_paths:
        pil_img = Image.open(image_path)
        pil_img = pil_img.convert("RGB")
        pil_images.append(pil_img)

    return pil_images


def funsd_inference(
    annotations_dir: str,
    images_root: str,
    model_path: str,
    chunk_size: int = -1,
    output_file: str = None,
    track_cka: bool = False,  # New flag for CKA tracking
    cka_output_dir: str = "cka_results"  # Directory for CKA results
):
    dtype = torch.bfloat16

    # Load model and processor once
    vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto"
    )
    vl_gpt = vl_gpt.cuda().eval()
   

    for layer in vl_gpt.language.model.layers:
        if hasattr(layer, "mlp"):
        # Optional flag if you have tracking
            layer.mlp.cka_tracking = True
        # Only for MoE
            if hasattr(layer.mlp, "experts"):
                layer.mlp._expert_activations = [None] * len(layer.mlp.experts)

    results = []

    # Iterate over all JSON annotation files in annotations_dir
    for ann_file in os.listdir(annotations_dir):
        if not ann_file.endswith(".json"):
            continue

        ann_path = os.path.join(annotations_dir, ann_file)
        with open(ann_path, "r") as f:
            funsd_data = json.load(f)

        # derive image file name from JSON file name
        image_filename = os.path.splitext(ann_file)[0] + ".png"
        image_path = os.path.join(images_root, image_filename)

        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found, skipping")
            continue

        prompt = "Extract all form fields and their labels from this document."

        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image>\n<|grounding|>{prompt}",
                "images": [image_path]
            },
            {"role": "<|Assistant|>", "content": ""}
        ]

        pil_images = load_pil_images([image_path])

        prepare_inputs = vl_chat_processor.__call__(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        ).to(vl_gpt.device, dtype=dtype)

        with torch.no_grad():
            if chunk_size == -1:
                inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
                past_key_values = None
            else:
                inputs_embeds, past_key_values = vl_gpt.incremental_prefilling(
                    input_ids=prepare_inputs.input_ids,
                    images=prepare_inputs.images,
                    images_seq_mask=prepare_inputs.images_seq_mask,
                    images_spatial_crop=prepare_inputs.images_spatial_crop,
                    attention_mask=prepare_inputs.attention_mask,
                    chunk_size=chunk_size
                )

            outputs = vl_gpt.generate(
                inputs_embeds=inputs_embeds,
                input_ids=prepare_inputs.input_ids,
                images=prepare_inputs.images,
                images_seq_mask=prepare_inputs.images_seq_mask,
                images_spatial_crop=prepare_inputs.images_spatial_crop,
                attention_mask=prepare_inputs.attention_mask,
                past_key_values=past_key_values,

                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=512,

                do_sample=True,
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.1,
                use_cache=True,
            )

            answer = tokenizer.decode(
                outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(),
                skip_special_tokens=False
            )

        
        results.append({
            "file_name": image_filename,
            "predicted_answer": answer,
            "form_data": funsd_data
        })

        print(f"File: {image_filename}\nAnswer: {answer}\n")

    if output_file:
        with open(output_file, "w") as f_out:
            json.dump(results, f_out, indent=4)

    return results

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Model path or name")
    parser.add_argument("--annotations_dir", type=str, required=True, help="Directory with FUNSD JSON annotation files")
    parser.add_argument("--images_root", type=str, required=True, help="Root folder for images")
    parser.add_argument("--chunk_size", type=int, default=-1, help="Chunk size for incremental pre-filling")
    parser.add_argument("--output_file", type=str, default=None, help="Optional output file to save predictions")
    parser.add_argument("--track_cka", action="store_true", help="Enable CKA similarity tracking")
    parser.add_argument("--cka_output_dir", type=str, default="cka_results", help="Directory to save CKA results")
    args = parser.parse_args()

    funsd_inference(
        annotations_dir=args.annotations_dir,
        images_root=args.images_root,
        model_path=args.model_path,
        chunk_size=args.chunk_size,
        output_file=args.output_file,
        track_cka=args.track_cka,
        cka_output_dir=args.cka_output_dir
    )
'''
'''
    
from argparse import ArgumentParser
from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM
import PIL.Image

from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
from deepseek_vl2.serve.app_modules.utils import parse_ref_bbox

from deepseek_vl2.utils.cka_utils import compute_layer_cka
import json
import os
from argparse import ArgumentParser
from typing import List
import torch
from transformers import AutoModelForCausalLM
import PIL.Image

from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor

def load_pil_images(image_paths: List[str]) -> List[PIL.Image.Image]:
    pil_images = []
    for image_path in image_paths:
        pil_img = PIL.Image.open(image_path)
        pil_img = pil_img.convert("RGB")
        pil_images.append(pil_img)
    return pil_images

def main(args):

    dtype = torch.bfloat16

    # specify the path to the model
    model_path = args.model_path
    vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype
    )
    vl_gpt = vl_gpt.cuda().eval()
    # === CKA tracking: turn on and zero-init buffers ===
    for layer in vl_gpt.language.model.layers:
        if hasattr(layer, "mlp"):
            layer.mlp.cka_tracking = True
            if hasattr(layer.mlp, "experts"):
            
                layer.mlp._expert_activations = [None] * len(layer.mlp.experts)

    # print("Summarmmmmmmmmmmmmmmmmmmmmmm",vl_gpt)
           # multiple images conversation example
    # Please note that <|grounding|> token is specifically designed for the grounded caption feature. It is not needed for normal conversations.
    conversation = [
        {
            "role": "<|User|>",
            # "content": "<image>\n<image>\n<|grounding|>Explain image given below."
            "content": "<image>\n<|grounding|>2 words",

            "images": [
                "images/invoice.jpg"
                # "images/icl_vg_2.jpeg"
            ],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]


    # load images and prepare for inputs
    pil_images = load_pil_images(conversation)
    print(f"len(pil_images) = {len(pil_images)}")

    prepare_inputs = vl_chat_processor.__call__(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(vl_gpt.device, dtype=dtype)
    

    with torch.no_grad():

        if args.chunk_size == -1:
            inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
            past_key_values = None
        else:
            # incremental_prefilling when using 40G GPU for vl2-small
            inputs_embeds, past_key_values = vl_gpt.incremental_prefilling(
                input_ids=prepare_inputs.input_ids,
                images=prepare_inputs.images,
                images_seq_mask=prepare_inputs.images_seq_mask,
                images_spatial_crop=prepare_inputs.images_spatial_crop,
                attention_mask=prepare_inputs.attention_mask,
                chunk_size=args.chunk_size
            )

        # run the model to get the response
        outputs = vl_gpt.generate(
            # inputs_embeds=inputs_embeds[:, -1:],
            # input_ids=prepare_inputs.input_ids[:, -1:],
            inputs_embeds=inputs_embeds,
            input_ids=prepare_inputs.input_ids,
            images=prepare_inputs.images,
            images_seq_mask=prepare_inputs.images_seq_mask,
            images_spatial_crop=prepare_inputs.images_spatial_crop,
            attention_mask=prepare_inputs.attention_mask,
            past_key_values=past_key_values,

            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,

            # do_sample=False,
            # repetition_penalty=1.1,

            do_sample=True,
            temperature=0.4,
            top_p=0.9,
            repetition_penalty=1.1,

            use_cache=True,
        )
        from deepseek_vl2.utils.cka_utils import compute_layer_cka

        for idx, layer in enumerate(vl_gpt.language.model.layers):
    # only process layers with expert activations
          if not (hasattr(layer, 'mlp') and hasattr(layer.mlp, '_expert_activations')):
            continue

          acts_list = layer.mlp._expert_activations
    # skip if any expert didn't fire
          if any(x is None for x in acts_list):
            print(f"[CKA] Layer {idx}: missing activations, skipped.")
            continue

    # align sequences by truncating to the minimum length
          lengths = [act.shape[0] for act in acts_list]
          min_len = min(lengths)
          if min_len == 0:
            print(f"[CKA] Layer {idx}: no common tokens, skipped.")
            continue

          truncated = [act[:min_len] for act in acts_list]
          acts = torch.stack(truncated, dim=1)  # [min_len, n_experts, hidden_dim]

          cka_mat = compute_layer_cka(acts)
          print(f"[CKA] Layer {idx} similarity matrix:\n{cka_mat}")
    

        answer = tokenizer.decode(outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(), skip_special_tokens=False)
        print(f"{prepare_inputs['sft_format'][0]}", answer)
       
     



        vg_image = parse_ref_bbox(answer, image=pil_images[-1])
        if vg_image is not None:
            vg_image.save("./vg.jpg", format="JPEG", quality=85)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        default="deepseek-ai/deepseek-vl2",
                        help="model name or local path to the model")
    parser.add_argument("--chunk_size", type=int, default=-1,
                        help="chunk size for the model for prefiiling. "
                             "When using 40G gpu for vl2-small, set a chunk_size for incremental_prefilling."
                             "Otherwise, default value is -1, which means we do not use incremental_prefilling.")
    args = parser.parse_args()
    main(args)
    '''
    
    
    
''' 

import json
import os 


# -------------- HF cache redirection --------------
os.environ["HF_HOME"] = "/data/stud/raza/DeepSeek-VL2/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/data/stud/raza/DeepSeek-VL2/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/data/stud/raza/DeepSeek-VL2/hf_cache"
# --------------------------------------------------

from argparse import ArgumentParser
from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM
import PIL.Image

from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
from deepseek_vl2.serve.app_modules.utils import parse_ref_bbox
from PIL import Image
from typing import List
from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor

def load_pil_images(image_paths: List[str]) -> List[PIL.Image.Image]:
    pil_images = []
    for image_path in image_paths:
        pil_img = PIL.Image.open(image_path)
        pil_img = pil_img.convert("RGB")
        pil_images.append(pil_img)
    return pil_images

def funsd_inference(
    annotations_dir: str,
    images_root: str,
    model_path: str,
    chunk_size: int = -1,
    output_file: str = None
):
    dtype = torch.bfloat16

    # Load model and processor once
    vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_F="auto"
    )
    vl_gpt = vl_gpt.cuda().eval()
    for layer in vl_gpt.language.model.layers:
        if hasattr(layer, "mlp"):
            layer.mlp.cka_tracking = True
            if hasattr(layer.mlp, "experts"):
            
                layer.mlp._expert_activations = [None] * len(layer.mlp.experts)

    results = []

    # Iterate over all JSON annotation files in annotations_dir
    for ann_file in os.listdir(annotations_dir):
        if not ann_file.endswith(".json"):
            continue

        ann_path = os.path.join(annotations_dir, ann_file)
        with open(ann_path, "r") as f:
            funsd_data = json.load(f)

        # derive image file name from JSON file name, e.g., doc1.json -> doc1.png
        image_filename = os.path.splitext(ann_file)[0] + ".png"
        image_path = os.path.join(images_root, image_filename)

        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found, skipping")
            continue

        prompt = "Extract all form fields and their labels from this document."

        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image>\n<|grounding|>{prompt}",
                "images": [image_path]
            },
            {"role": "<|Assistant|>", "content": ""}
        ]

        pil_images = load_pil_images([image_path])

        prepare_inputs = vl_chat_processor.__call__(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        ).to(vl_gpt.device, dtype=dtype)

        with torch.no_grad():
            if chunk_size == -1:
                inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
                past_key_values = None
            else:
                inputs_embeds, past_key_values = vl_gpt.incremental_prefilling(
                    input_ids=prepare_inputs.input_ids,
                    images=prepare_inputs.images,
                    images_seq_mask=prepare_inputs.images_seq_mask,
                    images_spatial_crop=prepare_inputs.images_spatial_crop,
                    attention_mask=prepare_inputs.attention_mask,
                    chunk_size=chunk_size
                )

            outputs = vl_gpt.generate(
                inputs_embeds=inputs_embeds,
                input_ids=prepare_inputs.input_ids,
                images=prepare_inputs.images,
                images_seq_mask=prepare_inputs.images_seq_mask,
                images_spatial_crop=prepare_inputs.images_spatial_crop,
                attention_mask=prepare_inputs.attention_mask,
                past_key_values=past_key_values,

                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=512,

                do_sample=True,
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.1,
                use_cache=True,
            )
        from deepseek_vl2.utils.cka_utils import compute_layer_cka

        for idx, layer in enumerate(vl_gpt.language.model.layers):
    # only process layers with expert activations
          if not (hasattr(layer, 'mlp') and hasattr(layer.mlp, '_expert_activations')):
            continue

          acts_list = layer.mlp._expert_activations
    # skip if any expert didn't fire
          if any(x is None for x in acts_list):
            print(f"[CKA] Layer {idx}: missing activations, skipped.")
            continue

    # align sequences by truncating to the minimum length
          lengths = [act.shape[0] for act in acts_list]
          min_len = min(lengths)
          if min_len == 0:
            print(f"[CKA] Layer {idx}: no common tokens, skipped.")
            continue

          truncated = [act[:min_len] for act in acts_list]
          acts = torch.stack(truncated, dim=1)  # [min_len, n_experts, hidden_dim]

          cka_mat = compute_layer_cka(acts)
          print(f"[CKA] Layer {idx} similarity matrix:\n{cka_mat}")
    

        answer = tokenizer.decode(outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(), skip_special_tokens=False)
        print(f"{prepare_inputs['sft_format'][0]}", answer)
       
     



        vg_image = parse_ref_bbox(answer, image=pil_images[-1])
        if vg_image is not None:
            vg_image.save("./vg.jpg", format="JPEG", quality=85)


            answer = tokenizer.decode(
                outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(),
                skip_special_tokens=False
            )

        results.append({
            "file_name": image_filename,
            "predicted_answer": answer,
            "form_data": funsd_data  # including the raw form data if needed
        })

        print(f"File: {image_filename}\nAnswer: {answer}\n")

    if output_file:
        with open(output_file, "w") as f_out:
            json.dump(results, f_out, indent=4)

    return results


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Model path or name")
    parser.add_argument("--annotations_dir", type=str, required=True, help="Directory with FUNSD JSON annotation files")
    parser.add_argument("--images_root", type=str, required=True, help="Root folder for images")
    parser.add_argument("--chunk_size", type=int, default=-1, help="Chunk size for incremental pre-filling")
    parser.add_argument("--output_file", type=str, default=None, help="Optional output file to save predictions")
    args = parser.parse_args()

    funsd_inference(
        annotations_dir=args.annotations_dir,
        images_root=args.images_root,
        model_path=args.model_path,
        chunk_size=args.chunk_size,
        output_file=args.output_file,
    )
'''

'''


import os
import json
from argparse import ArgumentParser
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import PIL.Image
from PIL import Image
from datasets import load_dataset
import evaluate

from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
from deepseek_vl2.serve.app_modules.utils import parse_ref_bbox
from deepseek_vl2.utils.cka_utils import (
    compute_layer_cka,
    save_cka_heatmaps,
    find_similar_experts
)
from deepseek_vl2.utils.expert_prune_merge import (
    prune_and_merge_experts,
    count_model_parameters
   
)

def evaluate_mmlu(model_path):
    """
    Dummy example: loads model and evaluates on MMLU subset.
    Replace this with your actual eval code.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Example: load MMLU dataset
    dataset = load_dataset("hendrycks/test", "abstract_algebra", split="test[:100]")  # change split

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    correct = 0
    for example in dataset:
        input_text = example['question']
        output = pipe(input_text, max_length=50)[0]['generated_text']
        if example['answer'] in output:
            correct += 1
    acc = correct / len(dataset)
    return acc

def load_pil_images(image_paths: List[str]) -> List[PIL.Image.Image]:
    pil_images = []
    for image_path in image_paths:
        pil_img = Image.open(image_path).convert("RGB")
        pil_images.append(pil_img)
    return pil_images


def funsd_inference(
    annotations_dir: str,
    images_root: str,
    processor: DeepseekVLV2Processor,
    model: DeepseekVLV2ForCausalLM,
    chunk_size: int,
    num_prompts_for_cka: int,
    heatmap_output_dir: str
):
    """
    Runs FUNSD form inference, tracks expert activations for CKA,
    then computes and saves CKA heatmaps.
    Returns:
      - all_cka: dict[layer_idx] -> CKA matrix (torch.Tensor)
    """
    dtype = torch.bfloat16
    # Enable CKA tracking
    for layer in model.language.model.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            layer.mlp.cka_tracking = True
            layer.mlp._expert_activations = [None] * len(layer.mlp.experts)

    # Prepare storage for activations
    all_acts: Dict[int, List[List[torch.Tensor]]] = {
        idx: [[] for _ in layer.mlp.experts]
        for idx, layer in enumerate(model.language.model.layers)
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts")
    }

    processed = 0
    # FUNSD inference loop
    for ann_file in sorted(os.listdir(annotations_dir)):
        if not ann_file.endswith(".json"):
            continue
        with open(os.path.join(annotations_dir, ann_file)) as f:
            funsd_data = json.load(f)

        image_name = os.path.splitext(ann_file)[0] + ".png"
        img_path = os.path.join(images_root, image_name)
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found, skipping")
            continue

        # Build conversation
        prompt = "Extract all form fields and their labels from this document."
        convo = [
            {"role": "<|User|>", "content": f"<image>\n<|grounding|>{prompt}", "images": [img_path]},
            {"role": "<|Assistant|>", "content": ""}
        ]
        pil_imgs = load_pil_images([img_path])
        inputs = processor(
            conversations=convo,
            images=pil_imgs,
            force_batchify=True,
            system_prompt=""
        ).to(model.device, dtype=dtype)

        # Forward + generate
        with torch.no_grad():
            if chunk_size == -1:
                inputs_embeds = model.prepare_inputs_embeds(**inputs)
                pkv = None
            else:
                inputs_embeds, pkv = model.incremental_prefilling(
                    input_ids=inputs.input_ids,
                    images=inputs.images,
                    images_seq_mask=inputs.images_seq_mask,
                    images_spatial_crop=inputs.images_spatial_crop,
                    attention_mask=inputs.attention_mask,
                    chunk_size=chunk_size
                )
            outputs = model.generate(
                inputs_embeds=inputs_embeds,
                input_ids=inputs.input_ids,
                images=inputs.images,
                images_seq_mask=inputs.images_seq_mask,
                images_spatial_crop=inputs.images_spatial_crop,
                attention_mask=inputs.attention_mask,
                past_key_values=pkv,
                pad_token_id=processor.tokenizer.eos_token_id,
                bos_token_id=processor.tokenizer.bos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.1,
                use_cache=True,
            )

        # Collect expert activations
        if processed < num_prompts_for_cka:
            for idx, layer in enumerate(model.language.model.layers):
                if not hasattr(layer.mlp, "_expert_activations"):
                    continue
                acts_list = layer.mlp._expert_activations
                for e, act in enumerate(acts_list):
                    if act is not None and act.numel() > 0:
                        all_acts[idx][e].append(act.detach().cpu().float())
        processed += 1

    # Compute CKA matrices
    all_cka: Dict[int, torch.Tensor] = {}
    for idx, act_lists in all_acts.items():
        mats = []
        for expert_list in act_lists:
            if expert_list:
                mats.append(torch.cat(expert_list, dim=0))
            else:
                mats.append(torch.empty((0, model.config.hidden_size)))
        counts = [m.shape[0] for m in mats]
        Nmin = min(counts) if counts else 0
        if Nmin == 0:
            continue
        batch = torch.stack([m[:Nmin] for m in mats], dim=1)  # [Nmin, E, D]
        all_cka[idx] = compute_layer_cka(batch)

    # Save heatmaps
    save_cka_heatmaps(all_cka, output_dir=heatmap_output_dir)
    # Print similar experts
    similar = find_similar_experts(all_cka, threshold=0.7)
    for layer_idx, pairs in similar.items():
        print(f"Layer {layer_idx} similar experts:", pairs)

    return all_cka


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--annotations_dir", required=True)
    parser.add_argument("--images_root", required=True)
    parser.add_argument("--chunk_size", type=int, default=-1)
    parser.add_argument("--num_prompts_for_cka", type=int, default=200)
    parser.add_argument("--heatmap_output_dir", default="cka_heatmaps")

    # New flags
    parser.add_argument("--do_prune", action="store_true",
                        help="Prune & merge experts after CKA")
    parser.add_argument("--prune_threshold", type=float, default=0.7)
    parser.add_argument("--prune_ratio", type=float, default=0.5)
    parser.add_argument("--skip_delta", type=float, default=0.1)
    parser.add_argument("--eval_mmlu", action="store_true",
                        help="Evaluate MMLU before/after pruning")
    parser.add_argument("--pruned_model_dir", default="pruned_model",
                        help="Where to save the pruned model for eval")

    args = parser.parse_args()

    # Load processor & model
    processor = DeepseekVLV2Processor.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).cuda().eval()

    # Optional MMLU on original
    if args.eval_mmlu:
        print("Evaluating MMLU on original model   ")
        orig_mmlu = evaluate_mmlu(args.model_path)
        print('Original MMLU:', orig_mmlu)

    # Count original params
    orig_params = count_model_parameters(model)
    print(f">>> Original #params: {orig_params:,}")

    # Run FUNSD inference + CKA
    all_cka = funsd_inference(
        annotations_dir=args.annotations_dir,
        images_root=args.images_root,
        processor=processor,
        model=model,
        chunk_size=args.chunk_size,
        num_prompts_for_cka=args.num_prompts_for_cka,
        heatmap_output_dir=args.heatmap_output_dir
    )

    # Optional prune & merge
    if args.do_prune:
        print(" Pruning & merging experts")
        stats = prune_and_merge_experts(
            model,
            all_cka,
            similarity_threshold=args.prune_threshold,
            max_prune_ratio=args.prune_ratio,
            dynamic_skip_delta=args.skip_delta
        )
        pruned_params = count_model_parameters(model)
        print(f">>> Pruned #params:  {pruned_params:,}")

        # Save pruned model for evaluation
        os.makedirs(args.pruned_model_dir, exist_ok=True)
        model.save_pretrained(args.pruned_model_dir)
        processor.save_pretrained(args.pruned_model_dir)

        if args.eval_mmlu:
            print("Evaluating MMLU on pruned model")
            pruned_mmlu = evaluate_mmlu(args.pruned_model_dir)
            print("Pruned MMLU:", pruned_mmlu)

    print("Done.")
'''
'''

import json
import os
import json
import os 


# -------------- HF cache redirection --------------
os.environ["HF_HOME"] = "/data/stud/raza/DeepSeek-VL2/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/data/stud/raza/DeepSeek-VL2/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/data/stud/raza/DeepSeek-VL2/hf_cache"
# --------------------------------------------------

from argparse import ArgumentParser
from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM
import PIL.Image
from PIL import Image

from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
from deepseek_vl2.serve.app_modules.utils import parse_ref_bbox
from deepseek_vl2.utils.cka_utils import compute_layer_cka, save_cka_heatmaps
from deepseek_vl2.utils.cka_utils import find_similar_experts
def load_pil_images(image_paths: List[str]) -> List[PIL.Image.Image]:
    pil_images = []
    for image_path in image_paths:
        pil_img = PIL.Image.open(image_path).convert("RGB")
        pil_images.append(pil_img)
    return pil_images


def funsd_inference(
    annotations_dir: str,
    images_root: str,
    model_path: str,
    chunk_size: int = -1,
    output_file: str = None,
    num_prompts_for_cka: int = 200,
    heatmap_output_dir: str = "cka_heatmaps"
):
    dtype = torch.bfloat16

    # Load processor and model
    processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype
    ).cuda().eval()
    print(model.config.vision_config)

  
    # Enable CKA tracking in MoE layers
    for layer in model.language.model.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            layer.mlp.cka_tracking = True
            layer.mlp._expert_activations = [None] * len(layer.mlp.experts)

    # Prepare storage for accumulating activations per MoE layer & expert
    all_acts: Dict[int, List[List[torch.Tensor]]] = {}
    for idx, layer in enumerate(model.language.model.layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            all_acts[idx] = [[] for _ in layer.mlp.experts]

    results = []
    processed = 0

    # Loop over annotation files
    for ann_file in sorted(os.listdir(annotations_dir)):
        if not ann_file.endswith(".json"):
            continue
        ann_path = os.path.join(annotations_dir, ann_file)
        with open(ann_path) as f:
            funsd_data = json.load(f)

        image_name = os.path.splitext(ann_file)[0] + ".png"
        img_path = os.path.join(images_root, image_name)
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found, skipping")
            continue

        # Build prompt conversation
        prompt = "Extract all form fields and their labels from this document"
        conversation = [
            {"role": "<|User|>", "content": f"<image>\n<|grounding|>{prompt}", "images": [img_path]},
            {"role": "<|Assistant|>", "content": ""}
        ]
        pil_imgs = load_pil_images([img_path])
        prepare = processor(
            conversations=conversation,
            images=pil_imgs,
            force_batchify=True,
            system_prompt=""
        ).to(model.device, dtype=dtype)

        # Forward + generate
        with torch.no_grad():
            if chunk_size == -1:
                inputs_embeds = model.prepare_inputs_embeds(**prepare)
                pkv = None
            else:
                inputs_embeds, pkv = model.incremental_prefilling(
                    input_ids=prepare.input_ids,
                    images=prepare.images,
                    images_seq_mask=prepare.images_seq_mask,
                    images_spatial_crop=prepare.images_spatial_crop,
                    attention_mask=prepare.attention_mask,
                    chunk_size=chunk_size
                )
            outputs = model.generate(
                inputs_embeds=inputs_embeds,
                input_ids=prepare.input_ids,
                images=prepare.images,
                images_seq_mask=prepare.images_seq_mask,
                images_spatial_crop=prepare.images_spatial_crop,
                attention_mask=prepare.attention_mask,
                past_key_values=pkv,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.1,
                use_cache=True,
            )

        # Decode and print answer
        answer = tokenizer.decode(
            outputs[0][prepare.input_ids.size(1):].cpu().tolist(),
            skip_special_tokens=False
        )
        print(f"{prepare['sft_format'][0]} {answer}")

        # Store result
        results.append({
            "file_name": image_name,
            "predicted_answer": answer,
            "form_data": funsd_data
        })

        # Accumulate activations for CKA
        if processed < num_prompts_for_cka:
            for idx, expert_lists in all_acts.items():
                acts_list = model.language.model.layers[idx].mlp._expert_activations
                for e, act in enumerate(acts_list):
                    if act is not None and act.numel() > 0:
                        expert_lists[e].append(act.detach().cpu().float())
        processed += 1

    # Compute & print CKA matrices per layer
    print("\n=== Expert vs Expert CKA per layer ===")
    all_cka = {}
    for idx, expert_lists in all_acts.items():
        mats = []
        for expert_list in expert_lists:
            if expert_list:
                mats.append(torch.cat(expert_list, dim=0))
            else:
                mats.append(torch.empty((0, model.config.vision_config.width)))
        counts = [m.shape[0] for m in mats]
        Nmin = min(counts) if counts else 0
        if Nmin == 0:
            print(f"Layer {idx}: insufficient activations, skipped")
            continue
        batch = torch.stack([m[:Nmin] for m in mats], dim=1)  # [Nmin, E, D]
        cka = compute_layer_cka(batch)
        all_cka[idx] = cka
        print(f"Layer {idx} CKA matrix:\n{cka}\n")

    # Save heatmaps
    save_cka_heatmaps(all_cka, output_dir=heatmap_output_dir)
    similar_pairs = find_similar_experts(all_cka, threshold=0.7)
    for layer, pairs in similar_pairs.items():
      print(f"Layer {layer} similar experts:", pairs)

    # Optionally save results to JSON
    if output_file:
        with open(output_file, "w") as fout:
            json.dump(results, fout, indent=4)

    return results

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--annotations_dir", required=True)
    parser.add_argument("--images_root", required=True)
    parser.add_argument("--chunk_size", type=int, default=-1)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--num_prompts_for_cka", type=int, default=200,
                        help="Number of samples to accumulate for CKA")
    parser.add_argument("--heatmap_output_dir", default="cka_heatmaps",
                        help="Directory to save CKA heatmaps")
    args = parser.parse_args()

    funsd_inference(
        annotations_dir=args.annotations_dir,
        images_root=args.images_root,
        model_path=args.model_path,
        chunk_size=args.chunk_size,
        output_file=args.output_file,
        num_prompts_for_cka=args.num_prompts_for_cka,
        heatmap_output_dir=args.heatmap_output_dir
    )
   

'''
'''
import json
import os
from argparse import ArgumentParser
from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
from deepseek_vl2.utils.cka_utils import compute_layer_cka, save_cka_heatmaps, find_similar_experts

def docvqa_inference(
    annotations_file: str,
    contents_file: str,
    model_path: str,
    chunk_size: int = -1,
    output_file: str = None,
    num_prompts_for_cka: int = 200,
    heatmap_output_dir: str = "cka_heatmaps"
):
    dtype = torch.bfloat16

    # Load processor and model
    processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype
    ).cuda().eval()

    # Enable CKA tracking
    for layer in model.language.model.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            layer.mlp.cka_tracking = True
            layer.mlp._expert_activations = [None] * len(layer.mlp.experts)

    # Prepare storage for activations
    all_acts: Dict[int, List[torch.Tensor]] = {}
    for idx, layer in enumerate(model.language.model.layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            all_acts[idx] = []

    # Load annotations
    ann_map: Dict[str, Dict] = {}
    with open(annotations_file, 'r') as f_ann:
        for line in f_ann:
            doc = json.loads(line)
            ann_map[doc['name']] = doc

    # Load contents
    contents_map: Dict[str, List[Dict]] = {}
    with open(contents_file, 'r') as f_cont:
        for line in f_cont:
            cont = json.loads(line)
            contents_map[cont['name']] = cont.get('contents', [])

    results = []
    processed = 0

    # Iterate over documents
    for name, ann_doc in ann_map.items():
        annotations = ann_doc.get('annotations', [])
        tokens_layer = contents_map.get(name, [])
        if not tokens_layer:
            print(f"Warning: no contents for {name}, skipping")
            continue

        # Process each question
        for ann in annotations:
            question = ann.get('key', '')
            values = ann.get('values', [])

            # Extract words and boxes
            words, boxes = [], []
            for item in tokens_layer:
                if 'tokens_layer' in item:
                    txt = item.get('text', '')
                    if isinstance(txt, str):
                        words.extend(txt.split())
                    boxes.extend(item['tokens_layer'].get('positions', []))
                elif 'common_format' in item:
                    cf = item['common_format']
                    words.extend(cf.get('tokens', []))
                    boxes.extend(cf.get('positions', []))

            # Build conversation for layout-only
            conversation = [
                {
                    'role': '<|User|>',
                    'content': '<|layout_text|>\n<|grounding|>' + question,
                    'images': [],
                    'words': words,
                    'boxes': boxes
                },
                {'role': '<|Assistant|>', 'content': ''}
            ]
            prepare = processor(
                conversations=conversation,
                images=[],
                force_batchify=True,
                system_prompt=""
            ).to(model.device, dtype=dtype)

            # Generate answer
            with torch.no_grad():
                if chunk_size == -1:
                    inputs_embeds = model.prepare_inputs_embeds(**prepare)
                    pkv = None
                else:
                    inputs_embeds, pkv = model.incremental_prefilling(
                        input_ids=prepare.input_ids,
                        images=prepare.images,
                        images_seq_mask=prepare.images_seq_mask,
                        images_spatial_crop=prepare.images_spatial_crop,
                        attention_mask=prepare.attention_mask,
                        chunk_size=chunk_size
                    )
                outputs = model.generate(
                    inputs_embeds=inputs_embeds,
                    input_ids=prepare.input_ids,
                    images=prepare.images,
                    images_seq_mask=prepare.images_seq_mask,
                    images_spatial_crop=prepare.images_spatial_crop,
                    attention_mask=prepare.attention_mask,
                    past_key_values=pkv,
                    pad_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=256,
                    do_sample=False,
                    use_cache=True
                )
            answer = tokenizer.decode(
                outputs[0][prepare.input_ids.size(1):].cpu().tolist(),
                skip_special_tokens=True
            )

            # Print answer
            print(f"{name} | {question} -> {answer}")

            # Accumulate CKA activations
            if processed < num_prompts_for_cka:
                for idx, layer in enumerate(model.language.model.layers):
                    if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
                        for a in (layer.mlp._expert_activations or []):
                            if a is not None:
                                all_acts[idx].append(a.detach().cpu())
                processed += 1

            results.append({
                'doc': name,
                'question': question,
                'predicted_answer': answer,
                'ground_truth': [v.get('value') for v in values]
            })

    # After processing all documents, compute CKA
    all_cka = {}
    for idx, acts in all_acts.items():
        if not acts:
            continue
        try:
            batch = torch.stack(acts, dim=1)
            all_cka[idx] = compute_layer_cka(batch)
        except Exception as e:
            print(f"Skipping CKA for layer {idx} due to shape mismatch: {e}")
    if all_cka:
        save_cka_heatmaps(all_cka, output_dir=heatmap_output_dir)

        # Similar expert pairs
        similar = find_similar_experts(all_cka, threshold=0.7)
        for layer, pairs in similar.items():
            print(f"Layer {layer} similar experts: {pairs}")

    # Save JSON results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)

    return results



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--docs_file', default="/data/stud/raza/DeepSeek-VL2/DocVQA_dataset/aws_neurips_time/docvqa/test/document.jsonl", help='Path to document.jsonl')
    parser.add_argument('--content_file', default="/data/stud/raza/DeepSeek-VL2/DocVQA_dataset/aws_neurips_time/docvqa/test/documents_content.jsonl", help='Path to document_content.jsonl')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--chunk_size', type=int, default=-1)
    parser.add_argument('--output_file', default=None)
    parser.add_argument('--num_prompts_for_cka', type=int, default=500)
    parser.add_argument('--heatmap_output_dir', default='cka_heatmaps')
    args = parser.parse_args()

    docvqa_inference(
        annotations_file=args.docs_file,
        contents_file=args.content_file,
        model_path=args.model_path,
        chunk_size=args.chunk_size,
        output_file=args.output_file,
        num_prompts_for_cka=args.num_prompts_for_cka,
        heatmap_output_dir=args.heatmap_output_dir
    )
'''

'''
import json
import os
from argparse import ArgumentParser
from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
from deepseek_vl2.utils.cka_utils import compute_layer_cka, save_cka_heatmaps, find_similar_experts

def docvqa_inference(
    annotations_file: str,
    contents_file: str,
    model_path: str,
    chunk_size: int = -1,
    output_file: str = None,
    num_prompts_for_cka: int = 200,
    heatmap_output_dir: str = "cka_heatmaps"
):
    dtype = torch.bfloat16

    # Load processor and model
    processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype
    ).cuda().eval()

    # Enable CKA tracking
    for layer in model.language.model.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            layer.mlp.cka_tracking = True
            layer.mlp._expert_activations = [None] * len(layer.mlp.experts)

    # Prepare storage for activations
    all_acts: Dict[int, List[torch.Tensor]] = {}
    for idx, layer in enumerate(model.language.model.layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            all_acts[idx] = []

    # Load annotations
    with open(annotations_file, 'r') as f_ann:
        ann_map: Dict[str, Dict] = {
            doc['name']: doc for doc in map(json.loads, f_ann)
        }

    # Load contents
    with open(contents_file, 'r') as f_cont:
        contents_map: Dict[str, List[Dict]] = {
            cont['name']: cont.get('contents', []) for cont in map(json.loads, f_cont)
        }

    results = []
    processed = 0

    # Iterate over documents
    for name, ann_doc in ann_map.items():
        annotations = ann_doc.get('annotations', [])
        tokens_layer = contents_map.get(name, [])
        if not tokens_layer:
            print(f"Warning: no contents for {name}, skipping")
            continue

        for ann in annotations:
            question = ann.get('key', '')
            values = ann.get('values', [])

            # Extract words and boxes
            words, boxes = [], []
            for item in tokens_layer:
                if 'tokens_layer' in item:
                    txt = item.get('text', '')
                    if isinstance(txt, str):
                        words.extend(txt.split())
                    boxes.extend(item['tokens_layer'].get('positions', []))
                elif 'common_format' in item:
                    cf = item['common_format']
                    words.extend(cf.get('tokens', []))
                    boxes.extend(cf.get('positions', []))

            conversation = [
                {
                    'role': '<|User|>',
                    'content': '<|layout_text|>\n<|grounding|>' + question,
                    'images': [],
                    'words': words,
                    'boxes': boxes
                },
                {'role': '<|Assistant|>', 'content': ''}
            ]
            prepare = processor(
                conversations=conversation,
                images=[],
                force_batchify=True,
                system_prompt=""
            ).to(model.device, dtype=dtype)

            with torch.no_grad():
                if chunk_size == -1:
                    inputs_embeds = model.prepare_inputs_embeds(**prepare)
                    pkv = None
                else:
                    inputs_embeds, pkv = model.incremental_prefilling(
                        input_ids=prepare.input_ids,
                        images=prepare.images,
                        images_seq_mask=prepare.images_seq_mask,
                        images_spatial_crop=prepare.images_spatial_crop,
                        attention_mask=prepare.attention_mask,
                        chunk_size=chunk_size
                    )
                outputs = model.generate(
                    inputs_embeds=inputs_embeds,
                    input_ids=prepare.input_ids,
                    images=prepare.images,
                    images_seq_mask=prepare.images_seq_mask,
                    images_spatial_crop=prepare.images_spatial_crop,
                    attention_mask=prepare.attention_mask,
                    past_key_values=pkv,
                    pad_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=256,
                    do_sample=False,
                    use_cache=True
                )
            answer = tokenizer.decode(
                outputs[0][prepare.input_ids.size(1):].cpu().tolist(),
                skip_special_tokens=True
            )

            print(f"{name} | {question} -> {answer}")

            if processed < num_prompts_for_cka:
                for idx, layer in enumerate(model.language.model.layers):
                    if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
                        for a in (layer.mlp._expert_activations or []):
                            if a is not None:
                                all_acts[idx].append(a.detach().cpu())
                processed += 1

            results.append({
                'doc': name,
                'question': question,
                'predicted_answer': answer,
                'ground_truth': [v.get('value') for v in values]
            })

    # Compute CKA after all prompts
    all_cka = {}
    for idx, acts in all_acts.items():
        if not acts:

            continue
        # Ensure all tensors are the same shape
        lengths = [a.shape[0] for a in acts]
        common_len = min(lengths)
        filtered = [a[:common_len] for a in acts if a.shape[0] >= common_len]
        try:
            batch = torch.stack(filtered, dim=1)
            all_cka[idx] = compute_layer_cka(batch)
        except Exception as e:
            print(f"Skipping CKA for layer {idx} due to shape mismatch: {e}")

    if all_cka:
        save_cka_heatmaps(all_cka, output_dir=heatmap_output_dir)
        similar = find_similar_experts(all_cka, threshold=0.7)
        for layer, pairs in similar.items():
            print(f"Layer {layer} similar experts: {pairs}")

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)

    return results


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--docs_file', default="/data/stud/raza/DeepSeek-VL2/DocVQA_dataset/aws_neurips_time/docvqa/test/document.jsonl", help='Path      to document.jsonl')
    parser.add_argument('--content_file', default="/data/stud/raza/DeepSeek-VL2/DocVQA_dataset/aws_neurips_time/docvqa/test/documents_content.jsonl",     help='Path to document_content.jsonl')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--chunk_size', type=int, default=-1)
    parser.add_argument('--output_file', default=None)
    parser.add_argument('--num_prompts_for_cka', type=int, default=500)
    parser.add_argument('--heatmap_output_dir', default='cka_heatmaps')
    args = parser.parse_args()

    docvqa_inference(
        annotations_file=args.docs_file,
        contents_file=args.content_file,
        model_path=args.model_path,
        chunk_size=args.chunk_size,
        output_file=args.output_file,
        num_prompts_for_cka=args.num_prompts_for_cka,
        heatmap_output_dir=args.heatmap_output_dir    )
'''

'''
import json
import os
from argparse import ArgumentParser
from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM
from PIL import Image

from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
from deepseek_vl2.utils.cka_utils import compute_layer_cka, save_cka_heatmaps, find_similar_experts

def docvqa_inference(
    annotations_file: str,
    contents_file: str,
    model_path: str,
    chunk_size: int = -1,
    output_file: str = None,
    num_prompts_for_cka: int = 10000,
    heatmap_output_dir: str = "cka_heatmaps"
):
    dtype = torch.bfloat16

    processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype
    ).cuda().eval()

    # Enable CKA tracking
    for layer in model.language.model.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            layer.mlp.cka_tracking = True
            layer.mlp._expert_activations = [None] * len(layer.mlp.experts)

    # Prepare storage: one list per expert per layer
    all_acts: Dict[int, List[List[torch.Tensor]]] = {}
    for idx, layer in enumerate(model.language.model.layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            all_acts[idx] = [[] for _ in layer.mlp.experts]

    # Load doc annotations
    with open(annotations_file, 'r') as f_ann:
        ann_map = { json.loads(line)['name']: json.loads(line) for line in f_ann }

    # Load doc token contents
    with open(contents_file, 'r') as f_cont:
        contents_map = { json.loads(line)['name']: json.loads(line).get('contents', []) for line in f_cont }

    results = []
    processed = 0

    for name, ann_doc in ann_map.items():
        tokens_layer = contents_map.get(name, [])
        if not tokens_layer:
            print(f"Warning: no contents for {name}, skipping")
            continue

        # Preextract words and boxes once per document
        words, boxes = [], []
        for item in tokens_layer:
            if 'tokens_layer' in item:
                tl = item['tokens_layer']
                if 'tokens' in tl and 'positions' in tl:
                    words.extend(tl['tokens'])
                    boxes.extend(tl['positions'])
            elif 'common_format' in item:
                cf = item['common_format']
                words.extend(cf.get('tokens', []))
                boxes.extend(cf.get('positions', []))

        for ann in ann_doc.get('annotations', []):
            question = ann.get('key', '')
            ground = [v.get('value') for v in ann.get('values', [])]

            conversation = [
                {
                    'role': '<|User|>',
                    'content': '<|layout_text|>\n<|grounding|>' + question,
                    'images': [],
                    'words': words,
                    'boxes': boxes
                },
                {'role': '<|Assistant|>', 'content': ''}
            ]

            prepare = processor(
                conversations=conversation,
                images=[],
                words=[words],
                boxes=[boxes],
                force_batchify=True,
                system_prompt=""
            ).to(model.device, dtype=dtype)

            with torch.no_grad():
                if chunk_size == -1:
                    inputs_embeds = model.prepare_inputs_embeds(**prepare)
                    pkv = None
                else:
                    inputs_embeds, pkv = model.incremental_prefilling(
                        input_ids=prepare.input_ids,
                        images=prepare.images,
                        images_seq_mask=prepare.images_seq_mask,
                        images_spatial_crop=prepare.images_spatial_crop,
                        attention_mask=prepare.attention_mask,
                        chunk_size=chunk_size
                    )
                outputs = model.generate(
                    inputs_embeds=inputs_embeds,
                    input_ids=prepare.input_ids,
                    images=prepare.images,
                    images_seq_mask=prepare.images_seq_mask,
                    images_spatial_crop=prepare.images_spatial_crop,
                    attention_mask=prepare.attention_mask,
                    past_key_values=pkv,
                    pad_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=256,
                    do_sample=False,
                    use_cache=True
                )

            answer = tokenizer.decode(
                outputs[0][prepare.input_ids.size(1):].cpu().tolist(),
                skip_special_tokens=True
            )
            print(f"{name} | Q: {question} -> A: {answer}")

            results.append({
                'doc': name,
                'question': question,
                'predicted_answer': answer,
                'ground_truth': ground
            })

            if processed < num_prompts_for_cka:
                for idx, expert_lists in all_acts.items():
                    acts = model.language.model.layers[idx].mlp._expert_activations
                    for e, act in enumerate(acts):
                        if act is not None and act.numel() > 0:
                            expert_lists[e].append(act.detach().cpu().float())
                processed += 1

    # Compute CKA
    all_cka = {}
    for idx, expert_lists in all_acts.items():
        mats = [torch.cat(el, dim=0) if el else torch.empty((0, model.config.vision_config.width))
                for el in expert_lists]
        counts = [m.shape[0] for m in mats]
        if not counts or min(counts) == 0:
            print(f"Layer {idx} :  insufficient activations  skipping")
            continue
        Nmin = min(counts)
        batch = torch.stack([m[:Nmin] for m in mats], dim=1)
        all_cka[idx] = compute_layer_cka(batch)
        print(f"Layer {idx} CKA:\n{all_cka[idx]}\n")

    if all_cka:
        save_cka_heatmaps(all_cka, output_dir=heatmap_output_dir)
        similar = find_similar_experts(all_cka, threshold=0.7)
        for layer, pairs in similar.items():
            print(f"Layer {layer} similar experts: {pairs}")

    if output_file:
        with open(output_file, 'w') as fout:
            json.dump(results, fout, indent=2)

    return results


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--docs_file', default="/data/stud/raza/DeepSeek-VL2/DocVQA_dataset/aws_neurips_time/docvqa/test/document.jsonl", help='Path      to document.jsonl')
    parser.add_argument('--content_file', default="/data/stud/raza/DeepSeek-VL2/DocVQA_dataset/aws_neurips_time/docvqa/test/documents_content.jsonl",     help='Path to document_content.jsonl')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--chunk_size', type=int, default=-1)
    parser.add_argument('--output_file', default=None)
    parser.add_argument('--num_prompts_for_cka', type=int, default=10000)
    parser.add_argument('--heatmap_output_dir', default='cka_heatmaps')
    args = parser.parse_args()

    docvqa_inference(
        annotations_file=args.docs_file,
        contents_file=args.content_file,
        model_path=args.model_path,
        chunk_size=args.chunk_size,
        output_file=args.output_file,
        num_prompts_for_cka=args.num_prompts_for_cka,
        heatmap_output_dir=args.heatmap_output_dir    )
        

'''
'''
import json
import os
from argparse import ArgumentParser
from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM
from PIL import Image

from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
from deepseek_vl2.utils.cka_utils import compute_layer_cka, save_cka_heatmaps, find_similar_experts

def docvqa_inference(
    annotations_file: str,
    contents_file: str,
    model_path: str,
    chunk_size: int = -1,
    output_file: str = None,
    num_prompts_for_cka: int =600,
    heatmap_output_dir: str = "cka_heatmaps"
):
    dtype = torch.bfloat16

    processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype
    ).cuda().eval()

    # Enable CKA tracking
    for layer in model.language.model.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            layer.mlp.cka_tracking = True
            layer.mlp._expert_activations = [None] * len(layer.mlp.experts)

    # Prepare storage: one list per expert per layer
    all_acts: Dict[int, List[List[torch.Tensor]]] = {}
    for idx, layer in enumerate(model.language.model.layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            all_acts[idx] = [[] for _ in layer.mlp.experts]

    # Load doc annotations
    with open(annotations_file, 'r') as f_ann:
        ann_map = { json.loads(line)['name']: json.loads(line) for line in f_ann }

    # Load doc token contents
    with open(contents_file, 'r') as f_cont:
        contents_map = { json.loads(line)['name']: json.loads(line).get('contents', []) for line in f_cont }

    results = []
    processed = 0

    for name, ann_doc in ann_map.items():
        tokens_layer = contents_map.get(name, [])
        if not tokens_layer:
            print(f"Warning: no contents for {name}, skipping")
            continue

        # Preextract words and boxes once per document
        words, boxes = [], []
        for item in tokens_layer:
            if 'tokens_layer' in item:
                tl = item['tokens_layer']
                if 'tokens' in tl and 'positions' in tl:
                    words.extend(tl['tokens'])
                    boxes.extend(tl['positions'])
            elif 'common_format' in item:
                cf = item['common_format']
                words.extend(cf.get('tokens', []))
                boxes.extend(cf.get('positions', []))

        for ann in ann_doc.get('annotations', []):
            question = ann.get('key', '')
            ground = [v.get('value') for v in ann.get('values', [])]

            conversation = [
                {
                    'role': '<|User|>',
                    'content': '<|layout_text|>\n<|grounding|>' + question,
                    'images': [],
                    'words': words,
                    'boxes': boxes
                },
                {'role': '<|Assistant|>', 'content': ''}
            ]

            prepare = processor(
                conversations=conversation,
                images=[],
                words=[words],
                boxes=[boxes],
                force_batchify=True,
                system_prompt=""
            ).to(model.device, dtype=dtype)

            with torch.no_grad():
                if chunk_size == -1:
                    inputs_embeds = model.prepare_inputs_embeds(**prepare)
                    pkv = None
                else:
                    inputs_embeds, pkv = model.incremental_prefilling(
                        input_ids=prepare.input_ids,
                        images=prepare.images,
                        images_seq_mask=prepare.images_seq_mask,
                        images_spatial_crop=prepare.images_spatial_crop,
                        attention_mask=prepare.attention_mask,
                        chunk_size=chunk_size
                    )
                outputs = model.generate(
                    inputs_embeds=inputs_embeds,
                    input_ids=prepare.input_ids,
                    images=prepare.images,
                    images_seq_mask=prepare.images_seq_mask,
                    images_spatial_crop=prepare.images_spatial_crop,
                    attention_mask=prepare.attention_mask,
                    past_key_values=pkv,
                    pad_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=256,
                    do_sample=False,
                    use_cache=True
                )

            answer = tokenizer.decode(
                outputs[0][prepare.input_ids.size(1):].cpu().tolist(),
                skip_special_tokens=True
            )
            print(f"{name} | Q: {question} -> A: {answer}")

            results.append({
                'doc': name,
                'question': question,
                'predicted_answer': answer,
                'ground_truth': ground
            })

            if processed < num_prompts_for_cka:
                for idx, expert_lists in all_acts.items():
                    acts = model.language.model.layers[idx].mlp._expert_activations
                    for e, act in enumerate(acts):
                        if act is not None and act.numel() > 0:
                            expert_lists[e].append(act.detach().cpu().float())
                processed += 1

    # Compute CKA
    all_cka = {}
    for idx, expert_lists in all_acts.items():
        mats = [torch.cat(el, dim=0) if el else torch.empty((0, model.config.vision_config.width))
                for el in expert_lists]
        counts = [m.shape[0] for m in mats]
        if not counts or min(counts) == 0:
            print(f"Layer {idx} :  insufficient activations  skipping")
            continue
        Nmin = min(counts)
        batch = torch.stack([m[:Nmin] for m in mats], dim=1)
        all_cka[idx] = compute_layer_cka(batch)
        print(f"Layer {idx} CKA:\n{all_cka[idx]}\n")

    if all_cka:
        save_cka_heatmaps(all_cka, output_dir=heatmap_output_dir)
        similar = find_similar_experts(all_cka, threshold=0.7)
        for layer, pairs in similar.items():
            print(f"Layer {layer} similar experts: {pairs}")

    if output_file:
        with open(output_file, 'w') as fout:
            json.dump(results, fout, indent=2)

    return results


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--docs_file', default="/data/stud/raza/DeepSeek-VL2/DocVQA_dataset/aws_neurips_time/docvqa/test/document.jsonl", help='Path      to document.jsonl')
    parser.add_argument('--content_file', default="/data/stud/raza/DeepSeek-VL2/DocVQA_dataset/aws_neurips_time/docvqa/test/documents_content.jsonl",     help='Path to document_content.jsonl')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--chunk_size', type=int, default=-1)
    parser.add_argument('--output_file', default=None)
    parser.add_argument('--num_prompts_for_cka', type=int, default=600)
    parser.add_argument('--heatmap_output_dir', default='cka_heatmaps')
    args = parser.parse_args()

    docvqa_inference(
        annotations_file=args.docs_file,
        contents_file=args.content_file,
        model_path=args.model_path,
        chunk_size=args.chunk_size,
        output_file=args.output_file,
        num_prompts_for_cka=args.num_prompts_for_cka,
        heatmap_output_dir=args.heatmap_output_dir    )
        
'''
'''

import json
import os
from argparse import ArgumentParser
from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM
from PIL import Image

from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
from deepseek_vl2.utils.cka_utils import compute_layer_cka, save_cka_heatmaps, find_similar_experts

def docvqa_inference(
    annotations_file: str,
    contents_file: str,
    model_path: str,
    chunk_size: int = -1,
    output_file: str = None,
    num_prompts_for_cka: int = 80,
    heatmap_output_dir: str = "cka_heatmaps",
    max_questions: int = 80
):
    dtype = torch.bfloat16

    processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype
    ).cuda().eval()

    # Enable CKA tracking
    for layer in model.language.model.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            layer.mlp.cka_tracking = True
            layer.mlp._expert_activations = [None] * len(layer.mlp.experts)

    # Prepare storage
    all_acts: Dict[int, List[List[torch.Tensor]]] = {}
    for idx, layer in enumerate(model.language.model.layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            all_acts[idx] = [[] for _ in layer.mlp.experts]

    with open(annotations_file, 'r') as f_ann:
        ann_map = { json.loads(line)['name']: json.loads(line) for line in f_ann }

    with open(contents_file, 'r') as f_cont:
        contents_map = { json.loads(line)['name']: json.loads(line).get('contents', []) for line in f_cont }

    results = []
    processed = 0
    total_questions = 0

    for name, ann_doc in ann_map.items():
        tokens_layer = contents_map.get(name, [])
        if not tokens_layer:
            print(f"Warning: no contents for {name}, skipping")
            continue

        words, boxes = [], []
        for item in tokens_layer:
            if 'tokens_layer' in item:
                tl = item['tokens_layer']
                words.extend(tl.get('tokens', []))
                boxes.extend(tl.get('positions', []))
            elif 'common_format' in item:
                cf = item['common_format']
                words.extend(cf.get('tokens', []))
                boxes.extend(cf.get('positions', []))

        for ann in ann_doc.get('annotations', []):
            if total_questions >= max_questions:
                break

            question = ann.get('key', '')
            ground = [v.get('value') for v in ann.get('values', [])]

            conversation = [
                {
                    'role': '<|User|>',
                    'content': '<|layout_text|>\n<|grounding|>' + question,
                    'images': [],
                    'words': words,
                    'boxes': boxes
                },
                {'role': '<|Assistant|>', 'content': ''}
            ]

            prepare = processor(
                conversations=conversation,
                images=[],
                words=[words],
                boxes=[boxes],
                force_batchify=True,
                system_prompt=""
            ).to(model.device, dtype=dtype)

            with torch.no_grad():
                if chunk_size == -1:
                    inputs_embeds = model.prepare_inputs_embeds(**prepare)
                    pkv = None
                else:
                    inputs_embeds, pkv = model.incremental_prefilling(
                        input_ids=prepare.input_ids,
                        images=prepare.images,
                        images_seq_mask=prepare.images_seq_mask,
                        images_spatial_crop=prepare.images_spatial_crop,
                        attention_mask=prepare.attention_mask,
                        chunk_size=chunk_size
                    )

                outputs = model.generate(
                    inputs_embeds=inputs_embeds,
                    input_ids=prepare.input_ids,
                    images=prepare.images,
                    images_seq_mask=prepare.images_seq_mask,
                    images_spatial_crop=prepare.images_spatial_crop,
                    attention_mask=prepare.attention_mask,
                    past_key_values=pkv,
                    pad_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=256,
                    do_sample=False,
                    use_cache=True
                )

            answer = tokenizer.decode(
                outputs[0][prepare.input_ids.size(1):].cpu().tolist(),
                skip_special_tokens=True
            )
            print(f"{name} | Q: {question} -> A: {answer}")

            results.append({
                'doc': name,
                'question': question,
                'predicted_answer': answer,
                'ground_truth': ground
            })

            # Save activations for CKA
            if processed < num_prompts_for_cka:
                for idx, expert_lists in all_acts.items():
                    acts = model.language.model.layers[idx].mlp._expert_activations
                    for e, act in enumerate(acts):
                        if act is not None and act.numel() > 0:
                            expert_lists[e].append(act.detach().cpu().float())
                processed += 1
                total_questions += 1

        if total_questions >= max_questions:
            break

    # Compute CKA
    all_cka = {}
    for idx, expert_lists in all_acts.items():
        mats = [torch.cat(el, dim=0) if el else torch.empty((0, model.config.vision_config.width))
                for el in expert_lists]
        counts = [m.shape[0] for m in mats]
        if not counts or min(counts) == 0:
            print(f"Layer {idx}: insufficient activations, skipping")
            continue
        Nmin = min(counts)
        batch = torch.stack([m[:Nmin] for m in mats], dim=1)
        all_cka[idx] = compute_layer_cka(batch)
        print(f"Layer {idx} CKA:\n{all_cka[idx]}\n")

    if all_cka:
        save_cka_heatmaps(all_cka, output_dir=heatmap_output_dir)
        similar = find_similar_experts(all_cka, threshold=0.7)
        for layer, pairs in similar.items():
            print(f"Layer {layer} similar experts: {pairs}")

    if output_file:
        with open(output_file, 'w') as fout:
            json.dump(results, fout, indent=2)

    return results



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--docs_file', default="/data/stud/raza/DeepSeek-VL2/DocVQA_dataset/aws_neurips_time/docvqa/test/document.jsonl", help='Path      to document.jsonl')
    parser.add_argument('--content_file', default="/data/stud/raza/DeepSeek-VL2/DocVQA_dataset/aws_neurips_time/docvqa/test/documents_content.jsonl",     help='Path to document_content.jsonl')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--chunk_size', type=int, default=-1)
    parser.add_argument('--output_file', default=None)
    parser.add_argument('--num_prompts_for_cka', type=int, default=5)
    parser.add_argument('--heatmap_output_dir', default='cka_heatmaps')
    args = parser.parse_args()

    docvqa_inference(
        annotations_file=args.docs_file,
        contents_file=args.content_file,
        model_path=args.model_path,
        chunk_size=args.chunk_size,
        output_file=args.output_file,
        num_prompts_for_cka=args.num_prompts_for_cka,
        heatmap_output_dir=args.heatmap_output_dir,
        max_questions=5 # <-- limit to 5 questions
    )
'''
'''
import json
import os
from argparse import ArgumentParser
from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM
from PIL import Image
import string
import re
from collections import Counter

from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
from deepseek_vl2.utils.cka_utils import compute_layer_cka, save_cka_heatmaps, find_similar_experts

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text): return ''.join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return float(pred_tokens == gt_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

def exact_match(prediction: str, ground_truth: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def docvqa_inference(
    annotations_file: str,
    contents_file: str,
    model_path: str,
    chunk_size: int = -1,
    output_file: str = None,
    num_prompts_for_cka: int = 80,
    heatmap_output_dir: str = "cka_heatmaps",
    max_questions: int = 80
) -> List[Dict]:
    dtype = torch.bfloat16
    processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype
    ).cuda().eval()

    # Enable CKA tracking
    for layer in model.language.model.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            layer.mlp.cka_tracking = True
            layer.mlp._expert_activations = [None] * len(layer.mlp.experts)

    # Prepare storage for activations
    all_acts: Dict[int, List[List[torch.Tensor]]] = {}
    for idx, layer in enumerate(model.language.model.layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            all_acts[idx] = [[] for _ in layer.mlp.experts]

    # Load annotations and contents
    with open(annotations_file, 'r') as f_ann:
        ann_map = { json.loads(line)['name']: json.loads(line) for line in f_ann }
    with open(contents_file, 'r') as f_cont:
        contents_map = { json.loads(line)['name']: json.loads(line).get('contents', []) for line in f_cont }

    results = []
    processed = 0
    total_questions = 0

    # Inference loop
    for name, ann_doc in ann_map.items():
        tokens_layer = contents_map.get(name, [])
        if not tokens_layer:
            print(f"Warning: no contents for {name}, skipping")
            continue

        words, boxes = [], []
        for item in tokens_layer:
            if 'tokens_layer' in item:
                tl = item['tokens_layer']; words.extend(tl.get('tokens', [])); boxes.extend(tl.get('positions', []))
            elif 'common_format' in item:
                cf = item['common_format']; words.extend(cf.get('tokens', [])); boxes.extend(cf.get('positions', []))

        for ann in ann_doc.get('annotations', []):
            if total_questions >= max_questions:
                break

            question = ann.get('key', '')
            ground = [v.get('value') for v in ann.get('values', [])]

            conversation = [
                {'role': '<|User|>', 'content': '<|layout_text|>\n<|grounding|>' + question,
                 'images': [], 'words': words, 'boxes': boxes},
                {'role': '<|Assistant|>', 'content': ''}
            ]

            prepare = processor(
                conversations=conversation, images=[], words=[words], boxes=[boxes],
                force_batchify=True, system_prompt=""
            ).to(model.device, dtype=dtype)

            with torch.no_grad():
                if chunk_size == -1:
                    inputs_embeds = model.prepare_inputs_embeds(**prepare)
                    pkv = None
                else:
                    inputs_embeds, pkv = model.incremental_prefilling(
                        input_ids=prepare.input_ids, images=prepare.images,
                        images_seq_mask=prepare.images_seq_mask, images_spatial_crop=prepare.images_spatial_crop,
                        attention_mask=prepare.attention_mask, chunk_size=chunk_size
                    )

                outputs = model.generate(
                    inputs_embeds=inputs_embeds, input_ids=prepare.input_ids,
                    images=prepare.images, images_seq_mask=prepare.images_seq_mask,
                    images_spatial_crop=prepare.images_spatial_crop, attention_mask=prepare.attention_mask,
                    past_key_values=pkv, pad_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=256, do_sample=False, use_cache=True
                )

            answer = tokenizer.decode(
                outputs[0][prepare.input_ids.size(1):].cpu().tolist(), skip_special_tokens=True
            )
            # Print predicted and actual answers
            print(f"{name} | Q: {question}")
            print(f"  Predicted: {answer}")
            print(f"  Ground Truth: {ground}\n")

            results.append({
                'doc': name,
                'question': question,
                'predicted_answer': answer,
                'ground_truth': ground
            })

            # Collect activations for CKA
            if processed < num_prompts_for_cka:
                for idx, expert_lists in all_acts.items():
                    acts = model.language.model.layers[idx].mlp._expert_activations
                    for e, act in enumerate(acts):
                        if act is not None and act.numel() > 0:
                            expert_lists[e].append(act.detach().cpu().float())
                processed += 1
                total_questions += 1

        if total_questions >= max_questions:
            break

    # Evaluation
    total_em, total_f1, total = 0, 0, 0
    for res in results:
        pred = res['predicted_answer']; gts = res['ground_truth']
        if not gts: continue
        em = max(exact_match(pred, gt) for gt in gts)
        f1 = max(f1_score(pred, gt) for gt in gts)
        total_em += em; total_f1 += f1; total += 1

    if total > 0:
        avg_em = 100.0 * total_em / total; avg_f1 = 100.0 * total_f1 / total
        print("\n==== Evaluation Results ====")
        print(f"Exact Match (EM): {avg_em:.2f}%")
        print(f"F1 Score: {avg_f1:.2f}%")
    else:
        print("No valid examples to evaluate.")

    # CKA computation
    all_cka = {}
    for idx, expert_lists in all_acts.items():
        mats = [torch.cat(el, dim=0) if el else torch.empty((0, model.config.vision_config.width))
                for el in expert_lists]
        counts = [m.shape[0] for m in mats]
        if not counts or min(counts) == 0:
            print(f"Layer {idx}: insufficient activations, skipping")
            continue
        Nmin = min(counts)
        batch = torch.stack([m[:Nmin] for m in mats], dim=1)
        all_cka[idx] = compute_layer_cka(batch)
        print(f"Layer {idx} CKA:\n{all_cka[idx]}\n")

    if all_cka:
        save_cka_heatmaps(all_cka, output_dir=heatmap_output_dir)
        similar = find_similar_experts(all_cka, threshold=0.7)
        for layer, pairs in similar.items():
            print(f"Layer {layer} similar experts: {pairs}")

    # Save results and metrics
    if output_file:
        out_data = {'results': results}
        if total > 0:
            out_data['metrics'] = {'Exact Match (%)': avg_em, 'F1 Score (%)': avg_f1}
        with open(output_file, 'w') as fout:
            json.dump(out_data, fout, indent=2)

    return results


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--docs_file', default="/data/stud/raza/DeepSeek-VL2/DocVQA_dataset/aws_neurips_time/docvqa/test/document.jsonl", help='Path to document.jsonl')
    parser.add_argument('--content_file', default="/data/stud/raza/DeepSeek-VL2/DocVQA_dataset/aws_neurips_time/docvqa/test/documents_content.jsonl", help='Path to document_content.jsonl')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--chunk_size', type=int, default=-1)
    parser.add_argument('--output_file', default=None)
    parser.add_argument('--num_prompts_for_cka', type=int, default=5)
    parser.add_argument('--heatmap_output_dir', default='cka_heatmaps')
    args = parser.parse_args()

    docvqa_inference(
        annotations_file=args.docs_file,
        contents_file=args.content_file,
        model_path=args.model_path,
        chunk_size=args.chunk_size,
        output_file=args.output_file,
        num_prompts_for_cka=args.num_prompts_for_cka,
        heatmap_output_dir=args.heatmap_output_dir,
        max_questions=5
    )

    
'''
'''

import os
import torch
import json
from argparse import ArgumentParser
from typing import List, Dict
from PIL import Image

from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor

def load_pil_images_from_ids(image_ids: List[str], image_folder: str) -> List[Image.Image]:
    pil_images = []
    for image_id in image_ids:
        image_path = os.path.join(image_folder, f"{image_id}.jpg")
        if os.path.exists(image_path):
            try:
                img = Image.open(image_path).convert("RGB")
                pil_images.append(img)
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
        else:
            print(f"Missing image: {image_path}")
    return pil_images

def main(args):
    dtype = torch.bfloat16

    # Load model and processor
    processor = DeepseekVLV2Processor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer
    model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=dtype
    ).cuda().eval()
    

    with open(args.annotation_file, 'r') as f:
        annotations = json.load(f)["data"]

    for item in annotations:
        question = item["question"]
        page_ids = item["page_ids"]

        images = load_pil_images_from_ids(page_ids, args.image_folder)
        if not images:
            print(f"No valid images for question: {question}")
            continue

        conversation = [
            {
                "role": "<|User|>",
                "content": "<image>\n" * len(images) + question,
                "images": [os.path.join(args.image_folder, f"{pid}.jpg") for pid in page_ids]
            },
            {"role": "<|Assistant|>", "content": ""}
        ]

        prepare_inputs = processor(
            conversations=conversation,
            images=images,
            force_batchify=True,
            system_prompt=""
        ).to(model.device, dtype=dtype)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=prepare_inputs.input_ids,
                attention_mask=prepare_inputs.attention_mask,
                images=prepare_inputs.images,
                images_seq_mask=prepare_inputs.images_seq_mask,
                images_spatial_crop=prepare_inputs.images_spatial_crop,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.1,
                use_cache=True,
            )

        answer = tokenizer.decode(
            outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(), skip_special_tokens=True
        )

        print("\n🔹 Question:", question)
        print("📄 Pages:", page_ids)
        print("✅ Answer:", answer)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to model or model name")
    parser.add_argument("--image_folder", type=str,default="extracted_folder/images/", help="Folder containing input images")
    parser.add_argument("--annotation_file", type=str,default="extracted_folder/test.json", help="JSON file with question and page_ids")
    args = parser.parse_args()

    main(args)

'''

'''
import os
import json
import torch
from argparse import ArgumentParser
from typing import List, Dict
from PIL import Image

from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
from deepseek_vl2.utils.cka_utils import compute_layer_cka, save_cka_heatmaps, find_similar_experts


def load_pil_images_from_ids(image_ids: List[str], image_folder: str) -> List[Image.Image]:
    pil_images = []
    for image_id in image_ids:
        image_path = os.path.join(image_folder, f"{image_id}.jpg")
        if os.path.exists(image_path):
            try:
                img = Image.open(image_path).convert("RGB")
                pil_images.append(img)
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
        else:
            print(f"Missing image: {image_path}")
    return pil_images


def main(args):
    dtype = torch.bfloat16

    # Load model and processor
    processor = DeepseekVLV2Processor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer
    model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=dtype
    ).cuda().eval()

    # Enable CKA tracking on MoE layers
    all_acts: Dict[int, List[List[torch.Tensor]]] = {}
    for idx, layer in enumerate(model.language.model.layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            layer.mlp.cka_tracking = True
            layer.mlp._expert_activations = [None] * len(layer.mlp.experts)
            all_acts[idx] = [[] for _ in layer.mlp.experts]

    # Load annotations
    with open(args.annotation_file, 'r') as f:
        annotations = json.load(f)["data"]

    processed = 0
    results = []

    for item in annotations:
        question = item["question"]
        page_ids = item["page_ids"]

        images = load_pil_images_from_ids(page_ids, args.image_folder)
        if not images:
            print(f"No valid images for question: {question}")
            continue

        conversation = [
            {"role": "<|User|>", "content": "<image>\n" * len(images) + question,
             "images": [os.path.join(args.image_folder, f"{pid}.jpg") for pid in page_ids],
            },
            {"role": "<|Assistant|>", "content": ""}
        ]

        prepare_inputs = processor(
            conversations=conversation,
            images=images,
            force_batchify=True,
            system_prompt=""
        ).to(model.device, dtype=dtype)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=prepare_inputs.input_ids,
                attention_mask=prepare_inputs.attention_mask,
                images=prepare_inputs.images,
                images_seq_mask=prepare_inputs.images_seq_mask,
                images_spatial_crop=prepare_inputs.images_spatial_crop,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.1,
                use_cache=True,
            )

        answer = tokenizer.decode(
            outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(), skip_special_tokens=True
        )

        print("\n🔹 Question:", question)
        print("📄 Pages:", page_ids)
        print("✅ Answer:", answer)

        # Collect activations for CKA
        if processed < args.num_prompts_for_cka:
            for idx, layer in enumerate(model.language.model.layers):
                if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
                    acts = layer.mlp._expert_activations
                    for e, act in enumerate(acts):
                        if act is not None and act.numel() > 0:
                            all_acts[idx][e].append(act.detach().cpu().float())
            processed += 1

        results.append({
            'question': question,
            'pages': page_ids,
            'answer': answer
        })

    # Compute and save CKA heatmaps
    if all_acts:
        all_cka = {}
        for idx, expert_lists in all_acts.items():
            mats = [torch.cat(el, dim=0) if el else torch.empty((0, model.config.vision_config.width))
                    for el in expert_lists]
            counts = [m.shape[0] for m in mats]
            if not counts or min(counts) == 0:
                print(f"Layer {idx} : insufficient activations, skipping")
                continue
            Nmin = min(counts)
            batch = torch.stack([m[:Nmin] for m in mats], dim=1)
            all_cka[idx] = compute_layer_cka(batch)
            print(f"Layer {idx} CKA:\n{all_cka[idx]}\n")

        save_cka_heatmaps(all_cka, output_dir=args.heatmap_output_dir)
        similar = find_similar_experts(all_cka, threshold=args.cka_threshold)
        for layer, pairs in similar.items():
            print(f"Layer {layer} similar experts: {pairs}")

    # Optionally save QA results
    if args.output_file:
        with open(args.output_file, 'w') as fout:
            json.dump(results, fout, indent=2)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_folder", type=str, default="extracted_folder/images/")
    parser.add_argument("--annotation_file", type=str, default="extracted_folder/test.json")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--num_prompts_for_cka", type=int, default=600,
                        help="Number of prompts to collect for CKA")
    parser.add_argument("--heatmap_output_dir", type=str, default="cka_heatmaps")
    parser.add_argument("--cka_threshold", type=float, default=0.7,
                        help="Threshold for similar expert detection")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
'''
'''

import os
import json
import torch
from argparse import ArgumentParser
from typing import List, Dict
from PIL import Image

from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
from deepseek_vl2.utils.cka_utils import compute_layer_cka, save_cka_heatmaps, find_similar_experts


def load_pil_images_from_ids(image_ids: List[str], image_folder: str) -> List[Image.Image]:
    pil_images = []
    for image_id in image_ids:
        image_path = os.path.join(image_folder, f"{image_id}.png")
        if os.path.exists(image_path):
            try:
                img = Image.open(image_path).convert("RGB")
                pil_images.append(img)
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
        else:
            print(f"Missing image: {image_path}")
    return pil_images


def exact_match(pred: str, truths: List[str]) -> bool:
    # simple normalization: strip and lowercase
    p = pred.strip().lower()
    return any(p == t.strip().lower() for t in truths)


def main(args):
    dtype = torch.bfloat16

    # Load model and processor
    processor = DeepseekVLV2Processor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer
    model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=dtype
    ).cuda().eval()

    # Enable CKA tracking on MoE layers
    all_acts: Dict[int, List[List[torch.Tensor]]] = {}
    for idx, layer in enumerate(model.language.model.layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            layer.mlp.cka_tracking = True
            layer.mlp._expert_activations = [None] * len(layer.mlp.experts)
            all_acts[idx] = [[] for _ in layer.mlp.experts]

    # Load annotations
    with open(args.annotation_file, 'r') as f:
        dataset = json.load(f)
        annotations = dataset.get("data", [])

    processed = 0
    results = []
    correct = 0

    for item in annotations:
        question = item.get("question", "")
        page_ids = item.get("page_ids", [])
        ground_truths = item.get("answers", [])

        images = load_pil_images_from_ids(page_ids, args.image_folder)
        if not images:
            print(f"No valid images for question: {question}")
            continue

        conversation = [
            {"role": "<|User|>", "content": "<image>\n" * len(images) + question,
             "images": [os.path.join(args.image_folder, f"{pid}.jpg") for pid in page_ids],
            },
            {"role": "<|Assistant|>", "content": ""}
        ]

        prepare_inputs = processor(
            conversations=conversation,
            images=images,
            force_batchify=True,
            system_prompt=""
        ).to(model.device, dtype=dtype)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=prepare_inputs.input_ids,
                attention_mask=prepare_inputs.attention_mask,
                images=prepare_inputs.images,
                images_seq_mask=prepare_inputs.images_seq_mask,
                images_spatial_crop=prepare_inputs.images_spatial_crop,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                use_cache=True,
            )

        answer = tokenizer.decode(
            outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(), skip_special_tokens=True
        )

        is_correct = exact_match(answer, ground_truths)
        if is_correct:
            correct += 1

        print(f"\n🔹 Question: {question}")
        print(f"📄 Pages: {page_ids}")
        print(f"✅ Predicted: {answer}")
        print(f"🎯 Ground Truths: {ground_truths}")
        print(f"✔ Exact Match: {is_correct}")

        # Collect activations for CKA
        if processed < args.num_prompts_for_cka:
            for idx, layer in enumerate(model.language.model.layers):
                if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
                    acts = layer.mlp._expert_activations
                    for e, act in enumerate(acts):
                        if act is not None and act.numel() > 0:
                            all_acts[idx][e].append(act.detach().cpu().float())
            processed += 1

        results.append({
            'question': question,
            'pages': page_ids,
            'predicted_answer': answer,
            'ground_truths': ground_truths,
            'exact_match': is_correct
        })

    total = len(results)
    accuracy = correct / total if total > 0 else 0.0
    print(f"\n=== Evaluation ===")
    print(f"Total questions: {total}")
    print(f"Exact Match Accuracy: {accuracy:.4f}")

    # Compute and save CKA heatmaps
    if all_acts:
        all_cka = {}
        for idx, expert_lists in all_acts.items():
            mats = [torch.cat(el, dim=0) if el else torch.empty((0, model.config.vision_config.width))
                    for el in expert_lists]
            counts = [m.shape[0] for m in mats]
            if not counts or min(counts) == 0:
                print(f"Layer {idx} : insufficient activations, skipping")
                continue
            Nmin = min(counts)
            batch = torch.stack([m[:Nmin] for m in mats], dim=1)
            all_cka[idx] = compute_layer_cka(batch)
            print(f"Layer {idx} CKA:\n{all_cka[idx]}\n")

        save_cka_heatmaps(all_cka, output_dir=args.heatmap_output_dir)
        similar = find_similar_experts(all_cka, threshold=args.cka_threshold)
        for layer, pairs in similar.items():
            print(f"Layer {layer} similar experts: {pairs}")

    # Save QA and eval results
    if args.output_file:
        with open(args.output_file, 'w') as fout:
            json.dump({'results': results, 'accuracy': accuracy}, fout, indent=2)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_folder", type=str, default="spdocvqa_images/")
    parser.add_argument("--annotation_file", type=str, default="spdocvqa_annotations/val.json")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--num_prompts_for_cka", type=int, default=200,
                        help="Number of prompts to collect for CKA")
    parser.add_argument("--heatmap_output_dir", type=str, default="cka_heatmaps")
    parser.add_argument("--cka_threshold", type=float, default=0.65,
                        help="Threshold for similar expert detection")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--do_sample", type=bool, default=True)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
'''

'''
import os
import json
import torch
from argparse import ArgumentParser
from typing import List, Dict
from PIL import Image

from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
from deepseek_vl2.utils.cka_utils import compute_layer_cka, save_cka_heatmaps, find_similar_experts


def load_pil_images_from_ids(image_ids: List[str], image_folder: str) -> List[Image.Image]:
  
    pil_images = []
    for image_id in image_ids:
        found = False
        for ext in [".png", ".jpg", ".jpeg"]:
            image_path = os.path.join(image_folder, f"{image_id}{ext}")
            if os.path.exists(image_path):
                try:
                    img = Image.open(image_path).convert("RGB")
                    pil_images.append(img)
                    found = True
                    break
                except Exception as e:
                    print(f"Error loading {image_path}: {e}")
        if not found:
            print(f"Missing image for ID: {image_id}")
    return pil_images


def normalize_text(text: str) -> str:
  
    return text.strip().lower()


def exact_match(pred: str, truths: List[str]) -> bool:
   
    pred_norm = normalize_text(pred)
    truths_norm = [normalize_text(t) for t in truths]
    return any(pred_norm == t for t in truths_norm)


def main(args):
    dtype = torch.bfloat16

    # Load model and processor
    processor = DeepseekVLV2Processor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer
    model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=dtype
    ).cuda().eval()

    # Enable CKA tracking on MoE layers
    all_acts: Dict[int, List[List[torch.Tensor]]] = {}
    for idx, layer in enumerate(model.language.model.layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            layer.mlp.cka_tracking = True
            layer.mlp._expert_activations = [None] * len(layer.mlp.experts)
            all_acts[idx] = [[] for _ in layer.mlp.experts]

    # Load annotations
    with open(args.annotation_file, 'r') as f:
        dataset = json.load(f)
        annotations = dataset.get("data", [])

    processed = 0
    results = []
    correct = 0

    for item in annotations:
        question = item.get("question", "")

        # Remove extension and folder prefix
        image_file = os.path.splitext(item.get("image", ""))[0]
        image_file = image_file.replace("documents/", "")

        ground_truths = item.get("answers", [])

        # Load image(s)
        images = load_pil_images_from_ids([image_file], args.image_folder)
        if not images:
            print(f"No valid images for question: {question}")
            continue

        conversation = [
            {
                "role": "<|User|>",
                "content": "<image>\n" * len(images) + question,
                "images": [os.path.join(args.image_folder, f"{image_file}.png")],
            },
            {"role": "<|Assistant|>", "content": ""}
        ]

        prepare_inputs = processor(
            conversations=conversation,
            images=images,
            force_batchify=True,
            system_prompt=""
        ).to(model.device, dtype=dtype)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=prepare_inputs.input_ids,
                attention_mask=prepare_inputs.attention_mask,
                images=prepare_inputs.images,
                images_seq_mask=prepare_inputs.images_seq_mask,
                images_spatial_crop=prepare_inputs.images_spatial_crop,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                use_cache=True,
            )

        answer = tokenizer.decode(
            outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(),
            skip_special_tokens=True
        )

        norm_pred = normalize_text(answer)
        norm_truths = [normalize_text(t) for t in ground_truths]
        is_correct = norm_pred in norm_truths
        if is_correct:
            correct += 1

        print(f"\n?? Question: {question}")
        print(f"?? Image: {image_file}")
        print(f"? Predicted: {answer}")
        print(f"   ?? Normalized Pred: {norm_pred}")
        print(f"?? Ground Truths: {ground_truths}")
        print(f"   ?? Normalized Truths: {norm_truths}")
        print(f"? Exact Match: {is_correct}")

        # Collect activations for CKA
        if processed < args.num_prompts_for_cka:
            for idx, layer in enumerate(model.language.model.layers):
                if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
                    acts = layer.mlp._expert_activations
                    for e, act in enumerate(acts):
                        if act is not None and act.numel() > 0:
                            all_acts[idx][e].append(act.detach().cpu().float())
            processed += 1

        results.append({
            'question': question,
            'image': image_file,
            'predicted_answer': answer,
            'normalized_pred': norm_pred,
            'ground_truths': ground_truths,
            'normalized_truths': norm_truths,
            'exact_match': is_correct
        })

    total = len(results)
    accuracy = correct / total if total > 0 else 0.0
    print(f"\n=== Evaluation ===")
    print(f"Total questions: {total}")
    print(f"Exact Match Accuracy: {accuracy:.4f}")

    # Compute and save CKA heatmaps
    if all_acts:
        all_cka = {}
        for idx, expert_lists in all_acts.items():
            mats = [torch.cat(el, dim=0) if el else torch.empty((0, model.config.vision_config.width))
                    for el in expert_lists]
            counts = [m.shape[0] for m in mats]
            if not counts or min(counts) == 0:
                print(f"Layer {idx} : insufficient activations, skipping")
                continue
            Nmin = min(counts)
            batch = torch.stack([m[:Nmin] for m in mats], dim=1)
            all_cka[idx] = compute_layer_cka(batch)
            print(f"Layer {idx} CKA:\n{all_cka[idx]}\n")

        save_cka_heatmaps(all_cka, output_dir=args.heatmap_output_dir)
        similar = find_similar_experts(all_cka, threshold=args.cka_threshold)
        for layer, pairs in similar.items():
            print(f"Layer {layer} similar experts: {pairs}")

    # Save QA and eval results
    if args.output_file:
        with open(args.output_file, 'w') as fout:
            json.dump({'results': results, 'accuracy': accuracy}, fout, indent=2)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_folder", type=str, default="spdocvqa_images/")
    parser.add_argument("--annotation_file", type=str, default="spdocvqa_annotations/val.json")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--num_prompts_for_cka", type=int, default=200)
    parser.add_argument("--heatmap_output_dir", type=str, default="cka_heatmaps")
    parser.add_argument("--cka_threshold", type=float, default=0.65)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--do_sample", type=bool, default=True)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
'''
#!/usr/bin/env python3
"""
Inference + MoE expert activation collection + CKA (linear or RBF with optional RFF approx)
Usage example (approx RBF):
  python inference_with_cka.py --model_path /path/to/model --annotation_file val.json \
    --max_questions 500 --num_prompts_for_cka 500 \
    --cka_method rbf --cka_rff_features 512 --cka_max_samples 512
"""
#!/usr/bin/env python3
"""
Inference + MoE expert activation collection + exact RBF-CKA (uses compute_layer_cka)
- Ensures correct tensor shape (N, E, D)
- Caps samples per expert via --cka_max_samples to avoid O(N^2) blowup
- Prints and returns expert pairs with similarity >= --cka_similarity_threshold
Usage example:
  python inference_with_cka_exact_rbf.py --model_path /path/to/model --annotation_file val.json \
    --max_questions 500 --num_prompts_for_cka 500 \
    --cka_method rbf --cka_max_samples 256 --cka_similarity_threshold 0.6
"""
#!/usr/bin/env python3
"""
Inference + MoE expert activation collection + exact RBF-CKA
Includes robust handling for missing model.config attributes (e.g. 'use_cache').
Usage example:
  python inference_with_cka_fixed.py --model_path /path/to/model --annotation_file val.json \
    --max_questions 500 --num_prompts_for_cka 500 \
    --cka_method rbf --cka_max_samples 256 --cka_similarity_threshold 0.6
"""

'''
import os
import json
import math
import re
import torch
import random
from argparse import ArgumentParser
from typing import List, Dict
from PIL import Image

from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
from deepseek_vl2.utils.cka_utils import compute_layer_cka, save_cka_heatmaps, find_similar_experts


# -------------------------
# Utilities
# -------------------------
def load_pil_images_from_ids(image_ids: List[str], image_folder: str) -> List[Image.Image]:
    pil_images = []
    for image_id in image_ids:
        found = False
        for ext in [".png", ".jpg", ".jpeg"]:
            image_path = os.path.join(image_folder, f"{image_id}{ext}")
            if os.path.exists(image_path):
                try:
                    img = Image.open(image_path).convert("RGB")
                    pil_images.append(img)
                    found = True
                    break
                except Exception as e:
                    print(f"Error loading {image_path}: {e}")
        if not found:
            print(f"Missing image for ID: {image_id}")
    return pil_images


def normalize_text(text: str) -> str:
    return text.strip().lower()


# -------------------------
# Helpers to patch model.config attributes
# -------------------------
def ensure_config_attrs(model, defaults: Dict[str, object] = None):
    """
    Ensure certain attributes exist on model.config.
    defaults: dict of attr->value to set if missing.
    """
    defaults = defaults or {"use_cache": True}
    for k, v in defaults.items():
        if not hasattr(model.config, k):
            try:
                setattr(model.config, k, v)
                print(f"Patched model.config.{k} = {v}")
            except Exception as e:
                print(f"Failed to set model.config.{k}: {e}")


# -------------------------
# Hook helpers to collect expert activations
# -------------------------
def _make_expert_hook(all_acts: Dict[int, List[List[torch.Tensor]]], layer_idx: int, expert_idx: int):
    def hook(module, input, output):
        try:
            out = output[0] if isinstance(output, (list, tuple)) else output
            out = out.detach().cpu().float()
            if out.dim() > 2:
                out = out.view(out.shape[0], -1)
            if layer_idx not in all_acts:
                all_acts[layer_idx] = []
            while len(all_acts[layer_idx]) <= expert_idx:
                all_acts[layer_idx].append([])
            all_acts[layer_idx][expert_idx].append(out)
        except Exception as e:
            print(f"Error in hook (layer {layer_idx} expert {expert_idx}): {e}")
    return hook


def attach_cka_hooks(model, all_acts: Dict[int, List[List[torch.Tensor]]]):
    hooks = []
    for idx, layer in enumerate(model.language.model.layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            num_experts = len(layer.mlp.experts)
            if idx not in all_acts:
                all_acts[idx] = [[] for _ in range(num_experts)]
            for e_idx, expert in enumerate(layer.mlp.experts):
                try:
                    h = expert.register_forward_hook(_make_expert_hook(all_acts, idx, e_idx))
                    hooks.append(h)
                except Exception as e:
                    print(f"Failed to register hook on layer {idx} expert {e_idx}: {e}")
    return hooks


def remove_hooks(hooks):
    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass


# -------------------------
# Main
# -------------------------
def main(args):
    dtype = torch.bfloat16

    # Load model and processor
    processor = DeepseekVLV2Processor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer
    model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=dtype
    ).cuda().eval()

    # Ensure common config attributes exist (fixes 'DeepseekVLV2Config' object has no attribute 'use_cache')
    ensure_config_attrs(model, defaults={"use_cache": True})

    # Prepare container for activations and attach hooks
    all_acts: Dict[int, List[List[torch.Tensor]]] = {}
    hooks = attach_cka_hooks(model, all_acts)
    total_hooks = sum(len(v) for v in all_acts.values())
    print(f"Attached hooks for {total_hooks} experts across {len(all_acts)} MoE layers.")

    # Load annotations
    with open(args.annotation_file, 'r') as f:
        dataset = json.load(f)
        annotations = dataset.get("data", [])

    processed_for_cka = 0
    processed_questions = 0
    results = []
    correct = 0

    # deterministic subsampling
    torch.manual_seed(0)
    random.seed(0)

    for item in annotations:
        if args.max_questions is not None and processed_questions >= args.max_questions:
            print(f"Reached max_questions={args.max_questions}, stopping.")
            break

        question = item.get("question", "")
        image_file = os.path.splitext(item.get("image", ""))[0]
        image_file = image_file.replace("documents/", "")
        ground_truths = item.get("answers", [])

        images = load_pil_images_from_ids([image_file], args.image_folder)
        if not images:
            print(f"No valid images for question: {question}")
            processed_questions += 1
            continue

        conversation = [
            {
                "role": "<|User|>",
                "content": "<image>\n" * len(images) + question,
                "images": [os.path.join(args.image_folder, f"{image_file}.png")],
            },
            {"role": "<|Assistant|>", "content": ""}
        ]

        prepare_inputs = processor(
            conversations=conversation,
            images=images,
            force_batchify=True,
            system_prompt=""
        ).to(model.device, dtype=dtype)

        # --- explicit forward to trigger hooks, with retry for missing config attrs ---
        with torch.no_grad():
            forward_kwargs = dict(
                input_ids=prepare_inputs.input_ids,
                attention_mask=prepare_inputs.attention_mask,
                images=prepare_inputs.images,
                images_seq_mask=prepare_inputs.images_seq_mask,
                images_spatial_crop=prepare_inputs.images_spatial_crop,
                return_dict=True,
            )
            try:
                _ = model(**forward_kwargs)
            except Exception as e:
                # Try to detect attribute missing on config and patch then retry once
                msg = str(e)
                m = re.search(r"'DeepseekVLV2Config' object has no attribute '(.+?)'", msg)
                if m:
                    attr = m.group(1)
                    try:
                        setattr(model.config, attr, True)
                        print(f"Detected missing model.config.{attr}; patched to True and retrying forward.")
                        try:
                            _ = model(**forward_kwargs)
                        except Exception as e2:
                            print(f"Retry forward failed after patching {attr}: {e2}")
                    except Exception as e2:
                        print(f"Failed to patch model.config.{attr}: {e2}")
                else:
                    print(f"Warning: forward pass for activation collection failed: {e}")

            # now call generate as before
            outputs = model.generate(
                input_ids=prepare_inputs.input_ids,
                attention_mask=prepare_inputs.attention_mask,
                images=prepare_inputs.images,
                images_seq_mask=prepare_inputs.images_seq_mask,
                images_spatial_crop=prepare_inputs.images_spatial_crop,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                use_cache=True,
            )

        # decode answer robustly
        try:
            out_seq = outputs[0]
            prompt_len = prepare_inputs.input_ids.shape[1]
            if out_seq.dim() == 1:
                decoded_ids = out_seq[prompt_len:].cpu().tolist()
            else:
                decoded_ids = out_seq[0][prompt_len:].cpu().tolist()
            answer = tokenizer.decode(decoded_ids, skip_special_tokens=True)
        except Exception:
            answer = tokenizer.decode(outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(), skip_special_tokens=True)

        norm_pred = normalize_text(answer)
        norm_truths = [normalize_text(t) for t in ground_truths]
        is_correct = norm_pred in norm_truths
        if is_correct:
            correct += 1

        print(f"\n?? Question: {question}")
        print(f"?? Image: {image_file}")
        print(f"? Predicted: {answer}")
        print(f"   ?? Normalized Pred: {norm_pred}")
        print(f"?? Ground Truths: {ground_truths}")
        print(f"   ?? Normalized Truths: {norm_truths}")
        print(f"? Exact Match: {is_correct}")

        # Count this prompt for CKA collection
        if processed_for_cka < args.num_prompts_for_cka:
            processed_for_cka += 1
            if processed_for_cka >= args.num_prompts_for_cka:
                print(f"Reached num_prompts_for_cka={args.num_prompts_for_cka}; removing hooks to stop further CKA collection.")
                remove_hooks(hooks)
                hooks = []

        processed_questions += 1

        results.append({
            'question': question,
            'image': image_file,
            'predicted_answer': answer,
            'normalized_pred': norm_pred,
            'ground_truths': ground_truths,
            'normalized_truths': norm_truths,
            'exact_match': is_correct
        })

    total = len(results)
    accuracy = correct / total if total > 0 else 0.0
    print(f"\n=== Evaluation ===")
    print(f"Total questions processed: {processed_questions}")
    print(f"Total QA results saved: {total}")
    print(f"Exact Match Accuracy: {accuracy:.4f}")

    # -------------------------
    # Aggregate activations and compute CKA (exact RBF)
    # -------------------------
    if all_acts:
        print("\nAggregating activations and computing CKA (exact RBF via compute_layer_cka)...")
        all_cka_final = {}
        similar_info = {}

        for idx, expert_lists in all_acts.items():
            mats = []
            counts = []
            for e_idx, el in enumerate(expert_lists):
                if not el:
                    mats.append(None)
                    counts.append(0)
                    continue
                try:
                    m = torch.cat(el, dim=0)  # (total_samples, features) on CPU float
                except Exception as e:
                    print(f"Layer {idx} expert {e_idx} concat failed: {e}")
                    mats.append(None)
                    counts.append(0)
                    continue
                mats.append(m)
                counts.append(m.shape[0])

            print(f"Layer {idx} expert sample counts: {counts}")

            valid = [(i, m) for i, m in enumerate(mats) if (m is not None and m.shape[0] >= 2)]
            if len(valid) < 2:
                print(f"Layer {idx}: not enough valid experts (valid={len(valid)}). Skipping CKA.")
                continue

            Nmin = min(m.shape[0] for _, m in valid)
            Ncap = min(Nmin, args.cka_max_samples)
            mats_tr = []
            valid_indices = []
            for i, m in valid:
                if m.shape[0] > Ncap:
                    perm = torch.randperm(m.shape[0])[:Ncap]
                    m_sub = m[perm]
                else:
                    m_sub = m[:Ncap]
                mats_tr.append(m_sub)
                valid_indices.append(i)

            try:
                # stack -> (Ncap, E_valid, D)
                stacked = torch.stack(mats_tr, dim=1).contiguous().float().cpu()
            except Exception as e:
                print(f"Layer {idx}: stacking mats_tr failed: {e}")
                continue

            print(f"Layer {idx}: calling compute_layer_cka with shape {stacked.shape} (N, E, D). This may be slow for large N.")

            try:
                cka_mat = compute_layer_cka(stacked)  # expects (N, E, D)
            except Exception as e:
                print(f"Layer {idx}: compute_layer_cka raised an exception: {e}")
                try:
                    alt = stacked.permute(1, 0, 2)  # (E, N, D)
                    cka_mat = compute_layer_cka(alt)
                except Exception as e2:
                    print(f"Layer {idx}: compute_layer_cka failed with alternative ordering as well: {e2}")
                    cka_mat = None

            if cka_mat is None:
                print(f"Layer {idx}: CKA computation failed, skipping layer.")
                continue

            if not isinstance(cka_mat, torch.Tensor):
                try:
                    cka_mat = torch.tensor(cka_mat)
                except Exception:
                    pass

            # Map to full expert index matrix
            E_full = len(expert_lists)
            full_mat = torch.full((E_full, E_full), float("nan"), dtype=cka_mat.dtype)
            for out_i, i_idx in enumerate(valid_indices):
                for out_j, j_idx in enumerate(valid_indices):
                    full_mat[i_idx, j_idx] = cka_mat[out_i, out_j]

            all_cka_final[idx] = full_mat

            # Find similar pairs >= threshold
            threshold = args.cka_similarity_threshold
            pairs = []
            E_valid = len(valid_indices)
            for a in range(E_valid):
                for b in range(a + 1, E_valid):
                    sim_val = float(cka_mat[a, b])
                    if sim_val >= threshold:
                        pairs.append((valid_indices[a], valid_indices[b], sim_val))
            similar_info[idx] = pairs
            print(f"Layer {idx}: computed CKA (valid_experts={E_valid}); pairs >= {threshold}: {len(pairs)}")

        if all_cka_final:
            try:
                save_cka_heatmaps(all_cka_final, output_dir=args.heatmap_output_dir)
                print(f"Saved CKA heatmaps to {args.heatmap_output_dir}")
            except Exception as e:
                print(f"save_cka_heatmaps failed: {e}")

        # Print similar pairs
        print("\n=== Similar expert pairs (threshold = {:.2f}) ===".format(args.cka_similarity_threshold))
        any_pairs = False
        for layer, pairs in similar_info.items():
            if pairs:
                any_pairs = True
                print(f"Layer {layer}:")
                for i, j, v in pairs:
                    print(f"  Expert {i} <-> Expert {j} : similarity = {v:.4f}")
        if not any_pairs:
            print("No expert pairs exceeded the similarity threshold.")

    # cleanup
    if hooks:
        remove_hooks(hooks)

    # Save QA results + similar info
    if args.output_file:
        out = {'results': results, 'accuracy': accuracy}
        out['cka_similar_pairs'] = {str(k): v for k, v in similar_info.items()}
        try:
            with open(args.output_file, 'w') as fout:
                json.dump(out, fout, indent=2)
            print(f"Saved QA + CKA similar results to {args.output_file}")
        except Exception as e:
            print(f"Failed to save output file {args.output_file}: {e}")


# -------------------------
# CLI
# -------------------------
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_folder", type=str, default="spdocvqa_images/")
    parser.add_argument("--annotation_file", type=str, default="spdocvqa_annotations/val.json")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--num_prompts_for_cka", type=int, default=200,
                        help="Number of prompts to collect activations for CKA; hooks removed after this.")
    parser.add_argument("--heatmap_output_dir", type=str, default="cka_heatmaps")
    parser.add_argument("--cka_threshold", type=float, default=0.65)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--do_sample", type=bool, default=True)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--max_questions", type=int, default=None,
                        help="Maximum number of questions to process. If not set, process all.")
    parser.add_argument("--cka_max_samples", type=int, default=256,
                        help="Maximum samples per expert to use for exact RBF compute (caps N).")
    parser.add_argument("--cka_method", type=str, default="rbf", choices=["linear", "rbf"],
                        help="Method for CKA. 'rbf' uses compute_layer_cka (repo). 'linear' uses fast linear fallback.")
    parser.add_argument("--cka_similarity_threshold", type=float, default=0.6,
                        help="Show expert pairs with similarity >= this value (0..1).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

'''

'''

import os
import json
import math
import re
import torch
import random
from argparse import ArgumentParser
from typing import List, Dict
from PIL import Image
from difflib import SequenceMatcher

from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
from deepseek_vl2.utils.cka_utils import compute_layer_cka, save_cka_heatmaps, find_similar_experts

# -------------------------
# Similarity + normalization helpers
# -------------------------
SIM_THRESHOLD = 0.8

def normalize(a: str) -> str:
    return a.strip().lower()

def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

# -------------------------
# Utilities
# -------------------------
def load_pil_images_from_ids(image_ids: List[str], image_folder: str) -> List[Image.Image]:
    pil_images = []
    for image_id in image_ids:
        found = False
        for ext in [".png", ".jpg", ".jpeg"]:
            image_path = os.path.join(image_folder, f"{image_id}{ext}")
            if os.path.exists(image_path):
                try:
                    img = Image.open(image_path).convert("RGB")
                    pil_images.append(img)
                    found = True
                    break
                except Exception as e:
                    print(f"Error loading {image_path}: {e}")
        if not found:
            print(f"Missing image for ID: {image_id}")
    return pil_images

# -------------------------
# Helpers to patch model.config attributes
# -------------------------
def ensure_config_attrs(model, defaults: Dict[str, object] = None):
    defaults = defaults or {"use_cache": True}
    for k, v in defaults.items():
        if not hasattr(model.config, k):
            try:
                setattr(model.config, k, v)
                print(f"Patched model.config.{k} = {v}")
            except Exception as e:
                print(f"Failed to set model.config.{k}: {e}")

# -------------------------
# Hook helpers to collect expert activations
# -------------------------
def _make_expert_hook(all_acts: Dict[int, List[List[torch.Tensor]]], layer_idx: int, expert_idx: int):
    def hook(module, input, output):
        try:
            out = output[0] if isinstance(output, (list, tuple)) else output
            out = out.detach().cpu().float()
            if out.dim() > 2:
                out = out.view(out.shape[0], -1)
            if layer_idx not in all_acts:
                all_acts[layer_idx] = []
            while len(all_acts[layer_idx]) <= expert_idx:
                all_acts[layer_idx].append([])
            all_acts[layer_idx][expert_idx].append(out)
        except Exception as e:
            print(f"Error in hook (layer {layer_idx} expert {expert_idx}): {e}")
    return hook

def attach_cka_hooks(model, all_acts: Dict[int, List[List[torch.Tensor]]]):
    hooks = []
    for idx, layer in enumerate(model.language.model.layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            num_experts = len(layer.mlp.experts)
            if idx not in all_acts:
                all_acts[idx] = [[] for _ in range(num_experts)]
            for e_idx, expert in enumerate(layer.mlp.experts):
                try:
                    h = expert.register_forward_hook(_make_expert_hook(all_acts, idx, e_idx))
                    hooks.append(h)
                except Exception as e:
                    print(f"Failed to register hook on layer {idx} expert {e_idx}: {e}")
    return hooks

def remove_hooks(hooks):
    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

# -------------------------
# Main
# -------------------------
def main(args):
    dtype = torch.bfloat16

    processor = DeepseekVLV2Processor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer
    model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=dtype
    ).cuda().eval()

    ensure_config_attrs(model, defaults={"use_cache": True})

    all_acts: Dict[int, List[List[torch.Tensor]]] = {}
    hooks = attach_cka_hooks(model, all_acts)
    total_hooks = sum(len(v) for v in all_acts.values())
    print(f"Attached hooks for {total_hooks} experts across {len(all_acts)} MoE layers.")

    with open(args.annotation_file, 'r') as f:
        dataset = json.load(f)
        annotations = dataset.get("data", [])

    processed_for_cka = 0
    processed_questions = 0
    results = []

    # Counters for metrics
    correct = 0
    tp = 0
    fp = 0
    fn = 0

    torch.manual_seed(0)
    random.seed(0)

    for item in annotations:
        if args.max_questions is not None and processed_questions >= args.max_questions:
            print(f"Reached max_questions={args.max_questions}, stopping.")
            break

        question = item.get("question", "")
        image_file = os.path.splitext(item.get("image", ""))[0]
        image_file = image_file.replace("documents/", "")
        ground_truths = item.get("answers", [])

        images = load_pil_images_from_ids([image_file], args.image_folder)
        if not images:
            print(f"No valid images for question: {question}")
            processed_questions += 1
            continue

        conversation = [
            {
                "role": "<|User|>",
                "content": "<image>\n" * len(images) + question,
                "images": [os.path.join(args.image_folder, f"{image_file}.png")],
            },
            {"role": "<|Assistant|>", "content": ""}
        ]

        prepare_inputs = processor(
            conversations=conversation,
            images=images,
            force_batchify=True,
            system_prompt=""
        ).to(model.device, dtype=dtype)

        with torch.no_grad():
            forward_kwargs = dict(
                input_ids=prepare_inputs.input_ids,
                attention_mask=prepare_inputs.attention_mask,
                images=prepare_inputs.images,
                images_seq_mask=prepare_inputs.images_seq_mask,
                images_spatial_crop=prepare_inputs.images_spatial_crop,
                return_dict=True,
            )
            try:
                _ = model(**forward_kwargs)
            except Exception as e:
                msg = str(e)
                m = re.search(r"'DeepseekVLV2Config' object has no attribute '(.+?)'", msg)
                if m:
                    attr = m.group(1)
                    try:
                        setattr(model.config, attr, True)
                        print(f"Detected missing model.config.{attr}; patched to True and retrying forward.")
                        _ = model(**forward_kwargs)
                    except Exception as e2:
                        print(f"Retry forward failed after patching {attr}: {e2}")
                else:
                    print(f"Warning: forward pass for activation collection failed: {e}")

            outputs = model.generate(
                input_ids=prepare_inputs.input_ids,
                attention_mask=prepare_inputs.attention_mask,
                images=prepare_inputs.images,
                images_seq_mask=prepare_inputs.images_seq_mask,
                images_spatial_crop=prepare_inputs.images_spatial_crop,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                use_cache=True,
            )

        try:
            out_seq = outputs[0]
            prompt_len = prepare_inputs.input_ids.shape[1]
            decoded_ids = out_seq[prompt_len:].cpu().tolist() if out_seq.dim()==1 else out_seq[0][prompt_len:].cpu().tolist()
            answer = tokenizer.decode(decoded_ids, skip_special_tokens=True)
        except Exception:
            answer = tokenizer.decode(outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(), skip_special_tokens=True)

        norm_pred = normalize(answer)
        norm_truths = [normalize(t) for t in ground_truths]

        # Similarity-matching
        best_sim = max([sim(norm_pred, gt) for gt in norm_truths], default=0)
        is_correct = best_sim >= SIM_THRESHOLD

        if is_correct:
            correct += 1
            tp += 1
        else:
            fp += 1
            fn += 1

        print(f"\n?? Question: {question}")
        print(f"?? Image: {image_file}")
        print(f"? Predicted: {answer}")
        print(f"?? GT: {ground_truths} | Best Similarity = {best_sim:.2f} | Correct={is_correct}")

        if processed_for_cka < args.num_prompts_for_cka:
            processed_for_cka += 1
            if processed_for_cka >= args.num_prompts_for_cka:
                print(f"Reached num_prompts_for_cka={args.num_prompts_for_cka}; removing hooks to stop further CKA collection.")
                remove_hooks(hooks)
                hooks = []

        processed_questions += 1
        results.append({
            'question': question,
            'image': image_file,
            'predicted_answer': answer,
            'normalized_pred': norm_pred,
            'ground_truths': ground_truths,
            'normalized_truths': norm_truths,
            'similarity': best_sim,
            'correct': is_correct
        })

    total = len(results)
    accuracy = correct / total if total > 0 else 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n=== Evaluation ===")
    print(f"Total: {total}")
    print(f"Similarity-based Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # -------------------------
    # Aggregation for CKA
    # -------------------------
    if all_acts:
        print("\nAggregating activations and computing CKA (exact RBF via compute_layer_cka)...")
        # ... (same as your original aggregation code)
        # (intentionally omitted here just to keep focus on the similarity-based evaluation changes)

# -------------------------
# CLI
# -------------------------
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_folder", type=str, default="spdocvqa_images/")
    parser.add_argument("--annotation_file", type=str, default="spdocvqa_annotations/val.json")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--num_prompts_for_cka", type=int, default=200)
    parser.add_argument("--heatmap_output_dir", type=str, default="cka_heatmaps")
    parser.add_argument("--cka_threshold", type=float, default=0.65)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--do_sample", type=bool, default=True)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--cka_max_samples", type=int, default=256)
    parser.add_argument("--cka_method", type=str, default="rbf", choices=["linear", "rbf"])
    parser.add_argument("--cka_similarity_threshold", type=float, default=0.6)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
'''

'''
import os

# -------------- HF cache redirection --------------
os.environ["HF_HOME"] = "/data/stud/raza/DeepSeek-VL2/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/data/stud/raza/DeepSeek-VL2/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/data/stud/raza/DeepSeek-VL2/hf_cache"
# --------------------------------------------------

import json
import math
import re
import torch
import random
from argparse import ArgumentParser
from typing import List, Dict
from PIL import Image
from difflib import SequenceMatcher

from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
from deepseek_vl2.utils.cka_utils import compute_layer_cka, save_cka_heatmaps, find_similar_experts


SIM_THRESHOLD = 0.8

def normalize(a: str) -> str:
    return a.strip().lower()

def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

# -------------------------
# Utilities
# -------------------------
def load_pil_images_from_ids(image_ids: List[str], image_folder: str) -> List[Image.Image]:
    pil_images = []
    for image_id in image_ids:
        found = False
        for ext in [".png", ".jpg", ".jpeg"]:
            image_path = os.path.join(image_folder, f"{image_id}{ext}")
            if os.path.exists(image_path):
                try:
                    img = Image.open(image_path).convert("RGB")
                    pil_images.append(img)
                    found = True
                    break
                except Exception as e:
                    print(f"Error loading {image_path}: {e}")
        if not found:
            print(f"Missing image for ID: {image_id}")
    return pil_images

# -------------------------
# Helpers to patch model.config attributes
# -------------------------
def ensure_config_attrs(model, defaults: Dict[str, object] = None):
    defaults = defaults or {"use_cache": True}
    for k, v in defaults.items():
        if not hasattr(model.config, k):
            try:
                setattr(model.config, k, v)
                print(f"Patched model.config.{k} = {v}")
            except Exception as e:
                print(f"Failed to set model.config.{k}: {e}")

# -------------------------
# Hook helpers to collect expert activations
# -------------------------
def _make_expert_hook(all_acts: Dict[int, List[List[torch.Tensor]]], layer_idx: int, expert_idx: int):
    def hook(module, input, output):
        try:
            out = output[0] if isinstance(output, (list, tuple)) else output
            out = out.detach().cpu().float()
            if out.dim() > 2:
                out = out.view(out.shape[0], -1)
            if layer_idx not in all_acts:
                all_acts[layer_idx] = []
            while len(all_acts[layer_idx]) <= expert_idx:
                all_acts[layer_idx].append([])
            all_acts[layer_idx][expert_idx].append(out)
        except Exception as e:
            print(f"Error in hook (layer {layer_idx} expert {expert_idx}): {e}")
    return hook

def attach_cka_hooks(model, all_acts: Dict[int, List[List[torch.Tensor]]]):
    hooks = []
    for idx, layer in enumerate(model.language.model.layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            num_experts = len(layer.mlp.experts)
            if idx not in all_acts:
                all_acts[idx] = [[] for _ in range(num_experts)]
            for e_idx, expert in enumerate(layer.mlp.experts):
                try:
                    h = expert.register_forward_hook(_make_expert_hook(all_acts, idx, e_idx))
                    hooks.append(h)
                except Exception as e:
                    print(f"Failed to register hook on layer {idx} expert {e_idx}: {e}")
    return hooks

def remove_hooks(hooks):
    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

# -------------------------
# Main
# -------------------------
def main(args):
    dtype = torch.bfloat16

    processor = DeepseekVLV2Processor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer
    model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=dtype
    ).cuda().eval()

    ensure_config_attrs(model, defaults={"use_cache": True})

    all_acts: Dict[int, List[List[torch.Tensor]]] = {}
    hooks = attach_cka_hooks(model, all_acts)
    total_hooks = sum(len(v) for v in all_acts.values())
    print(f"Attached hooks for {total_hooks} experts across {len(all_acts)} MoE layers.")

    with open(args.annotation_file, 'r') as f:
        dataset = json.load(f)
        annotations = dataset.get("data", [])

    processed_for_cka = 0
    processed_questions = 0
    results = []

    # Counters for metrics
    correct = 0
    tp = 0
    fp = 0
    fn = 0

    torch.manual_seed(0)
    random.seed(0)

    for item in annotations:
        if args.max_questions is not None and processed_questions >= args.max_questions:
            print(f"Reached max_questions={args.max_questions}, stopping.")
            break

        question = item.get("question", "")
        image_file = os.path.splitext(item.get("image", ""))[0]
        image_file = image_file.replace("documents/", "")
        ground_truths = item.get("answers", [])

        images = load_pil_images_from_ids([image_file], args.image_folder)
        if not images:
            print(f"No valid images for question: {question}")
            processed_questions += 1
            continue

        conversation = [
            {
                "role": "<|User|>",
                "content": "<image>\n" * len(images) + question,
                "images": [os.path.join(args.image_folder, f"{image_file}.png")],
            },
            {"role": "<|Assistant|>", "content": ""}
        ]

        prepare_inputs = processor(
            conversations=conversation,
            images=images,
            force_batchify=True,
            system_prompt=""
        ).to(model.device, dtype=dtype)
        
        
        
        

        with torch.no_grad():
            forward_kwargs = dict(
                input_ids=prepare_inputs.input_ids,
                attention_mask=prepare_inputs.attention_mask,
                images=prepare_inputs.images,
                images_seq_mask=prepare_inputs.images_seq_mask,
                images_spatial_crop=prepare_inputs.images_spatial_crop,
                return_dict=True,
            )
            try:
                _ = model(**forward_kwargs)
            except Exception as e:
                msg = str(e)
                m = re.search(r"'DeepseekVLV2Config' object has no attribute '(.+?)'", msg)
                if m:
                    attr = m.group(1)
                    try:
                        setattr(model.config, attr, True)
                        print(f"Detected missing model.config.{attr}; patched to True and retrying forward.")
                        _ = model(**forward_kwargs)
                    except Exception as e2:
                        print(f"Retry forward failed after patching {attr}: {e2}")
                else:
                    print(f"Warning: forward pass for activation collection failed: {e}")

            outputs = model.generate(
                input_ids=prepare_inputs.input_ids,
                attention_mask=prepare_inputs.attention_mask,
                images=prepare_inputs.images,
                images_seq_mask=prepare_inputs.images_seq_mask,
                images_spatial_crop=prepare_inputs.images_spatial_crop,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                use_cache=True,
            )

        try:
            out_seq = outputs[0]
            prompt_len = prepare_inputs.input_ids.shape[1]
            decoded_ids = out_seq[prompt_len:].cpu().tolist() if out_seq.dim()==1 else out_seq[0][prompt_len:].cpu().tolist()
            answer = tokenizer.decode(decoded_ids, skip_special_tokens=True)
        except Exception:
            answer = tokenizer.decode(outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(), skip_special_tokens=True)

        norm_pred = normalize(answer)
        norm_truths = [normalize(t) for t in ground_truths]
        best_sim = max([sim(norm_pred, gt) for gt in norm_truths], default=0)
        is_correct = best_sim >= SIM_THRESHOLD

        if is_correct:
            correct += 1
            tp += 1
        else:
            fp += 1
            fn += 1

        print(f"\n?? Question: {question}")
        print(f"?? Image: {image_file}")
        print(f"? Predicted: {answer}")
        print(f"?? GT: {ground_truths} | Best Similarity = {best_sim:.2f} | Correct={is_correct}")

        if processed_for_cka < args.num_prompts_for_cka:
            processed_for_cka += 1
            if processed_for_cka >= args.num_prompts_for_cka:
                print(f"Reached num_prompts_for_cka={args.num_prompts_for_cka}; removing hooks to stop further CKA collection.")
                remove_hooks(hooks)
                hooks = []

        processed_questions += 1
        results.append({
            'question': question,
            'image': image_file,
            'predicted_answer': answer,
            'normalized_pred': norm_pred,
            'ground_truths': ground_truths,
            'normalized_truths': norm_truths,
            'similarity': best_sim,
            'correct': is_correct
        })

    total = len(results)
    accuracy = correct / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n=== Evaluation ===")
    print(f"Total: {total}")
    print(f"Similarity-based Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    if all_acts:
        print("\nAggregating activations and computing CKA...")
       

# -------------------------
# CLI
# -------------------------
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_folder", type=str, default="spdocvqa_images/")
    parser.add_argument("--annotation_file", type=str, default="spdocvqa_annotations/val.json")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--num_prompts_for_cka", type=int, default=200)
    parser.add_argument("--heatmap_output_dir", type=str, default="cka_heatmaps")
    parser.add_argument("--cka_threshold", type=float, default=0.65)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--do_sample", type=bool, default=True)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--cka_max_samples", type=int, default=256)
    parser.add_argument("--cka_method", type=str, default="rbf", choices=["linear", "rbf"])
    parser.add_argument("--cka_similarity_threshold", type=float, default=0.6)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
'''

import os

# -------------- HF cache redirection --------------
os.environ["HF_HOME"] = "/data/stud/raza/DeepSeek-VL2/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/data/stud/raza/DeepSeek-VL2/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/data/stud/raza/DeepSeek-VL2/hf_cache"
# --------------------------------------------------

import json
import math
import re
import torch
import random
from argparse import ArgumentParser
from typing import List, Dict
from PIL import Image
from difflib import SequenceMatcher

from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor

SIM_THRESHOLD = 0.8


def normalize(a: str) -> str:
    return a.strip().lower()


def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


# --- custom Levenshtein distance (pure python) ---
def levenshtein(a: str, b: str) -> int:
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            ins = dp[j] + 1
            dele = dp[j - 1] + 1
            sub = prev + (0 if a[i - 1] == b[j - 1] else 1)
            dp[j] = min(ins, dele, sub)
            prev = cur
    return dp[n]


# -------------------------
# Utilities
# -------------------------
def load_pil_images_from_ids(image_ids: List[str], image_folder: str) -> List[Image.Image]:
    pil_images = []
    for image_id in image_ids:
        found = False
        for ext in [".png", ".jpg", ".jpeg"]:
            image_path = os.path.join(image_folder, f"{image_id}{ext}")
            if os.path.exists(image_path):
                try:
                    img = Image.open(image_path).convert("RGB")
                    pil_images.append(img)
                    found = True
                    break
                except Exception as e:
                    print(f"Error loading {image_path}: {e}")
        if not found:
            print(f"Missing image for ID: {image_id}")
    return pil_images


# -------------------------
# Helpers to patch model.config attributes
# -------------------------
def ensure_config_attrs(model, defaults: Dict[str, object] = None):
    defaults = defaults or {"use_cache": True}
    for k, v in defaults.items():
        if not hasattr(model.config, k):
            try:
                setattr(model.config, k, v)
                print(f"Patched model.config.{k} = {v}")
            except Exception as e:
                print(f"Failed to set model.config.{k}: {e}")


# -------------------------
# Main
# -------------------------
def main(args):
    dtype = torch.bfloat16

    processor = DeepseekVLV2Processor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer
    model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto"  # <-- splits across all GPUs
    ).eval()
    ensure_config_attrs(model, defaults={"use_cache": True})

    with open(args.annotation_file, 'r') as f:
        dataset = json.load(f)
        annotations = dataset.get("data", [])

    processed_questions = 0
    results = []

    # Counters for metrics
    correct = 0
    tp = 0
    fp = 0
    fn = 0
    anls_total = 0  # to accumulate ANLS per question

    torch.manual_seed(0)
    random.seed(0)

    for item in annotations:
        if args.max_questions is not None and processed_questions >= args.max_questions:
            print(f"Reached max_questions={args.max_questions}, stopping.")
            break

        question = item.get("question", "")
        image_file = os.path.splitext(item.get("image", ""))[0]
        image_file = image_file.replace("documents/", "")
        ground_truths = item.get("answers", [])

        images = load_pil_images_from_ids([image_file], args.image_folder)
        if not images:
            print(f"No valid images for question: {question}")
            processed_questions += 1
            continue

        conversation = [
            {
                "role": "<|User|>",
                "content": "<image>\n" * len(images) + question,
                "images": [os.path.join(args.image_folder, f"{image_file}.png")],
            },
            {"role": "<|Assistant|>", "content": ""}
        ]

        prepare_inputs = processor(
            conversations=conversation,
            images=images,
            force_batchify=True,
            system_prompt=""
        ).to(model.device, dtype=dtype)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=prepare_inputs.input_ids,
                attention_mask=prepare_inputs.attention_mask,
                images=prepare_inputs.images,
                images_seq_mask=prepare_inputs.images_seq_mask,
                images_spatial_crop=prepare_inputs.images_spatial_crop,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                use_cache=True,
            )

        try:
            out_seq = outputs[0]
            prompt_len = prepare_inputs.input_ids.shape[1]
            decoded_ids = out_seq[prompt_len:].cpu().tolist() if out_seq.dim() == 1 else out_seq[0][prompt_len:].cpu().tolist()
            answer = tokenizer.decode(decoded_ids, skip_special_tokens=True)
        except Exception:
            answer = tokenizer.decode(outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(), skip_special_tokens=True)

        norm_pred = normalize(answer)
        norm_truths = [normalize(t) for t in ground_truths]
        best_sim = max([sim(norm_pred, gt) for gt in norm_truths], default=0)
        is_correct = best_sim >= SIM_THRESHOLD

        # compute per-question ANLS
        best_anls = 0
        for t in norm_truths:
            dist = levenshtein(norm_pred, t)
            max_len = max(len(norm_pred), len(t))
            score = 1 - dist / max_len if max_len > 0 else 0
            best_anls = max(best_anls, score)
        anls_total += best_anls

        if is_correct:
            correct += 1
            tp += 1
        else:
            fp += 1
            fn += 1

        print(f"\n?? Question: {question}")
        print(f"?? Image: {image_file}")
        print(f"? Predicted: {answer}")
        print(f"?? GT: {ground_truths} | Best Similarity = {best_sim:.2f} | Correct={is_correct}")
        print(f"ANLS for this question: {best_anls:.4f}")  # <-- print per question ANLS

        processed_questions += 1
        results.append({
            'question': question,
            'image': image_file,
            'predicted_answer': answer,
            'normalized_pred': norm_pred,
            'ground_truths': ground_truths,
            'normalized_truths': norm_truths,
            'similarity': best_sim,
            'correct': is_correct,
            'anls': best_anls
        })

    total = len(results)
    accuracy = correct / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    avg_anls = anls_total / total if total > 0 else 0.0

    print(f"\n=== Evaluation ===")
    print(f"Total: {total}")
    print(f"Similarity-based Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Average ANLS: {avg_anls:.4f}")


# -------------------------
# CLI
# -------------------------
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_folder", type=str, default="spdocvqa_images/")
    parser.add_argument("--annotation_file", type=str, default="spdocvqa_annotations/val.json")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--do_sample", type=bool, default=True)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--max_questions", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)


'''
import os


# -------------- HF cache redirection --------------
os.environ["HF_HOME"] = "/data/stud/raza/DeepSeek-VL2/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/data/stud/raza/DeepSeek-VL2/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/data/stud/raza/DeepSeek-VL2/hf_cache"
# --------------------------------------------------
import json
import math
import re
import torch
import random
from argparse import ArgumentParser
from typing import List, Dict
from PIL import Image
from difflib import SequenceMatcher

from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor

# Recommended to help avoid CUDA fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

SIM_THRESHOLD = 0.8


def normalize(a: str) -> str:
    return a.strip().lower()


def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


# --- custom Levenshtein distance (pure python) ---
def levenshtein(a: str, b: str) -> int:
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            ins = dp[j] + 1
            dele = dp[j - 1] + 1
            sub = prev + (0 if a[i - 1] == b[j - 1] else 1)
            dp[j] = min(ins, dele, sub)
            prev = cur
    return dp[n]


def load_pil_images_from_ids(image_ids: List[str], image_folder: str) -> List[Image.Image]:
    pil_images = []
    for image_id in image_ids:
        found = False
        for ext in [".png", ".jpg", ".jpeg"]:
            image_path = os.path.join(image_folder, f"{image_id}{ext}")
            if os.path.exists(image_path):
                try:
                    img = Image.open(image_path).convert("RGB")
                    pil_images.append(img)
                    found = True
                    break
                except Exception as e:
                    print(f"Error loading {image_path}: {e}")
        if not found:
            print(f"Missing image for ID: {image_id}")
    return pil_images


def ensure_config_attrs(model, defaults: Dict[str, object] = None):
    defaults = defaults or {"use_cache": True}
    for k, v in defaults.items():
        if not hasattr(model.config, k):
            try:
                setattr(model.config, k, v)
                print(f"Patched model.config.{k} = {v}")
            except Exception as e:
                print(f"Failed to set model.config.{k}: {e}")


def main(args):
    dtype = torch.bfloat16

    processor = DeepseekVLV2Processor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer

    # === modified to use multiple GPUs automatically ===
    model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto"  # <-- splits across all GPUs
    ).eval()

    ensure_config_attrs(model, defaults={"use_cache": True})

    with open(args.annotation_file, 'r') as f:
        dataset = json.load(f)
        annotations = dataset.get("data", [])

    processed_questions = 0
    results = []

    correct, tp, fp, fn = 0, 0, 0, 0
    anls_total = 0

    torch.manual_seed(0)
    random.seed(0)

    for item in annotations:
        if args.max_questions is not None and processed_questions >= args.max_questions:
            print(f"Reached max_questions={args.max_questions}, stopping.")
            break

        question = item.get("question", "")
        image_file = os.path.splitext(item.get("image", ""))[0]
        image_file = image_file.replace("documents/", "")
        ground_truths = item.get("answers", [])

        images = load_pil_images_from_ids([image_file], args.image_folder)
        if not images:
            print(f"No valid images for question: {question}")
            processed_questions += 1
            continue

        conversation = [
            {
                "role": "<|User|>",
                "content": "<image>\n" * len(images) + question,
                "images": [os.path.join(args.image_folder, f"{image_file}.png")],
            },
            {"role": "<|Assistant|>", "content": ""}
        ]

        prepare_inputs = processor(
            conversations=conversation,
            images=images,
            force_batchify=True,
            system_prompt=""
        ).to(model.device, dtype=dtype)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=prepare_inputs.input_ids,
                attention_mask=prepare_inputs.attention_mask,
                images=prepare_inputs.images,
                images_seq_mask=prepare_inputs.images_seq_mask,
                images_spatial_crop=prepare_inputs.images_spatial_crop,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                use_cache=True,
            )

        try:
            out_seq = outputs[0]
            prompt_len = prepare_inputs.input_ids.shape[1]
            decoded_ids = out_seq[prompt_len:].cpu().tolist() if out_seq.dim() == 1 else out_seq[0][prompt_len:].cpu().tolist()
            answer = tokenizer.decode(decoded_ids, skip_special_tokens=True)
        except Exception:
            answer = tokenizer.decode(outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(), skip_special_tokens=True)

        norm_pred = normalize(answer)
        norm_truths = [normalize(t) for t in ground_truths]
        best_sim = max([sim(norm_pred, gt) for gt in norm_truths], default=0)
        is_correct = best_sim >= SIM_THRESHOLD

        best_anls = 0
        for t in norm_truths:
            dist = levenshtein(norm_pred, t)
            max_len = max(len(norm_pred), len(t))
            score = 1 - dist / max_len if max_len > 0 else 0
            best_anls = max(best_anls, score)
        anls_total += best_anls

        if is_correct:
            correct += 1
            tp += 1
        else:
            fp += 1
            fn += 1

        print(f"\n?? Question: {question}")
        print(f"?? Image: {image_file}")
        print(f"? Predicted: {answer}")
        print(f"?? GT: {ground_truths} | Best Similarity = {best_sim:.2f} | Correct={is_correct}")
        print(f"ANLS for this question: {best_anls:.4f}")

        processed_questions += 1
        results.append({
            'question': question,
            'image': image_file,
            'predicted_answer': answer,
            'normalized_pred': norm_pred,
            'ground_truths': ground_truths,
            'normalized_truths': norm_truths,
            'similarity': best_sim,
            'correct': is_correct,
            'anls': best_anls
        })

    total = len(results)
    accuracy = correct / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    avg_anls = anls_total / total if total > 0 else 0.0

    print(f"\n=== Evaluation ===")
    print(f"Total: {total}")
    print(f"Similarity-based Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Average ANLS: {avg_anls:.4f}")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_folder", type=str, default="spdocvqa_images/")
    parser.add_argument("--annotation_file", type=str, default="spdocvqa_annotations/val.json")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--do_sample", type=bool, default=True)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--max_questions", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
'''