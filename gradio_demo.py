import argparse
import datetime
import os
import shutil
import tempfile
import threading
import time
import traceback
from datetime import datetime
from typing import Tuple, List, Any, Dict

import einops
import gradio as gr
import numpy as np
import requests
import torch
from PIL import Image
from PIL import PngImagePlugin
from gradio_imageslider import ImageSlider

from SUPIR.models.SUPIR_model import SUPIRModel
from SUPIR.util import HWC3, upscale_image, convert_dtype
from SUPIR.util import create_SUPIR_model
from SUPIR.utils.compare import create_comparison_video
from SUPIR.utils.face_restoration_helper import FaceRestoreHelper
from SUPIR.utils.model_fetch import get_model
from SUPIR.utils.status_container import StatusContainer, MediaData
from llava.llava_agent import LLavaAgent
import ui_helpers
from ui_helpers import is_video, extract_video, compile_video, is_image, get_video_params, printt

SUPIR_REVISION = "v40"

parser = argparse.ArgumentParser()
parser.add_argument("--ip", type=str, default='127.0.0.1', help="IP address for the server to listen on.")
parser.add_argument("--share", type=str, default=False, help="Set to True to share the app publicly.")
parser.add_argument("--port", type=int, help="Port number for the server to listen on.")
parser.add_argument("--log_history", action='store_true', default=False, help="Enable logging of request history.")
parser.add_argument("--loading_half_params", action='store_true', default=False,
                    help="Enable loading model parameters in half precision to reduce memory usage.")
parser.add_argument("--use_tile_vae", action='store_true', default=False,
                    help="Enable tiling for the VAE to handle larger images with limited memory.")
parser.add_argument("--use_fast_tile", action='store_true', default=False,
                    help="Use a faster tile encoding/decoding, may impact quality.")
parser.add_argument("--encoder_tile_size", type=int, default=512,
                    help="Tile size for the encoder. Larger sizes may improve quality but require more memory.")
parser.add_argument("--decoder_tile_size", type=int, default=64,
                    help="Tile size for the decoder. Larger sizes may improve quality but require more memory.")
parser.add_argument("--load_8bit_llava", action='store_true', default=False,
                    help="Load the LLAMA model in 8-bit precision to save memory.")
parser.add_argument("--load_4bit_llava", action='store_true', default=True,
                    help="Load the LLAMA model in 4-bit precision to significantly reduce memory usage.")
parser.add_argument("--ckpt", type=str, default='Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors',
                    help="Path to the checkpoint file for the model.")
parser.add_argument("--ckpt_browser", action='store_true', default=True, help="Enable a checkpoint selection dropdown.")
parser.add_argument("--ckpt_dir", type=str, default='models/checkpoints',
                    help="Directory where model checkpoints are stored.")
parser.add_argument("--theme", type=str, default='default',
                    help="Theme for the UI. Use 'default' or specify a custom theme.")
parser.add_argument("--open_browser", action='store_true', default=True,
                    help="Automatically open the web browser when the server starts.")
parser.add_argument("--outputs_folder", type=str, default='outputs', help="Folder where output files will be saved.")
parser.add_argument("--debug", action='store_true', default=False,
                    help="Enable debug mode, disables open_browser, and adds ui buttons for testing elements.")

args = parser.parse_args()
ui_helpers.ui_args = args

total_vram = 100000
auto_unload = False
if torch.cuda.is_available():
    # Get total GPU memory
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print("Total VRAM: ", total_vram, "GB")
    # If total VRAM <= 12GB, set auto_unload to True
    auto_unload = total_vram <= 12

    if total_vram <= 24:
        args.loading_half_params = True
        args.use_tile_vae = True
        print("Loading half params")

server_ip = args.ip
if args.debug:
    args.open_browser = False

if args.ckpt_dir == "models/checkpoints":
    args.ckpt_dir = os.path.join(os.path.dirname(__file__), args.ckpt_dir)
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir, exist_ok=True)

if torch.cuda.device_count() >= 2:
    SUPIR_device = 'cuda:0'
    LLaVA_device = 'cuda:1'
elif torch.cuda.device_count() == 1:
    SUPIR_device = 'cuda:0'
    LLaVA_device = 'cuda:0'
else:
    SUPIR_device = 'cpu'
    LLaVA_device = 'cpu'

face_helper = None
model: SUPIRModel = None
llava_agent = None
models_loaded = False
unique_counter = 0
status_container = StatusContainer()

# Store this globally so we can update variables more easily
elements_dict = {}

single_process = False
is_processing = False
last_used_checkpoint = None


def refresh_models_click():
    new_model_list = list_models()
    return gr.update(choices=new_model_list)


def refresh_styles_click():
    new_style_list = list_styles()
    style_list = list(new_style_list.keys())
    return gr.update(choices=style_list)


def select_style(style_name, current_prompt=None, values=False):
    print(f"Selected style: {style_name}")
    style_list = list_styles()
    print(f"Selected style: {style_name}")

    if style_name in style_list.keys():
        style_pos, style_neg, style_llava = style_list[style_name]
        if values:
            return style_pos, style_neg, style_llava
        return gr.update(value=style_pos), gr.update(value=style_neg), gr.update(value=style_llava)
    if values:
        return "", "", ""
    return gr.update(value=""), gr.update(value=""), gr.update(value="")


def open_folder():
    open_folder_path = os.path.abspath(args.outputs_folder)
    os.startfile(open_folder_path)


def list_models():
    model_dir = args.ckpt_dir
    output = []
    if os.path.exists(model_dir):
        output = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if
                  f.endswith('.safetensors') or f.endswith('.ckpt')]
    else:
        local_model_dir = os.path.join(os.path.dirname(__file__), args.ckpt_dir)
        if os.path.exists(local_model_dir):
            output = [os.path.join(local_model_dir, f) for f in os.listdir(local_model_dir) if
                      f.endswith('.safetensors') or f.endswith('.ckpt')]
    if os.path.exists(args.ckpt) and args.ckpt not in output:
        output.append(args.ckpt)
    else:
        if os.path.exists(os.path.join(os.path.dirname(__file__), args.ckpt)):
            output.append(os.path.join(os.path.dirname(__file__), args.ckpt))
    # Sort the models
    output = [os.path.basename(f) for f in output]
    # Ensure the values are unique
    output = list(set(output))
    output.sort()
    return output


def get_ckpt_path(ckpt_path):
    if os.path.exists(ckpt_path):
        return ckpt_path
    else:
        if os.path.exists(args.ckpt_dir):
            return os.path.join(args.ckpt_dir, ckpt_path)
        local_model_dir = os.path.join(os.path.dirname(__file__), args.ckpt_dir)
        if os.path.exists(local_model_dir):
            return os.path.join(local_model_dir, ckpt_path)
    return None


def list_styles():
    styles_path = os.path.join(os.path.dirname(__file__), 'styles')
    output = {}
    style_files = []
    llava_prompt = default_llava_prompt
    for root, dirs, files in os.walk(styles_path):
        for file in files:
            if file.endswith('.csv'):
                style_files.append(os.path.join(root, file))
    for style_file in style_files:
        with open(style_file, 'r') as f:
            lines = f.readlines()
            # Parse lines, skipping the first line
            for line in lines[1:]:
                line = line.strip()
                if len(line) > 0:
                    name = line.split(',')[0]
                    cap_line = line.replace(name + ',', '')
                    captions = cap_line.split('","')
                    if len(captions) >= 2:
                        positive_prompt = captions[0].replace('"', '')
                        negative_prompt = captions[1].replace('"', '')
                        if "{prompt}" in positive_prompt:
                            positive_prompt = positive_prompt.replace("{prompt}", "")

                        if "{prompt}" in negative_prompt:
                            negative_prompt = negative_prompt.replace("{prompt}", "")

                        if len(captions) == 3:
                            llava_prompt = captions[2].replace('"', "")

                        output[name] = (positive_prompt, negative_prompt, llava_prompt)

    return output


def selected_model():
    models = list_models()
    target_model = args.ckpt
    if os.path.basename(target_model) in models:
        return target_model
    else:
        if len(models) > 0:
            return models[0]
    return None


def load_face_helper():
    global face_helper
    if face_helper is None:
        face_helper = FaceRestoreHelper(
            device='cpu',
            upscale_factor=1,
            face_size=1024,
            use_parse=True,
            det_model='retinaface_resnet50'
        )


def load_model(selected_model, selected_checkpoint, sampler='DPMPP2M', device='cpu', progress=gr.Progress()):
    global model, last_used_checkpoint

    # Determine the need for model loading or updating
    need_to_load_model = last_used_checkpoint is None or last_used_checkpoint != selected_checkpoint
    need_to_update_model = selected_model != (model.current_model if model else None)
    if need_to_update_model:
        del model
        model = None

    # Resolve checkpoint path
    checkpoint_paths = [
        selected_checkpoint,
        os.path.join(args.ckpt_dir, selected_checkpoint),
        os.path.join(os.path.dirname(__file__), args.ckpt_dir, selected_checkpoint)
    ]
    checkpoint_use = next((path for path in checkpoint_paths if os.path.exists(path)), None)
    if checkpoint_use is None:
        raise FileNotFoundError(f"Checkpoint {selected_checkpoint} not found.")

    # Check if we need to load a new model
    if need_to_load_model or model is None:
        torch.cuda.empty_cache()
        last_used_checkpoint = checkpoint_use
        model_cfg = "options/SUPIR_v0_tiled.yaml" if args.use_tile_vae else "options/SUPIR_v0.yaml"
        model = create_SUPIR_model(model_cfg, supir_sign=selected_model[-1], device=device, ckpt=checkpoint_use,
                                   sampler=sampler)
        model.current_model = selected_model
        if args.loading_half_params:
            model = model.half()
        if args.use_tile_vae:
            model.init_tile_vae(encoder_tile_size=512, decoder_tile_size=64, use_fast=args.use_fast_tile)
        if progress is not None:
            progress(1, desc="SUPIR loaded.")


def load_llava():
    global llava_agent
    if llava_agent is None:
        llava_path = get_model('liuhaotian/llava-v1.5-7b')
        llava_agent = LLavaAgent(llava_path, device=LLaVA_device, load_8bit=args.load_8bit_llava,
                                 load_4bit=args.load_4bit_llava)


def unload_llava():
    global llava_agent
    if args.load_4bit_llava or args.load_8bit_llava:
        printt("Clearing LLaVA.")
        clear_llava()
        printt("LLaVA cleared.")
    else:
        printt("Unloading LLaVA.")
        llava_agent = llava_agent.to('cpu')
        torch.cuda.empty_cache()
        printt("LLaVA unloaded.")


def clear_llava():
    global llava_agent
    del llava_agent
    llava_agent = None
    torch.cuda.empty_cache()


def all_to_cpu_background():
    global face_helper, model, llava_agent
    printt("Moving all to CPU")
    if face_helper is not None:
        face_helper = face_helper.to('cpu')
        printt("Face helper moved to CPU")
    if model is not None:
        model = model.to('cpu')
        model.move_to('cpu')
        printt("Model moved to CPU")
    if llava_agent is not None:
        unload_llava()
    printt("All moved to CPU")


def all_to_cpu():
    cpu_thread = threading.Thread(target=all_to_cpu_background)
    cpu_thread.start()


def to_gpu(elem_to_load, device):
    if elem_to_load is not None:
        elem_to_load = elem_to_load.to(device)
        if getattr(elem_to_load, 'move_to', None):
            elem_to_load.move_to(device)
        torch.cuda.set_device(device)
    return elem_to_load


def update_model_settings(model_type, param_setting):
    """
    Returns a series of gr.updates with settings based on the model type.
    If 'model_type' contains 'lightning', it uses the settings for a 'lightning' SDXL model.
    Otherwise, it uses the settings for a normal SDXL model.
    s_cfg_Quality, spt_linear_CFG_Quality, s_cfg_Fidelity, spt_linear_CFG_Fidelity, edm_steps
    """
    # Default settings for a "lightning" SDXL model
    lightning_settings = {
        's_cfg_Quality': 2.0,
        'spt_linear_CFG_Quality': 2.0,
        's_cfg_Fidelity': 1.5,
        'spt_linear_CFG_Fidelity': 1.5,
        'edm_steps': 8
    }

    # Default settings for a normal SDXL model
    normal_settings = {
        's_cfg_Quality': 7.5,
        'spt_linear_CFG_Quality': 4.0,
        's_cfg_Fidelity': 4.0,
        'spt_linear_CFG_Fidelity': 1.0,
        'edm_steps': 50
    }

    # Choose the settings based on the model type
    settings = lightning_settings if 'Lightning' in model_type else normal_settings

    if param_setting == "Quality":
        s_cfg = settings['s_cfg_Quality']
        spt_linear_CFG = settings['spt_linear_CFG_Quality']
    else:
        s_cfg = settings['s_cfg_Fidelity']
        spt_linear_CFG = settings['spt_linear_CFG_Fidelity']

    return gr.update(value=s_cfg), gr.update(value=spt_linear_CFG), gr.update(value=settings['edm_steps'])


def update_inputs(input_file, upscale_amount):
    file_input = gr.update(visible=True)
    image_input = gr.update(visible=False)
    video_input = gr.update(visible=False)
    res_output = gr.update(value="")
    if is_image(input_file):
        image_input = gr.update(visible=True, value=input_file)
        file_input = gr.update(visible=False)
        res_output = gr.update(value=update_target_resolution(input_file, upscale_amount))
    elif is_video(input_file):
        video_input = gr.update(visible=True, value=input_file)
        file_input = gr.update(visible=False)
        res_output = gr.update(value=update_target_resolution(input_file, upscale_amount))
    elif input_file is None:
        file_input = gr.update(visible=True, value=None)
    return file_input, image_input, video_input, res_output


def update_target_resolution(img, do_upscale):
    # Read the input image dimensions
    if img is None:
        return ""
    if is_image(img):
        with Image.open(img) as img:
            width, height = img.size
            width_org, height_org = img.size
    elif is_video(img):
        params = get_video_params(img)
        width, height = params['width'], params['height']
        width_org, height_org = params['width'], params['height']
    else:
        return ""

    # Apply the do_upscale ratio
    width *= do_upscale
    height *= do_upscale

    # Ensure both dimensions are at least 1024 pixels
    if min(width, height) < 1024:
        do_upscale_factor = 1024 / min(width, height)
        width *= do_upscale_factor
        height *= do_upscale_factor

    # Update the target resolution label
    return f"Input: {int(width_org)}x{int(height_org)} px, {width_org * height_org / 1e6:.2f} Megapixels / Estimated Output Resolution: {int(width)}x{int(height)} px, {width * height / 1e6:.2f} Megapixels"


def read_image_metadata(image_path):
    if image_path is None:
        return
    # Check if the file exists
    if not os.path.exists(image_path):
        return "File does not exist."

    # Get the last modified date and format it
    last_modified_timestamp = os.path.getmtime(image_path)
    last_modified_date = datetime.fromtimestamp(last_modified_timestamp).strftime('%d %B %Y, %H:%M %p - UTC')

    # Open the image and extract metadata
    with Image.open(image_path) as img:
        width, height = img.size
        megapixels = (width * height) / 1e6

        metadata_str = f"Last Modified Date: {last_modified_date}\nMegapixels: {megapixels:.2f}\n"

        # Extract metadata based on image format
        if img.format == 'JPEG':
            exif_data = img._getexif()
            if exif_data:
                for tag, value in exif_data.items():
                    tag_name = Image.ExifTags.TAGS.get(tag, tag)
                    metadata_str += f"{tag_name}: {value}\n"
        else:
            metadata = img.info
            if metadata:
                for key, value in metadata.items():
                    metadata_str += f"{key}: {value}\n"
            else:
                metadata_str += "No additional metadata found."

    return metadata_str


def update_elements(status_label):
    print(f"Label changed: {status_label}")
    prompt_el = gr.update()
    result_gallery_el = gr.update(height=400)
    result_slider_el = gr.update(height=400)
    result_video_el = gr.update(height=400)
    comparison_video_el = gr.update(height=400, visible=False)
    seed = None
    face_gallery_items = []
    evt_id = ""
    if not is_processing:
        output_data = status_container.image_data
        if len(output_data) == 1:
            image_data = output_data[0]
            caption = image_data.caption
            print(f"Caption: {caption}")
            prompt_el = gr.update(value=caption)
            if len(image_data.outputs) > 0:
                outputs = image_data.outputs
                params = image_data.metadata_list
                if len(params) != len(outputs):
                    print("Mismatch in outputs and metadata list, using defaults.")
                    params = [status_container.process_params] * len(outputs)
                first_output = outputs[0]
                first_params = params[0]
                seed = first_params.get('seed', "")
                face_gallery_items = first_params.get('face_gallery', [])
                evt_id = first_params.get('event_id', "")
                if image_data.media_type == "image":
                    if image_data.comparison_video:
                        comparison_video_el = gr.update(value=image_data.comparison_video, visible=True)
                    result_slider_el = gr.update(value=[image_data.media_path, first_output], visible=True)
                    result_gallery_el = gr.update(value=None, visible=False)
                    result_video_el = gr.update(value=None, visible=False)
                elif image_data.media_type == "video":
                    prompt_el = gr.update(value="")
                    result_video_el = gr.update(value=first_output, visible=True)
                    result_gallery_el = gr.update(value=None, visible=False)
                    result_slider_el = gr.update(value=None, visible=False)
        elif len(output_data) > 1:
            first_output_data = output_data[0]
            if len(first_output_data.outputs):
                first_params = first_output_data.metadata_list[0] if first_output_data.metadata_list else status_container.process_params
                seed = first_params.get('seed', "")
                face_gallery_items = first_params.get('face_gallery', [])
                evt_id = first_params.get('event_id', "")
            all_outputs = []
            for output_data in output_data:
                all_outputs.extend(output_data.outputs)
            result_gallery_el = gr.update(value=all_outputs, visible=True)
            result_slider_el = gr.update(value=None, visible=False)
            result_video_el = gr.update(value=None, visible=False)
    seed_el = gr.update(value=seed)
    event_id_el = gr.update(value=evt_id)
    face_gallery_el = gr.update(value=face_gallery_items)

    return prompt_el, result_gallery_el, result_slider_el, result_video_el, comparison_video_el, event_id_el, seed_el, face_gallery_el


def populate_slider_single():
    # Fetch the image at http://www.marketingtool.online/en/face-generator/img/faces/avatar-1151ce9f4b2043de0d2e3b7826127998.jpg
    # and use it as the input image
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_path.write(requests.get(
        "http://www.marketingtool.online/en/face-generator/img/faces/avatar-1151ce9f4b2043de0d2e3b7826127998.jpg").content)
    temp_path.close()
    lowres_path = temp_path.name.replace('.jpg', '_lowres.jpg')
    with Image.open(temp_path.name) as img:
        current_dims = (img.size[0] // 2, img.size[1] // 2)
        resized_dims = (img.size[0] // 4, img.size[1] // 4)
        img = img.resize(current_dims)
        img.save(temp_path.name)
        img = img.resize(resized_dims)
        img.save(lowres_path)
    return (gr.update(value=[lowres_path, temp_path.name], visible=True,
                      elem_classes=["active", "preview_slider", "preview_box"]),
            gr.update(visible=False, value=None, elem_classes=["preview_box"]))


def populate_gallery():
    # Fetch the image at http://www.marketingtool.online/en/face-generator/img/faces/avatar-1151ce9f4b2043de0d2e3b7826127998.jpg
    # and use it as the input image
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_path.write(requests.get(
        "http://www.marketingtool.online/en/face-generator/img/faces/avatar-1151ce9f4b2043de0d2e3b7826127998.jpg").content)
    temp_path.close()
    lowres_path = temp_path.name.replace('.jpg', '_lowres.jpg')
    with Image.open(temp_path.name) as img:
        current_dims = (img.size[0] // 2, img.size[1] // 2)
        resized_dims = (img.size[0] // 4, img.size[1] // 4)
        img = img.resize(current_dims)
        img.save(temp_path.name)
        img = img.resize(resized_dims)
        img.save(lowres_path)
    return gr.update(value=[lowres_path, temp_path.name], visible=True,
                     elem_classes=["preview_box", "active"]), gr.update(visible=False, value=None,
                                                                        elem_classes=["preview_slider", "preview_box"])


def start_single_process(*element_values):
    global status_container, is_processing
    status_container = StatusContainer()
    status_container.is_batch = False
    values_dict = zip(elements_dict.keys(), element_values)
    values_dict = dict(values_dict)
    main_prompt = values_dict['main_prompt']
    # Delete input_image, prompt, batch_process_folder, outputs_folder from values_dict

    input_image = values_dict['src_file']
    if input_image is None:
        return "No input image provided."

    image_files = [input_image]

    # Make a dictionary to store the image data and path
    img_data = []

    if is_video(input_image):
        status_container.is_video = True
        extracted_folder = os.path.join(args.outputs_folder, "extracted_frames")
        if os.path.exists(extracted_folder):
            shutil.rmtree(extracted_folder)
        os.makedirs(extracted_folder, exist_ok=True)
        extract_success, video_params = extract_video(input_image, extracted_folder)
        if extract_success:
            status_container.video_params = video_params
        for file in os.listdir(extracted_folder):
            full_path = os.path.join(extracted_folder, file)
            media_data = MediaData(media_path=full_path)
            media_data.caption = main_prompt
            img_data.append(media_data)
    else:
        for file in image_files:
            try:
                media_data = MediaData(media_path=file)
                img = Image.open(file)
                media_data.media_data = np.array(img)
                media_data.caption = main_prompt
                img_data.append(media_data)
            except:
                pass
    result = "An exception occurred. Please try again."
    # auto_deload_llava, batch_process_folder, main_prompt, output_video_format, output_video_quality, outputs_folder,video_duration, video_fps, video_height, video_width
    keys_to_pop = ['auto_deload_llava', 'batch_process_folder', 'main_prompt', 'output_video_format',
                   'output_video_quality', 'outputs_folder', 'video_duration', 'video_fps', 'video_height',
                   'video_width', 'src_file']

    values_dict['outputs_folder'] = args.outputs_folder
    status_container.process_params = values_dict
    values_dict = {k: v for k, v in values_dict.items() if k not in keys_to_pop}

    try:
        _, result = batch_process(img_data, **values_dict)
    except Exception as e:
        print(f"An exception occurred: {e} at {traceback.format_exc()}")
        is_processing = False
    return result


def start_batch_process(element_values: List[Any]):
    global status_container, is_processing
    status_container = StatusContainer()
    status_container.is_batch = True
    values_dict = dict(zip(elements_dict.keys(), element_values))
    batch_process_folder = values_dict['batch_process_folder']
    outputs_folder = values_dict['outputs_folder']
    values_dict = {k: v for k, v in values_dict if k not in ['src_file']}
    main_prompt = values_dict['main_prompt']
    if not batch_process_folder:
        return "No input folder provided."
    if not os.path.exists(batch_process_folder):
        return "The input folder does not exist."

    if len(outputs_folder) < 2:
        outputs_folder = args.outputs_folder

    image_files = [file for file in os.listdir(batch_process_folder) if is_image(file)]

    # Make a dictionary to store the image data and path
    img_data = []
    for file in image_files:
        media_data = MediaData(media_path=os.path.join(batch_process_folder, file))
        img = Image.open(os.path.join(batch_process_folder, file))
        media_data.media_data = np.array(img)
        media_data.caption = main_prompt
        img_data.append(media_data)

    # Store it globally
    status_container.image_data = img_data
    result = "An exception occurred. Please try again."
    try:
        keys_to_pop = ['auto_deload_llava', 'batch_process_folder', 'main_prompt', 'output_video_format',
                       'output_video_quality', 'outputs_folder', 'video_duration', 'video_fps', 'video_height',
                       'video_width', 'src_file']

        status_container.outputs_folder = outputs_folder
        status_container.process_params = values_dict

        values_dict = {k: v for k, v in values_dict if k not in keys_to_pop}
        result, _ = batch_process(img_data, **values_dict)
    except Exception as e:
        print(f"An exception occurred: {e} at {traceback.format_exc()}")
        is_processing = False
    return result


def llava_process(inputs: List[MediaData], temp, p, question=None, save_captions=False, progress=gr.Progress()):
    global llava_agent, status_container
    outputs = []
    total_steps = len(inputs) + 1
    step = 0
    progress(step / total_steps, desc="Loading LLaVA...")
    load_llava()
    step += 1
    printt("Moving LLaVA to GPU.")
    llava_agent = to_gpu(llava_agent, LLaVA_device)
    printt("LLaVA moved to GPU.")
    progress(step / total_steps, desc="LLaVA loaded, captioning images...")
    for md in inputs:
        img = md.media_data
        img_path = md.media_path
        progress(step / total_steps, desc=f"Processing image {step}/{len(inputs)} with LLaVA...")
        lq = HWC3(img)
        lq = Image.fromarray(lq.astype('uint8'))
        caption = llava_agent.gen_image_caption([lq], temperature=temp, top_p=p, qs=question)
        caption = caption[0]
        md.caption = caption
        outputs.append(md)
        if save_captions:
            cap_path = os.path.splitext(img_path)[0] + ".txt"
            with open(cap_path, 'w') as cf:
                cf.write(caption)
        if not is_processing:  # Check if batch processing has been stopped
            break
        step += 1

    progress(step / total_steps, desc="LLaVA processing completed.")
    status_container.image_data = outputs
    return f"LLaVA Processing Completed: {len(inputs)} images processed at {time.ctime()}."


def supir_process(inputs: List[MediaData], a_prompt, n_prompt, num_samples,
                  upscale, edm_steps,
                  s_stage1, s_stage2, s_cfg, seed, sampler, s_churn, s_noise, color_fix_type, diff_dtype, ae_dtype,
                  linear_cfg, linear_s_stage2, spt_linear_cfg, spt_linear_s_stage2, model_select,
                  ckpt_select, num_images, random_seed, apply_llava, face_resolution, apply_bg, apply_face,
                  face_prompt, dont_update_progress=False, unload=True,
                  progress=gr.Progress()):
    global model, status_container, event_id
    main_begin_time = time.time()
    total_images = len(inputs) * num_images
    total_progress = total_images + 1
    if unload:
        total_progress += 1
    counter = 0
    progress(counter / total_progress, desc="Loading SUPIR Model...")
    load_model(model_select, ckpt_select, sampler, progress=progress)
    to_gpu(model, SUPIR_device)

    counter += 1
    progress(counter / total_progress, desc="Model Loaded, Processing Images...")
    model.ae_dtype = convert_dtype(ae_dtype)
    model.model.dtype = convert_dtype(diff_dtype)

    idx = 0
    output_data = []
    processed_images = 0
    params = status_container.process_params

    for image_data in inputs:
        gen_params_list = []
        img_params = params.copy()
        img = image_data.media_data
        image_path = image_data.media_path
        progress(counter / total_progress, desc=f"Processing image {counter}/{total_images}...")
        if img is None:
            printt(f"Image {counter}/{total_images} is None, loading from disk.")
            with Image.open(image_path) as img:
                img = np.array(img)

        printt(f"Processing image {counter}/{total_images}...")

        # Prompt is stored directly in the image data
        img_prompt = image_data.caption
        idx = idx + 1

        # See if there is a caption file
        if not apply_llava:
            cap_path = os.path.join(os.path.splitext(image_path)[0] + ".txt")
            if os.path.exists(cap_path):
                printt(f"Loading caption from {cap_path}...")
                with open(cap_path, 'r') as cf:
                    img_prompt = cf.read()

        img = HWC3(img)
        printt("Upscaling image (pre)...")
        img = upscale_image(img, upscale, unit_resolution=32, min_size=1024)

        lq = np.array(img)
        lq = lq / 255 * 2 - 1
        lq = torch.tensor(lq, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(SUPIR_device)[:, :3, :, :]

        _faces = []
        if not dont_update_progress and progress is not None:
            progress(counter / total_images, desc=f"Upscaling Images {counter}/{total_images}")
        face_captions = [img_prompt]

        if apply_face:
            lq = np.array(img)
            load_face_helper()
            if face_helper is None or not isinstance(face_helper, FaceRestoreHelper):
                raise ValueError('Face helper not loaded')
            face_helper.clean_all()
            face_helper.read_image(lq)
            # get face landmarks for each face
            face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
            face_helper.align_warp_face()

            lq = lq / 255 * 2 - 1
            lq = torch.tensor(lq, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(SUPIR_device)[:, :3, :, :]

            if len(face_prompt) > 1:
                face_captions = [face_prompt]
            to_gpu(face_helper, SUPIR_device)

        results = []

        for _ in range(num_images):
            gen_params = img_params.copy()
            gen_params['evt_id'] = event_id
            result = None
            if random_seed or num_images > 1:
                seed = np.random.randint(0, 2147483647)
            gen_params['seed'] = seed
            start_time = time.time()  # Track the start time

            def process_sample(model, input_data, caption, face_resolution=None, is_face=False):
                samples = model.batchify_sample(input_data, caption, num_steps=edm_steps, restoration_scale=s_stage1,
                                                s_churn=s_churn, s_noise=s_noise, cfg_scale=s_cfg,
                                                control_scale=s_stage2, seed=seed,
                                                num_samples=num_samples, p_p=a_prompt, n_p=n_prompt,
                                                color_fix_type=color_fix_type,
                                                use_linear_cfg=linear_cfg, use_linear_control_scale=linear_s_stage2,
                                                cfg_scale_start=spt_linear_cfg, control_scale_start=spt_linear_s_stage2)

                if is_face and face_resolution < 1024:
                    samples = samples[:, :, 512 - face_resolution // 2:512 + face_resolution // 2,
                              512 - face_resolution // 2:512 + face_resolution // 2]
                return samples

            if apply_face:
                faces = []
                for face in face_helper.cropped_faces:
                    face = np.array(face) / 255 * 2 - 1
                    face = torch.tensor(face, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(SUPIR_device)[:, :3,
                           :, :]
                    faces.append(face)

                for index, face in enumerate(faces):
                    progress(index / len(faces), desc=f"Upscaling Face {index}/{len(faces)}")
                    caption = face_captions[index]
                    samples = process_sample(model, face, [caption], face_resolution, is_face=True)
                    samples = torch.nn.functional.interpolate(samples, size=face_helper.face_size, mode='bilinear',
                                                              align_corners=False)
                    x_samples = (einops.rearrange(samples,
                                                  'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().round().clip(0,
                                                                                                                    255).astype(
                        np.uint8)
                    face_helper.add_restored_face(x_samples[0])
                gen_params["face_gallery"] = faces

            # Look ma, we can do either now!
            if apply_bg:
                caption = [img_prompt]
                samples = process_sample(model, lq, caption)
                _bg = (einops.rearrange(samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().round().clip(0,
                                                                                                                   255).astype(
                    np.uint8)
                if apply_face:
                    face_helper.get_inverse_affine(None)
                    result = face_helper.paste_faces_to_input_image(upsample_img=_bg[0])
                else:
                    result = _bg[0]

            if not apply_face and not apply_bg:
                caption = [img_prompt]
                print("Batchifying sample...")
                samples = process_sample(model, lq, caption)
                x_samples = (
                        einops.rearrange(samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().round().clip(
                    0, 255).astype(np.uint8)
                result = x_samples[0]

            image_generation_time = time.time() - start_time
            if result is not None:
                results.append(result)
                gen_params_list.append(gen_params)
            desc = f"Image {counter}/{total_images} upscale completed in {image_generation_time:.2f} seconds"
            counter += 1
            progress(counter / total_images, desc=desc)

        # Update outputs
        image_data.outputs = results
        image_data.metadata_list = gen_params_list
        # save_image will return the image data with the outputs as paths
        image_data = save_image(image_data, status_container.is_video)

        # Append the image data to the output data after saving
        output_data.append(image_data)

        progress(counter / total_images, desc=f"Image {counter}/{total_images} processed.")
        processed_images = counter
        # Now break
        if not is_processing:
            break
        counter += 1
    # Now we update the status container
    status_container.image_data = output_data
    if not is_processing or unload:
        progress(counter / total_images, desc="Unloading SUPIR...")
        all_to_cpu()
        progress(counter / total_images, desc="SUPIR Unloaded.")
    main_end_time = time.time()
    global unique_counter
    unique_counter = unique_counter + 1
    return f"Image Upscaling Completed: processed {processed_images} images at in {main_end_time - main_begin_time:.2f} seconds #{unique_counter}"


def batch_process(img_data,
                  a_prompt, ae_dtype, apply_bg, apply_face, apply_llava, apply_supir, ckpt_select, color_fix_type,
                  diff_dtype, edm_steps, face_prompt, face_resolution, linear_CFG, linear_s_stage2,
                  make_comparison_video, model_select, n_prompt, num_images, num_samples, qs, random_seed,
                  s_cfg, s_churn, s_noise, s_stage1, s_stage2, sampler, save_captions, seed, spt_linear_CFG,
                  spt_linear_s_stage2, temperature, top_p, upscale, progress=gr.Progress()
                  ):
    global is_processing, llava_agent, model, status_container
    ckpt_select = get_ckpt_path(ckpt_select)
    tiled = "TiledRestore" if args.use_tile_vae else "Restore"
    sampler_cls = f"sgm.modules.diffusionmodules.sampling.{tiled}{sampler}Sampler"
    if not ckpt_select:
        msg = "No checkpoint selected. Please select a checkpoint to continue."
        return msg
    start_time = time.time()
    last_result = "Select something to do."
    if not apply_llava and not apply_supir:
        msg = "No processing selected. Please select LLaVA, SUPIR, or both to continue."
        printt(msg)
        return msg, msg
    if is_processing:
        msg = "Batch processing already in progress."
        printt(msg)
        return msg, msg
    if len(img_data) == 0:
        msg = "No images to process."
        printt(msg)
        return msg, msg

    params = status_container.process_params
    is_processing = True
    # Get the list of image files in the folder
    total_images = len(img_data)

    # Total images, times number of images, plus 2 for load/unload
    total_supir_steps = total_images * num_images + 2 if apply_supir else 0

    # Total images, plus 2 for load/unload
    total_llava_steps = total_images + 2 if apply_llava else 0

    # Total steps, plus 1 for saving outputs
    total_steps = total_supir_steps + total_llava_steps + 1
    counter = 0
    # Disable llava for video...because...uh...yeah, video.
    if status_container.is_video:
        apply_llava = False
    progress(0, desc=f"Processing {total_images} images...")
    printt(f"Processing {total_images} images...", reset=True)
    if apply_llava:
        printt('Processing LLaVA')
        last_result = llava_process(img_data, temperature, top_p, qs, save_captions, progress=progress)
        printt('LLaVA processing completed')
        # Update the img_data from the captioner
        img_data = status_container.image_data
    # Check for cancellation
    if not is_processing and model is not None:
        progress(total_steps / total_steps, desc="Cancelling SUPIR...")
        all_to_cpu()
        return f"Batch Processing Completed: Cancelled at {time.ctime()}.", last_result
    counter += total_llava_steps
    if apply_supir:
        progress(counter / total_steps, desc="Processing images...")
        printt("Processing images (Stage 2)")
        last_result = supir_process(img_data, a_prompt, n_prompt, num_samples, upscale, edm_steps, s_stage1, s_stage2,
                                    s_cfg, seed, sampler_cls, s_churn, s_noise, color_fix_type, diff_dtype, ae_dtype,
                                    linear_CFG, linear_s_stage2, spt_linear_CFG, spt_linear_s_stage2, model_select,
                                    ckpt_select,
                                    num_images, random_seed, apply_llava, face_resolution, apply_bg, apply_face,
                                    face_prompt, unload=True, progress=progress)
        printt("Processing images (Stage 2) completed")
    counter += total_supir_steps
    progress(counter / total_steps, desc="Processing completed.")

    if status_container.is_video and apply_supir:
        printt("Processing outputs...")
        progress(counter / total_steps, desc="Compiling video...")
        extracted_folder = os.path.join(args.outputs_folder, "extracted_frames")
        output_quality = params.get('output_quality', 'medium')
        output_format = params.get('output_format', 'mp4')
        media_output = compile_video(extracted_folder, args.outputs_folder, status_container.video_params,
                                     output_quality,
                                     output_format)
        if media_output:
            status_container.image_data = [media_output]
            printt("Video compiled successfully.")
        else:
            printt("Video compilation failed.")
    elif make_comparison_video:
        updates = []
        progress(counter / total_steps, desc="Creating comparison videos...")
        for image_data in status_container.image_data:
            output = save_compare_video(image_data, params)
            updates.append(output)
        status_container.image_data = updates
    progress(1, desc="Processing Completed in " + str(time.time() - start_time) + " seconds.")

    is_processing = False
    end_time = time.time()
    global unique_counter
    unique_counter = unique_counter + 1
    return f"Batch Processing Completed: processed {total_images * num_images} images at in {end_time - start_time:.2f} seconds #{unique_counter}", last_result


def save_image(image_data: MediaData, is_video_frame: bool):
    global status_container
    if len(image_data.metadata_list) >= 1:
        params = image_data.metadata_list[0]
    else:
        params = status_container.process_params
    save_caption = params.get('save_captions', False)
    output_dir = params.get('outputs_folder', args.outputs_folder)
    metadata_dir = os.path.join(output_dir, "images_meta_data")
    if not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir)
    results = image_data.outputs
    image_path = image_data.media_path
    result_paths = []
    for i, result in enumerate(results):
        event_dict = params.copy()

        if len(image_data.metadata_list) > i:
            event_dict = image_data.metadata_list[i]

        evt_id = event_dict.get('evt_id', str(time.time_ns()))

        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        if len(base_filename) > 250:
            base_filename = base_filename[:250]

        img = Image.fromarray(result)

        if args.log_history and not is_video_frame:
            history_path = f'./history/{evt_id[:5]}/{evt_id[5:]}'
            os.makedirs(history_path, exist_ok=True)
            with open(os.path.join(history_path, 'logs.txt'), 'w') as f:
                f.write(str(event_dict))
            img.save(os.path.join(history_path, f'HQ_{i}.png'))

        save_path = os.path.join(output_dir, f'{base_filename}.png')

        # If processing video, just save the image
        if is_video_frame:
            save_path = image_path
            img.save(save_path, "PNG")
        else:
            index = 1
            while os.path.exists(save_path):
                save_path = os.path.join(output_dir, f'{base_filename}_{str(index).zfill(4)}.png')
                index += 1

            meta = PngImagePlugin.PngInfo()
            for key, value in event_dict.items():
                try:
                    meta.add_text(key, str(value))
                except:
                    pass
            caption = image_data.caption.strip()
            if caption:
                meta.add_text('caption', caption)
                # This will save the caption to a file with the same name as the image
                if save_caption:
                    caption_path = f'{os.path.splitext(save_path)[0]}.txt'
                    with open(caption_path, 'w') as f:
                        f.write(caption)
            img.save(save_path, "PNG", pnginfo=meta)

            metadata_path = os.path.join(metadata_dir, f'{os.path.splitext(os.path.basename(save_path))[0]}.txt')
            with open(metadata_path, 'w') as f:
                for key, value in event_dict.items():
                    try:
                        f.write(f'{key}: {value}\n')
                    except:
                        pass
        result_paths.append(save_path)
    image_data.outputs = result_paths
    return image_data


def save_compare_video(image_data: MediaData, params):
    image_path = image_data.media_path
    output_dir = params.get('outputs_folder', args.outputs_folder)

    compare_videos_dir = os.path.join(output_dir, "compare_videos")
    if not os.path.exists(compare_videos_dir):
        os.makedirs(compare_videos_dir)

    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(output_dir, f'{base_filename}.mp4')
    index = 1
    while os.path.exists(save_path):
        save_path = os.path.join(output_dir, f'{base_filename}_{str(index).zfill(4)}.mp4')
        index += 1

    video_path = os.path.join(compare_videos_dir, f'{base_filename}.mp4')
    full_save_image_path = os.path.abspath(save_path)
    org_image_absolute_path = os.path.abspath(image_path)
    create_comparison_video(org_image_absolute_path, full_save_image_path, video_path, params)
    image_data.comparison_video = video_path
    return image_data


def stop_batch_upscale(progress=gr.Progress()):
    global is_processing
    progress(1, f"Stop command giving please wait to stop")
    print('\n***Stop command giving please wait to stop***\n')
    is_processing = False


def load_and_reset(param_setting):
    e_steps = 50
    sstage2 = 1.0
    sstage1 = -1.0
    schurn = 5
    snoise = 1.003
    ap = 'Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo - ' \
         'realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, skin pore ' \
         'detailing, hyper sharpness, perfect without deformations.'
    np = 'painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, ' \
         '3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, ' \
         'signature, jpeg artifacts, deformed, lowres, over-smooth'
    cfix_type = 'Wavelet'
    l_s_stage2 = 0.0
    l_s_s_stage2 = False
    l_cfg = True
    if param_setting == "Quality":
        s_cfg = 7.5
        spt_linear_CFG = 4.0
    elif param_setting == "Fidelity":
        s_cfg = 4.0
        spt_linear_CFG = 1.0
    else:
        raise NotImplementedError
    return e_steps, s_cfg, sstage2, sstage1, schurn, snoise, ap, np, cfix_type, l_cfg, l_s_stage2, spt_linear_CFG, l_s_s_stage2


def submit_feedback(evt_id, f_score, f_text):
    if args.log_history:
        with open(f'./history/{evt_id[:5]}/{evt_id[5:]}/logs.txt', 'r') as f:
            event_dict = eval(f.read())
        f.close()
        event_dict['feedback'] = {'score': f_score, 'text': f_text}
        with open(f'./history/{evt_id[:5]}/{evt_id[5:]}/logs.txt', 'w') as f:
            f.write(str(event_dict))
        f.close()
        return 'Submit successfully, thank you for your comments!'
    else:
        return 'Submit failed, the server is not set to log history.'


preview_full = False


def toggle_full_preview():
    global preview_full
    gal_classes = ["preview_col"]
    btn_classes = ["slider_button"]

    if preview_full:
        preview_full = False
    else:
        preview_full = True
        gal_classes.append("full_preview")
        btn_classes.append("full")
    return gr.update(elem_classes=gal_classes), gr.update(elem_classes=btn_classes), gr.update(elem_classes=btn_classes)


def toggle_compare_elements(enable: bool) -> Tuple[gr.update, gr.update]:
    return gr.update(visible=enable), gr.update(visible=enable), gr.update(visible=enable)


title_md = """
# **SUPIR: Practicing Model Scaling for Photo-Realistic Image Restoration**

1 Click Installer (auto download models as well) : https://www.patreon.com/posts/99176057

FFmpeg Install Tutorial : https://youtu.be/-NjNy7afOQ0 &emsp; [[Paper](https://arxiv.org/abs/2401.13627)] &emsp; [[Project Page](http://supir.xpixel.group/)] &emsp; [[How to play](https://github.com/Fanghua-Yu/SUPIR/blob/master/assets/DemoGuide.png)]
"""

claim_md = """
## **Terms of use**

By using this service, users are required to agree to the following terms: The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research. Please submit a feedback to us if you get any inappropriate answer! We will collect those to keep improving our models. For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.

## **License**
While the original readme for the project *says* it's non-commercial, it was *actually* released under the MIT license. That means that the project can be used for whatever you want.

And yes, it would certainly be nice if anything anybody stuck in a random readme were the ultimate gospel when it comes to licensing, unfortunately, that's just
not how the world works. MIT license means FREE FOR ANY PURPOSE, PERIOD.
The service is a research preview ~~intended for non-commercial use only~~, subject to the model [License](https://github.com/Fanghua-Yu/SUPIR#MIT-1-ov-file) of SUPIR.
"""

css_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'css', 'style.css'))

js_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'javascript', 'demo.js'))
with open(js_file) as f:
    js = f.read()

head = f"""
<script type="text/javascript">{js}</script>
"""

refresh_symbol = "\U000027F3"  # ⟳
dl_symbol = "\U00002B73"  # ⭳
fullscreen_symbol = "\U000026F6"  # ⛶


def show_output(selected_file):
    if is_video(selected_file):
        return gr.update(visible=False), gr.update(visible=True, value=selected_file)
    elif is_image(selected_file):
        return gr.update(visible=True, value=selected_file), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=False)

default_llava_prompt = "Describe this image and its style in a very detailed manner. The image is a realistic photography, not an art painting."
prompt_styles = list_styles()
# Make a list of prompt_styles keys
prompt_styles_keys = list(prompt_styles.keys())

selected_pos, selected_neg, llava_style_prompt = select_style(
                            prompt_styles_keys[0] if len(prompt_styles_keys) > 0 else "", default_llava_prompt,True)

block = gr.Blocks(title='SUPIR', theme=args.theme, css=css_file, head=head).queue()

with block:
    with gr.Tab("Upscale"):
        # Execution buttons
        with gr.Column(scale=1):
            with gr.Row():
                start_single_button = gr.Button(value="Process Single")
                start_batch_button = gr.Button(value="Process Batch")
                stop_batch_button = gr.Button(value="Cancel")
        with gr.Column(scale=1):
            with gr.Row():
                output_label = gr.Label(label="Progress", elem_classes=["progress_label"])
        with gr.Row(equal_height=True):
            with gr.Column(elem_classes=['preview_col']) as input_col:
                src_input_file = gr.File(type="filepath", label="Input", elem_id="file-input",
                                         elem_classes=["preview_box"], height=400, visible=True, file_count="single",
                                         file_types=["image", "video"])
                src_image_display = gr.Image(type="filepath", elem_id="image-input", label="Input Image",
                                             elem_classes=["preview_box"], height=400, sources=None,
                                             visible=False)
                src_video_display = gr.Video(elem_id="video-input", label="Input Video", elem_classes=["preview_box"],
                                             height=400, sources=None, visible=False)
            with gr.Column(visible=False, elem_classes=['preview_col']) as comparison_video_col:
                comparison_video = gr.Video(label="Comparison Video", elem_classes=["preview_box"], height=400,
                                            visible=False)
            with gr.Column(elem_classes=['preview_col'], elem_id="preview_column") as result_col:
                result_gallery = gr.Gallery(label='Output', elem_id="gallery2", elem_classes=["preview_box"],
                                            height=400, visible=False, rows=2, columns=4, allow_preview=True,
                                            show_download_button=False, show_share_button=False)
                result_slider = ImageSlider(label='Output', interactive=False, show_download_button=True,
                                            elem_id="gallery1",
                                            elem_classes=["preview_box", "preview_slider", "active"],
                                            height=400, container=True)
                result_video = gr.Video(label="Output Video", elem_classes=["preview_box"], height=400, visible=False)
                slider_dl_button = gr.Button(value=dl_symbol, elem_classes=["slider_button"], visible=True,
                                             elem_id="download_button")
                slider_full_button = gr.Button(value=fullscreen_symbol, elem_classes=["slider_button"], visible=True,
                                               elem_id="fullscreen_button")
        with gr.Row():
            with gr.Column():
                with gr.Accordion("General options", open=True):
                    target_res_textbox = gr.Textbox(label="Input / Output Resolution", value="", interactive=False)
                    if args.debug:
                        populate_slider_button = gr.Button(value="Populate Slider")
                        populate_gallery_button = gr.Button(value="Populate Gallery")
                        populate_slider_button.click(fn=populate_slider_single, outputs=[result_slider, result_gallery],
                                                     show_progress=True, queue=True)
                        populate_gallery_button.click(fn=populate_gallery, outputs=[result_gallery, result_slider],
                                                      show_progress=True, queue=True)
                    with gr.Row():
                        apply_llava_checkbox = gr.Checkbox(label="Apply LLaVa", value=False)
                        apply_supir_checkbox = gr.Checkbox(label="Apply SUPIR", value=True)
                    show_select = args.ckpt_browser
                    with gr.Row(elem_id="model_select_row", visible=show_select):
                        ckpt_select_dropdown = gr.Dropdown(label="Model", choices=list_models(),
                                                           value=selected_model(),
                                                           interactive=True)
                        refresh_models_button = gr.Button(value=refresh_symbol, elem_classes=["refresh_button"],
                                                          size="sm")
                    with gr.Row(visible=show_select):
                        ckpt_type = gr.Dropdown(label="Checkpoint Type", choices=["Standard SDXL", "SDXL Lightning"],
                                                value="Standard SDXL")
                    with gr.Row(elem_id="style_select_row"):
                        prompt_style_dropdown = gr.Dropdown(label="Default Prompt Style",
                                                            choices=prompt_styles_keys,
                                                            value=prompt_styles_keys[0] if len(
                                                                prompt_styles_keys) > 0 else "")
                        refresh_styles_button = gr.Button(value=refresh_symbol, elem_classes=["refresh_button"],
                                                          size="sm")

                    upscale_slider = gr.Slider(label="Upscale Size", minimum=1, maximum=8, value=1, step=0.1)
                    prompt_textbox = gr.Textbox(label="Prompt", value="")
                    face_prompt_textbox = gr.Textbox(label="Face Prompt",
                                                     placeholder="Optional, uses main prompt if not provided",
                                                     value="")
                with gr.Accordion("LLaVA options", open=False):
                    with gr.Column():
                        auto_unload_llava = gr.Checkbox(label="Auto Unload LLaVA (Low VRAM)", value=auto_unload)
                        setattr(auto_unload_llava, "do_not_save_to_config", True)

                    temperature_slider = gr.Slider(label="Temperature", minimum=0., maximum=1.0, value=0.2, step=0.1)
                    top_p_slider = gr.Slider(label="Top P", minimum=0., maximum=1.0, value=0.7, step=0.1)
                    qs_textbox = gr.Textbox(label="Question",
                                            value=llava_style_prompt)
                with gr.Accordion("SUPIR options", open=False):
                    with gr.Row():
                        with gr.Column():
                            num_images_slider = gr.Slider(label="Number Of Images To Generate", minimum=1, maximum=200
                                                          , value=1, step=1)
                            num_samples_slider = gr.Slider(label="Batch Size", minimum=1,
                                                           maximum=4, value=1, step=1)
                        with gr.Column():
                            random_seed_checkbox = gr.Checkbox(label="Randomize Seed", value=True)
                    with gr.Row():
                        edm_steps_slider = gr.Slider(label="Steps", minimum=1, maximum=200, value=50, step=1)
                        s_cfg_slider = gr.Slider(label="Text Guidance Scale", minimum=1.0, maximum=15.0, value=7.5,
                                                 step=0.1)
                        s_stage2_slider = gr.Slider(label="Stage2 Guidance Strength", minimum=0., maximum=1., value=1.,
                                                    step=0.05)
                        s_stage1_slider = gr.Slider(label="Stage1 Guidance Strength", minimum=-1.0, maximum=6.0,
                                                    value=-1.0,
                                                    step=1.0)
                        seed_slider = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                        sampler_dropdown = gr.Dropdown(label="Sampler", choices=["EDM", "DPMPP2M"],
                                                       value="DPMPP2M")
                        s_churn_slider = gr.Slider(label="S-Churn", minimum=0, maximum=40, value=5, step=1)
                        s_noise_slider = gr.Slider(label="S-Noise", minimum=1.0, maximum=1.1, value=1.003, step=0.001)

                    with gr.Row():
                        a_prompt_textbox = gr.Textbox(label="Default Positive Prompt",
                                                      value=selected_pos)
                        n_prompt_textbox = gr.Textbox(label="Default Negative Prompt",
                                                      value=selected_neg)
                with gr.Accordion("Video options", open=False):
                    with gr.Column():
                        output_vq_dropdown = gr.Dropdown(label="Video Quality", choices=["Low", "Medium", "High"],
                                                         value="High")
                        output_vf_dropdown = gr.Dropdown(label="Video Format", choices=["mp4", "mkv"], value="mp4")

            with gr.Column():
                with gr.Accordion("Batch options", open=True):
                    with gr.Row():
                        with gr.Column():
                            batch_process_folder_textbox = gr.Textbox(
                                label="Batch Input Folder - Can use image captions from .txt",
                                placeholder="R:\SUPIR video\comparison_images")
                            outputs_folder_textbox = gr.Textbox(
                                label="Batch Output Path - Leave empty to save to default.",
                                placeholder="R:\SUPIR video\comparison_images\outputs")
                            save_captions_checkbox = gr.Checkbox(label="Save Captions", value=True)
                with gr.Accordion("Advanced options", open=False):
                    with gr.Row():
                        with gr.Column():
                            param_setting_select = gr.Dropdown(["Quality", "Fidelity"], interactive=True,
                                                               label="Param Setting",
                                                               value="Quality")
                        with gr.Column():
                            reset_button = gr.Button(value="Reset Param", scale=2)
                    with gr.Row():
                        with gr.Column():
                            linear_cfg_checkbox = gr.Checkbox(label="Linear CFG", value=True)
                            spt_linear_cfg_slider = gr.Slider(label="CFG Start", minimum=1.0,
                                                              maximum=9.0, value=4.0, step=0.5)
                        with gr.Column():
                            linear_s_stage2_checkbox = gr.Checkbox(label="Linear Stage2 Guidance", value=False)
                            spt_linear_s_stage2_slider = gr.Slider(label="Guidance Start", minimum=0,
                                                                   maximum=1, value=0, step=0.05)
                    with gr.Row():
                        with gr.Column():
                            diff_dtype_radio = gr.Radio(['fp32', 'fp16', 'bf16'], label="Diffusion Data Type",
                                                        value="bf16",
                                                        interactive=True)
                        with gr.Column():
                            ae_dtype_radio = gr.Radio(['fp32', 'bf16'], label="Auto-Encoder Data Type", value="bf16",
                                                      interactive=True)
                        with gr.Column():
                            color_fix_type_radio = gr.Radio(["None", "AdaIn", "Wavelet"], label="Color-Fix Type",
                                                            value="Wavelet",
                                                            interactive=True)
                        with gr.Column():
                            model_select_radio = gr.Radio(["v0-Q", "v0-F"], label="Model Selection", value="v0-Q",
                                                          interactive=True)
                with gr.Accordion("Face options", open=False):
                    face_resolution_slider = gr.Slider(label="Text Guidance Scale", minimum=256, maximum=2048,
                                                       value=1024,
                                                       step=32)
                    with gr.Row():
                        with gr.Column():
                            apply_bg_checkbox = gr.Checkbox(label="BG restoration", value=False)
                        with gr.Column():
                            apply_face_checkbox = gr.Checkbox(label="Face restoration", value=False)
                with gr.Accordion("Comparison Video options", open=False):
                    with gr.Row():
                        make_comparison_video_checkbox = gr.Checkbox(
                            label="Generate Comparison Video (Input vs Output) (You need to have FFmpeg installed)",
                            value=False)
                    with gr.Row(visible=False) as compare_video_row:
                        video_duration_textbox = gr.Textbox(label="Duration", value="5")
                        video_fps_textbox = gr.Textbox(label="FPS", value="30")
                        video_width_textbox = gr.Textbox(label="Width", value="1920")
                        video_height_textbox = gr.Textbox(label="Height", value="1080")
    with gr.Tab("Image Metadata"):
        with gr.Row():
            metadata_image_input = gr.Image(type="filepath", label="Upload Image")
            metadata_output = gr.Textbox(label="Image Metadata", lines=25, max_lines=50)
        metadata_image_input.change(fn=read_image_metadata, inputs=[metadata_image_input], outputs=[metadata_output])
    with gr.Tab("Restored Faces"):
        with gr.Row():
            face_gallery = gr.Gallery(label='Faces', show_label=False, elem_id="gallery2")
    with gr.Tab("Outputs", elem_id="output_tab"):
        with gr.Row():
            with gr.Column():
                output_files = gr.FileExplorer(label="Output Folder", file_count="single", elem_id="output_folder",
                                               root_dir=args.outputs_folder)
            with gr.Column():
                output_image = gr.Image(type="filepath", label="Output Image", elem_id="output_image", visible=False)
                output_video = gr.Video(label="Output Video", elem_id="output_video", visible=False)

    with gr.Tab("About"):
        gr.HTML(f"<H2>About {SUPIR_REVISION}</H2>")
        gr.Markdown(title_md)
        with gr.Row():
            gr.Markdown(claim_md)
            event_id = gr.Textbox(label="Event ID", value="", visible=False)
        with gr.Accordion("Feedback", open=False):
            fb_score = gr.Slider(label="Feedback Score", minimum=1, maximum=5, value=3, step=1,
                                 interactive=True)
            fb_text = gr.Textbox(label="Feedback Text", value="",
                                 placeholder='Please enter your feedback here.')
            submit_button = gr.Button(value="Submit Feedback")

    #prompt_el, result_gallery_el, result_slider_el, result_video_el, comparison_video_el, event_id_el, seed_el, face_gallery_el
    output_elements = [
        prompt_textbox, result_gallery, result_slider, result_video, comparison_video, event_id, seed_slider, face_gallery
    ]

    refresh_models_button.click(fn=refresh_models_click, outputs=[ckpt_select_dropdown])
    refresh_styles_button.click(fn=refresh_styles_click, outputs=[prompt_style_dropdown])

    elements_dict = {
        "a_prompt": a_prompt_textbox,
        "ae_dtype": ae_dtype_radio,
        "apply_bg": apply_bg_checkbox,
        "apply_face": apply_face_checkbox,
        "apply_llava": apply_llava_checkbox,
        "apply_supir": apply_supir_checkbox,
        "auto_deload_llava": auto_unload_llava,
        "batch_process_folder": batch_process_folder_textbox,
        "ckpt_select": ckpt_select_dropdown,
        "color_fix_type": color_fix_type_radio,
        "diff_dtype": diff_dtype_radio,
        "edm_steps": edm_steps_slider,
        "face_prompt": face_prompt_textbox,
        "face_resolution": face_resolution_slider,
        "linear_CFG": linear_cfg_checkbox,
        "linear_s_stage2": linear_s_stage2_checkbox,
        "main_prompt": prompt_textbox,
        "make_comparison_video": make_comparison_video_checkbox,
        "model_select": model_select_radio,
        "n_prompt": n_prompt_textbox,
        "num_images": num_images_slider,
        "num_samples": num_samples_slider,
        "output_video_format": output_vf_dropdown,
        "output_video_quality": output_vq_dropdown,
        "outputs_folder": outputs_folder_textbox,
        "qs": qs_textbox,
        "random_seed": random_seed_checkbox,
        "s_cfg": s_cfg_slider,
        "s_churn": s_churn_slider,
        "s_noise": s_noise_slider,
        "s_stage1": s_stage1_slider,
        "s_stage2": s_stage2_slider,
        "sampler": sampler_dropdown,
        "save_captions": save_captions_checkbox,
        "seed": seed_slider,
        "spt_linear_CFG": spt_linear_cfg_slider,
        "spt_linear_s_stage2": spt_linear_s_stage2_slider,
        "src_file": src_input_file,
        "temperature": temperature_slider,
        "top_p": top_p_slider,
        "upscale": upscale_slider,
        "video_duration": video_duration_textbox,
        "video_fps": video_fps_textbox,
        "video_height": video_height_textbox,
        "video_width": video_width_textbox,
    }

    elements = list(elements_dict.values())

    start_single_button.click(fn=start_single_process, inputs=elements, outputs=output_label,
                              show_progress=True, queue=True)
    start_batch_button.click(fn=start_batch_process, inputs=elements, outputs=output_label,
                             show_progress=True, queue=True)
    stop_batch_button.click(fn=stop_batch_upscale, show_progress=True, queue=True)
    reset_button.click(fn=load_and_reset, inputs=[param_setting_select],
                       outputs=[edm_steps_slider, s_cfg_slider, s_stage2_slider, s_churn_slider,
                                s_noise_slider, a_prompt_textbox, n_prompt_textbox,
                                color_fix_type_radio, linear_cfg_checkbox, linear_s_stage2_checkbox,
                                spt_linear_cfg_slider, spt_linear_s_stage2_slider])

    # We just read the output_label and update all the elements when we find "Processing Complete"
    output_label.change(fn=update_elements, show_progress=False, queue=True, inputs=[output_label],
                        outputs=output_elements)

    output_files.change(fn=show_output, inputs=[output_files], outputs=[output_image, output_video])

    prompt_style_dropdown.change(fn=select_style, inputs=[prompt_style_dropdown, qs_textbox],
                                 outputs=[a_prompt_textbox, n_prompt_textbox, qs_textbox])

    make_comparison_video_checkbox.change(fn=toggle_compare_elements, inputs=[make_comparison_video_checkbox],
                                          outputs=[comparison_video_col, compare_video_row, comparison_video])
    submit_button.click(fn=submit_feedback, inputs=[event_id, fb_score, fb_text], outputs=[fb_text])
    upscale_slider.change(fn=update_target_resolution, inputs=[src_image_display, upscale_slider],
                          outputs=[target_res_textbox])

    # slider_dl_button.click(fn=download_slider_image, inputs=[result_slider], show_progress=False, queue=True)
    slider_full_button.click(fn=toggle_full_preview, outputs=[result_col, slider_full_button, slider_dl_button],
                             show_progress=False, queue=True, js="toggleFullscreen")

    src_input_file.change(fn=update_inputs, inputs=[src_input_file, upscale_slider],
                          outputs=[src_input_file, src_image_display, src_video_display, target_res_textbox])
    src_image_display.clear(fn=update_inputs, inputs=[src_image_display, upscale_slider],
                            outputs=[src_input_file, src_image_display, src_video_display, target_res_textbox])
    src_video_display.clear(fn=update_inputs, inputs=[src_video_display, upscale_slider],
                            outputs=[src_input_file, src_image_display, src_video_display, target_res_textbox])
    # s_cfg_Quality, spt_linear_CFG_Quality, s_cfg_Fidelity, spt_linear_CFG_Fidelity, edm_steps
    model_settings_elements = [s_cfg_slider, spt_linear_cfg_slider, edm_steps_slider]

    ckpt_type.change(fn=update_model_settings, inputs=[ckpt_type, param_setting_select],
                     outputs=model_settings_elements)


    def do_nothing():
        pass


    slider_dl_button.click(js="downloadImage", show_progress=False, queue=True, fn=do_nothing)

if args.port is not None:  # Check if the --port argument is provided
    block.launch(server_name=server_ip, server_port=args.port, share=args.share, inbrowser=args.open_browser)
else:
    block.launch(server_name=server_ip, share=args.share, inbrowser=args.open_browser)
