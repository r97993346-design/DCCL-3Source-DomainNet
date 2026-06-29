DEFAULT_DG_STYLE_PROMPTS = [
    "black-and-white sketch style",
    "line drawing style",
    "cartoon style",
    "clipart style",
    "flat illustration style",
    "watercolor painting style",
    "oil painting style",
    "artistic painting style",
    "low-texture style",
    "edge-dominant style",
]

RISE_STYLE_PROMPT_SUBSET = [
    "change the image appearance to a sketch drawing style",
    "convert the image to a cartoon illustration style",
    "render the image in a painting style",
    "make the image look like clipart",
    "change only the texture and color style",
]

_TEMPLATE = (
    "Change only the domain style of this image to {style}, while preserving "
    "the {class_name} identity, shape, pose, structure, and category. Do not add or remove objects."
)


def build_diffusion_edit_prompt(class_name, style):
    return _TEMPLATE.format(class_name=str(class_name).replace("_", " "), style=style)


def get_style_prompt_bank(mode="default_dg_style"):
    if mode == "rise_style_subset":
        return list(RISE_STYLE_PROMPT_SUBSET)
    if mode in ("default_dg_style", "default"):
        return list(DEFAULT_DG_STYLE_PROMPTS)
    raise ValueError("Unknown causal prompt bank: {}".format(mode))
