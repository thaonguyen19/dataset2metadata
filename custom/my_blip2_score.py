from lavis.models import load_model_and_preprocess
import torch
import torch.nn as nn
from torchvision import transforms
from dataset2metadata.postprocessors import select
from functools import partial


def blip_score_pre(x):
    return x


class Blip2ScoreWrapper(nn.Module):

    name = 'blip2'
    raw_inputs = ['image', 'text']
    preprocessors = ['blip2-score-pre', 'blip2-score-pre']
    #preprocessors = ['clip-aug', 'clip-tokens']
    dependencies = []
    to_device = True

    def __init__(self, device) -> None:
        super().__init__()
        self.model, self.vis_processors, self.text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=device, is_eval=True)
        self.device = device
        print(f'instantiated {self.name} on {device}')

    def forward(self, xs, ts):
        #transform = transforms.Compose([
        #    transforms.ToPILImage()
        #])
        itm_scores, itc_scores = [], []
        print(len(xs))
        #for x, t in zip(xs, ts):
            
        img = self.vis_processors["eval"](xs)#.to(self.device)#.unsqueeze(0).to(self.device)
        txt = self.text_processors["eval"](ts)
        
        itm_output = self.model({"image": img, "text_input": txt}, match_head="itm")
        itm_score = torch.nn.functional.softmax(itm_output, dim=1)[:, 1].item()

        itc_score = self.model({"image": img, "text_input": txt}, match_head='itc')

        itm_scores.append(itm_score)
        itc_scores.append(itc_score)

        return itm_scores, itc_scores


# define preprocessor map
preprocessor_lookup = {
    'blip2-score-pre': blip_score_pre,
}

# define model loopup
model_lookup = {
    'blip2_score_model': Blip2ScoreWrapper,
}

# postprocessors
postprocess_parquet_lookup = {
    'blip2-itm': partial(select, model='blip2_score_model', index=0, to_cpu=False),
    'blip2-itc': partial(select, model='blip2_score_model', index=1, to_cpu=False) 
}
postprocess_feature_lookup = {}
