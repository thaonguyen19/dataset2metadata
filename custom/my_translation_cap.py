# custom.py implementation
from functools import partial
import clip

import fasttext
import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from clip import clip
from dataset2metadata.postprocessors import identity, select, batched_dot_product_index
import dataset2metadata.preprocessors as pre

lang_detect_model = fasttext.load_model('/fsx/home-thaottn/fasttext_model.bin')
lang_code_to_id = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", use_auth_token=True).lang_code_to_id

def translate_pre(t):
    t = t.replace("\n", "")
    pred = lang_detect_model.predict(t)
    lang = pred[0][0].split("__label__")[1]
    #conf = pred[1][0]
    if lang not in lang_code_to_id:
        t = 'Not a valid language'
        lang = 'eng_Latn'
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", use_auth_token=True, src_lang=lang, use_fast=False)
    tokenized_t = tokenizer(t)
    return tokenized_t#, lang, conf


class TranslateClipL14Wrapper(nn.Module):

    name = 'translate-model'
    raw_inputs = ['image', 'text', 'text']
    preprocessors = ['clip-aug', 'identity', 'lang-tokens']
    dependencies = []
    to_device = False

    def __init__(self, device) -> None:
        super().__init__()
        self.lang_detect_model = lang_detect_model
        self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", use_auth_token=True).to(device)
        self.eng_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", use_auth_token=True, src_lang='eng_Latn', use_fast=False)

        #self.model.eval()
        #self.model.config.vision_config.min_length = 5
        #self.model.config.vision_config.max_length = 40
        
        #self.model.config.vision_config.do_sample = True
        #self.model.config.vision_config.top_k = 50
        #self.model.config.vision_config.temperature = 0.75

        self.l14, _ = clip.load('ViT-L/14', device=device)
        self.l14.eval()
        self.device = device
        print(f'instantiated {self.name} on {device}')

    def forward(self, x_clip, t_clip, t_tokenized):
        
        t_clip = [s.replace("\n", "") for s in t_clip]
        preds = self.lang_detect_model.predict(t_clip)
        langs = [pred[0].split('__label__')[1] for pred in preds[0]] 
        confs = [pred[0] for pred in preds[1]] 

        t_clip = clip.tokenize(t_clip, truncate=True).to(self.device)
        batch_encodings = self.eng_tokenizer.batch_encode_plus([t['input_ids'] for t in t_tokenized], padding=True,  return_tensors='pt',  is_split_into_words=True, add_special_tokens=False)
        translated_tokens = self.model.generate(**batch_encodings.to(self.device), forced_bos_token_id=self.eng_tokenizer.lang_code_to_id["eng_Latn"], max_length=100)
        translated_text = self.eng_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True) #TODO: check eng_tokenizer

        l14_image_feature = self.l14.encode_image(x_clip.to(self.device))
        l14_translated_text_feature = self.l14.encode_text(clip.tokenize(translated_text, truncate=True).to(self.device))
        l14_org_text_feature = self.l14.encode_text(t_clip.to(self.device))

        l14_image_feature = l14_image_feature / l14_image_feature.norm(dim=1, keepdim=True)
        l14_translated_text_feature = l14_translated_text_feature / l14_translated_text_feature.norm(dim=1, keepdim=True)
        l14_org_text_feature = l14_org_text_feature / l14_org_text_feature.norm(dim=1, keepdim=True)

        return l14_image_feature, l14_translated_text_feature, l14_org_text_feature, translated_text, langs, confs


# map preprocessor strings to preprocessor functions
preprocessor_lookup = {
    'lang-tokens': translate_pre,
}

# map model strings to model classes
model_lookup = {
    'translate-model': TranslateClipL14Wrapper,
}

# map postprocessor strings to functions, outputs saved to column-store parquet
postprocess_parquet_lookup = {
    'translated-cap': partial(select, model='translate-model', index=3, to_cpu=False),
    'lang': partial(select, model='translate-model', index=4, to_cpu=False),
    'conf': partial(select, model='translate-model', index=5, to_cpu=False),
    'oai-clip-l14-translated-score': partial(batched_dot_product_index, i=0, j=1, model='translate-model'),
    'oai-clip-l14-original-score': partial(batched_dot_product_index, i=0, j=2, model='translate-model')
}

postprocess_feature_lookup = {
}
