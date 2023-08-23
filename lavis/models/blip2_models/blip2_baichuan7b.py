"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_opt import OPTForCausalLM, OPTConfig
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import BertTokenizer

@registry.register_model("blip2_baichuan7b")
class Blip2AutoLLM(Blip2Base):
    """
    BLIP2 model.
    Supported model types:
    Usage:
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "baichuan7b": "configs/models/zdtc/blip2_baichuan7b.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        llm_model="",
        prompt="",
        max_txt_len=56,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        logging.info("load LLM model:{}".format(llm_model))
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model, padding_side='right', trust_remote_code=True)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model, torch_dtype=torch.float16, trust_remote_code=True)

        # from transformers import LlamaTokenizer
        # self.llm_tokenizer = LlamaTokenizer.from_pretrained(llm_model)
        # self.llm = AutoModelForCausalLM.from_pretrained(llm_model, torch_dtype=torch.float16)

        # self.eos_token_id = self.llm_tokenizer(
        #     "</s>", add_special_tokens=False
        # ).input_ids[0]
        # self.eos_token = "</s>"

        for param in self.llm.parameters():
            param.requires_grad = False

        self.eos_token_id = self.llm_tokenizer.eos_token_id
        self.eos_token = self.llm_tokenizer.eos_token

        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llm.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.prompt = prompt
        # prompt_tokens = self.llm_tokenizer(self.prompt, return_tensors="pt")
        # self.prompt_length = prompt_tokens.attention_mask.sum(1)

    def forward(self, samples):
        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_query = self.llm_proj(query_output.last_hidden_state)
        atts_query = torch.ones(inputs_query.size()[:-1], dtype=torch.long).to(image.device)

        self.llm_tokenizer.padding_side = "right"

        if self.prompt:
            prompts = [self.prompt.format(l) for l in samples['lang']]
            prompts_length = [self.llm_tokenizer(p, return_tensors="pt").attention_mask.sum(1) for p in prompts]
            text = samples["text_input"]
            for i in range(len(text)):
                text[i] = prompts[i] + text[i] + self.eos_token
            # text = [self.prompt+ t + self.eos_token for t in samples["text_input"]]
        else:
            text = [t + self.eos_token for t in samples["text_input"]]

        llm_tokens = self.llm_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)

        targets = llm_tokens.input_ids.masked_fill(
            llm_tokens.input_ids == self.llm_tokenizer.pad_token_id, -100
        )
        if self.prompt:
            for i, l in enumerate(prompts_length):
                targets[i][:l] = -100
            # targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt

        empty_targets = (
            torch.ones(atts_query.size(), dtype=torch.long).to(image.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        # inputs_embeds = self.llm.model.embed_tokens(llm_tokens.input_ids)
        inputs_embeds = self.llm.get_input_embeddings(llm_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_query, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_query, llm_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"loss": loss}
    
    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=None,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=1,
    ):
        
        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_query = self.llm_proj(query_output.last_hidden_state)
        atts_query = torch.ones(inputs_query.size()[:-1], dtype=torch.long).to(image.device)

        text_input = [self.prompt] * image.size(0)

        llm_tokens = self.llm_tokenizer(
            text_input,
            padding="longest",
            return_tensors="pt"
        ).to(image.device)

        with self.maybe_autocast():
            inputs_embeds = self.llm.get_input_embeddings()(llm_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_query, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_query, llm_tokens.attention_mask], dim=1)

            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

        outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        return output_text

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llm_model = cfg.get("llm_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 56)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            llm_model=llm_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
        )

        if cfg.get("load_finetuned", False) or cfg.get("load_pretrained", False):
            model.load_checkpoint_from_config(cfg)

        return model
