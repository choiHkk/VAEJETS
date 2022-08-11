from .modules import (
    VarianceAdaptor, 
    MultiPeriodDiscriminator, 
    Generator, 
    ResidualCouplingBlock, 
    PosteriorEncoder, 
)
from utils.tools import get_mask_from_lengths, partial
from transformer import Encoder
import torch.nn as nn
import torch
import json
import os



class VAEJETSSynthesizer(nn.Module):
    def __init__(self, preprocess_config, model_configs, train_config):
        super(VAEJETSSynthesizer, self).__init__()
        self.preprocess_config = preprocess_config
        self.synthesizer_config = model_configs[0]
        self.generator_config = model_configs[1]
        self.train_config = train_config
        
        self.generator_config["num_mels"] = self.synthesizer_config["transformer"]["encoder_hidden"]
        self.generator_config["gin_channels"] = self.synthesizer_config["speaker_encoder"]["speaker_encoder_hidden"]
        speaker_ids_path = os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")
        assert os.path.isfile(speaker_ids_path)
        with open(speaker_ids_path, "r", encoding='utf8') as f:
            n_speaker = len(json.load(f))
        self.speaker_emb = nn.Embedding(
            n_speaker,
            self.synthesizer_config["speaker_encoder"]["speaker_encoder_hidden"],
        )

        self.encoder = Encoder(self.synthesizer_config)
        self.posterior_encoder = PosteriorEncoder(
            self.preprocess_config, self.synthesizer_config)
        self.variance_adaptor = VarianceAdaptor(
            self.preprocess_config, self.synthesizer_config, self.train_config)
        self.flow = ResidualCouplingBlock(self.synthesizer_config)
        self.generator = Generator(self.generator_config)


    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None, 
        mel_lens=None,
        max_mel_len=None,
        cwt_spec_targets=None,
        cwt_mean_target=None,
        cwt_std_target=None,
        uv=None,
        e_targets=None, 
        attn_priors=None, 
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        step=None, 
        gen=False, 
        noise_scale=1.0, 
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )
        x = self.encoder(texts, src_masks)
        g = self.speaker_emb(speakers).unsqueeze(-1)
        
        (
            m_p, 
            logs_p, 
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks, 
            attn_h, 
            attn_s, 
            attn_logprob
        ) = self.variance_adaptor(
            x, 
            src_lens, 
            src_masks,
            mels, 
            mel_lens, 
            mel_masks, 
            max_mel_len, 
            cwt_spec_targets,
            cwt_mean_target,
            cwt_std_target,
            uv,
            e_targets, 
            attn_priors, 
            g, 
            p_control,
            e_control,
            d_control,
            step, 
            gen, 
        )
        
        if not gen:
            z, m_q, logs_q, _ = self.posterior_encoder(mels, (~mel_masks).float().unsqueeze(1), g=g)
            z_p = self.flow(z, (~mel_masks).float().unsqueeze(1), g=g)
            z, indices = partial(
                y=z, 
                segment_size=self.generator_config["segment_size"], 
                hop_size=self.preprocess_config["preprocessing"]["stft"]["hop_length"])
        else:
            m_q, logs_q, indices = None, None, [None, None]
            z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale 
            z = self.flow(z_p, (~mel_masks).float().unsqueeze(1), g=g, reverse=True)
        
        wav = self.generator(z, g=g)

        return (
            wav,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            indices, 
            src_lens,
            mel_lens,
            attn_h, 
            attn_s, 
            attn_logprob, 
            z_p, 
            m_p, 
            logs_p, 
            m_q, 
            logs_q
        )
    
    def voice_conversion(self, mels, mel_lens, max_mel_len, sid_src, sid_tgt):
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )
        g_src = self.speaker_emb(sid_src).unsqueeze(-1)
        g_tgt = self.speaker_emb(sid_tgt).unsqueeze(-1)
        z, m_q, logs_q, y_mask = self.posterior_encoder(mels, (~mel_masks).float().unsqueeze(1), g=g_src)
        z_p = self.flow(z, y_mask, g=g_src)
        z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
        o_hat = self.generator(z_hat * y_mask, g=g_tgt)
        return o_hat, y_mask, (z, z_p, z_hat)
