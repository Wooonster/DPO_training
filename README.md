# DPO

![rlhf&dpo](https://raw.githubusercontent.com/chunhuizhang/personal_chatgpt/7d7e162872ba5917192d0c8f4d361682138ddaed/tutorials/trl_hf/imgs/rlhf_dpo.png)

## Loss

$$
\begin{align}
\mathcal{L}_{\text{DPO}}(\pi_{\theta};\pi_{\text{ref}}) &=-\mathbb{E}_{x,y_{w},y_{l}\sim \mathcal{D}}\left[ \log\sigma\left( \beta \log \frac{\pi_{\theta}(y_{w}|x)}{\pi_{\text{ref}}(y_{w}|x)} - \beta \log \frac{\pi_{\theta}(y_{l}|x)}{\pi_{\text{ref}}(y_{l}|x)} \right) \right]  \\
&= -\mathbb{E}_{x,y_{w},y_{l}\sim \mathcal{D}}\left[ \log\sigma \bigg(\beta \right( ( \log\pi (y_w|x) - \log\pi (y_l|x) ) -  ( \log\pi_{\text{ref}}(y_w|x) - \log\pi_{\text{ref}}(y_l|x) ) \left)\bigg) \right] 
\end{align}
$$


## Acknownledgement
- [bilibili@蓝斯诺特](https://github.com/lansinuote/Simple_LLM_DPO)
- [bilibili@偷星九月333](https://github.com/wyf3/llm_related/blob/main/train_llm_from_scratch/dpo_train.py)