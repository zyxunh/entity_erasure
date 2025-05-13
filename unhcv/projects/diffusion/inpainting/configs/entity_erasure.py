unet_extra_config = dict(main_branch_requires_grad=False,
                         original_attn_unite_entity=False,
                         inject_place="cross_attention", share_query=True, train_query=False)
unet_cls = "attention_controlnet"
ldm_wrap_config = dict(scheduler_config=dict(cls="DDIMScheduler"))
controlnet_cond_drop_ratio = 0.05
