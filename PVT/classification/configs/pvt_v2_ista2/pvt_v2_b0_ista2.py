cfg = dict(
    model='pvt_v2_b0',
    drop_path=0.1,
    clip_grad=None,
    # ista2_args
    ista2_args=dict(
        qk_norm=True,
        v_norm=True,
        ista2_method='v',
        attn_inter_topk=1,
        rep_tkn_type='mean',
    )
)
